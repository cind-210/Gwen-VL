"""
GWen text-only language model.

The model is a compact, readable PyTorch implementation of the Qwen3.5 text
backbone ideas: 3 linear-attention blocks followed by 1 full-attention block,
Partial RoPE, QK-Norm, SwiGLU, and tied token embeddings.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from inspect import signature
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    import causal_conv1d

    _HAS_CAUSAL_CONV1D = True
except ImportError:
    causal_conv1d = None
    _HAS_CAUSAL_CONV1D = False

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk_gated_delta_rule

    _HAS_FLA_GATED_DELTA = True
except Exception:
    fla_chunk_gated_delta_rule = None
    _HAS_FLA_GATED_DELTA = False


@dataclass
class GWenConfig:
    vocab_size: int = 248320
    hidden_size: int = 384
    num_hidden_layers: int = 8
    intermediate_size: int = 1280
    hidden_act: str = "silu"
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02

    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 96
    attention_bias: bool = False
    attn_output_gate: bool = True
    gated_attention: str = "sigmoid"  # none, sigmoid, headwise, elementwise

    linear_num_key_heads: int = 6
    linear_num_value_heads: int = 6
    linear_key_head_dim: int = 64
    linear_value_head_dim: int = 64
    linear_conv_kernel_dim: int = 4
    linear_chunk_size: int = 64
    linear_attention_backend: str = "gdn"
    gdn_kernel_backend: str = "auto"  # auto, fla, torch

    full_attention_interval: int = 4
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    use_cache: bool = True

    def __post_init__(self) -> None:
        self.validate()

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def full_attention_q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def full_attention_kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def linear_key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def layer_types(self) -> List[str]:
        return [
            "full_attention" if (i + 1) % self.full_attention_interval == 0 else "linear_attention"
            for i in range(self.num_hidden_layers)
        ]

    def validate(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.linear_num_key_heads != self.linear_num_value_heads:
            raise ValueError("GWen expects equal linear key/value heads for stable GDN state layout")
        if self.linear_key_head_dim != self.linear_value_head_dim:
            raise ValueError("GWen expects equal linear key/value head dims for gated delta recurrence")
        if int(self.head_dim * self.partial_rotary_factor) % 2 != 0:
            raise ValueError("full attention rotary dim must be even")
        if int(self.linear_key_head_dim * self.partial_rotary_factor) % 2 != 0:
            raise ValueError("linear attention rotary dim must be even")
        if self.gated_attention not in {"none", "sigmoid", "headwise", "elementwise"}:
            raise ValueError("gated_attention must be one of: none, sigmoid, headwise, elementwise")
        if self.linear_attention_backend not in {"gdn", "full"}:
            raise ValueError("linear_attention_backend must be one of: gdn, full")
        if self.gdn_kernel_backend not in {"auto", "fla", "torch"}:
            raise ValueError("gdn_kernel_backend must be one of: auto, fla, torch")

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "GWenConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def gwen8k_hybrid(cls) -> "GWenConfig":
        return cls(
            vocab_size=8192,
            hidden_size=768,
            num_hidden_layers=8,
            intermediate_size=2816,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=96,
            linear_num_key_heads=8,
            linear_num_value_heads=8,
            linear_key_head_dim=96,
            linear_value_head_dim=96,
            full_attention_interval=4,
            attn_output_gate=True,
            gated_attention="sigmoid",
            linear_attention_backend="gdn",
            max_position_embeddings=8192,
        )

    @classmethod
    def gwen8k_hybrid_128m(cls) -> "GWenConfig":
        config = cls.gwen8k_hybrid()
        config.num_hidden_layers = 12
        return config

    @classmethod
    def gwen8k_hybrid_256m(cls) -> "GWenConfig":
        return cls(
            vocab_size=8192,
            hidden_size=1024,
            num_hidden_layers=16,
            intermediate_size=3328,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=128,
            linear_num_key_heads=8,
            linear_num_value_heads=8,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            full_attention_interval=4,
            attn_output_gate=True,
            gated_attention="sigmoid",
            linear_attention_backend="gdn",
            max_position_embeddings=8192,
        )

CONFIG_PRESETS = {
    "gwen8k_hybrid": GWenConfig.gwen8k_hybrid,
    "gwen8k_hybrid_128m": GWenConfig.gwen8k_hybrid_128m,
    "gwen8k_hybrid_256m": GWenConfig.gwen8k_hybrid_256m,
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


def precompute_freqs_cis(dim: int, max_len: int, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int, # 对 q, k 进行 rotary pos emb，其默认至为0.25，即对 head_dim 的前 1/4 维度进行旋转位置编码，剩余维度保持不变。这种部分旋转的位置编码方法可以在保持模型性能的同时，减少计算开销和内存使用，特别是在 head_dim 较大的情况下。
) -> Tuple[torch.Tensor, torch.Tensor]:
    if rotary_dim <= 0:
        return q, k
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    cos = cos[..., : rotary_dim // 2].repeat_interleave(2, dim=-1).unsqueeze(2)
    sin = sin[..., : rotary_dim // 2].repeat_interleave(2, dim=-1).unsqueeze(2)
    q_rot = (q_rot.float() * cos + rotate_half(q_rot.float()) * sin).to(q.dtype)
    k_rot = (k_rot.float() * cos + rotate_half(k_rot.float()) * sin).to(k.dtype)
    return torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, seq_len, num_kv_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, seq_len, num_kv_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, seq_len, num_kv_heads * n_rep, head_dim)


def causal_depthwise_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    cache: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, channels = x.shape
    kernel = weight.shape[-1]
    x_t = x.transpose(1, 2).contiguous()

    if _HAS_CAUSAL_CONV1D and cache is None:
        out = causal_conv1d.causal_conv1d_fn(x_t, weight.squeeze(1), bias, activation=None)
        new_cache = F.pad(x_t, (max(kernel - seq_len, 0), 0))[:, :, -kernel + 1 :].detach()
        return out.transpose(1, 2).contiguous(), new_cache

    if cache is not None:
        x_cat = torch.cat([cache.to(x_t.dtype), x_t], dim=-1)
    else:
        x_cat = F.pad(x_t, (kernel - 1, 0))
    out = F.conv1d(x_cat, weight.to(x_t.dtype), bias.to(x_t.dtype), groups=channels)
    out = out[:, :, -seq_len:]
    new_cache = x_cat[:, :, -kernel + 1 :].detach()
    return out.transpose(1, 2).contiguous(), new_cache


class FullAttention(nn.Module):
    def __init__(self, config: GWenConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_key_value_groups
        self.rotary_dim = int(config.head_dim * config.partial_rotary_factor) 

        self.q_proj = nn.Linear(config.hidden_size, config.full_attention_q_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.full_attention_kv_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.full_attention_kv_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.full_attention_q_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.gated_attention = config.gated_attention if config.attn_output_gate else "none"
        if self.gated_attention == "headwise":
            self.g_proj = nn.Linear(config.hidden_size, config.num_attention_heads, bias=config.attention_bias)
        elif self.gated_attention in {"elementwise", "sigmoid"}:
            self.g_proj = nn.Linear(config.hidden_size, config.full_attention_q_dim, bias=config.attention_bias)
        else:
            self.g_proj = None
        self.dropout = config.dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        batch, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin, self.rotary_dim)

        if past_key_value is not None:
            k = torch.cat([past_key_value["key"].to(k.dtype), k], dim=1)
            v = torch.cat([past_key_value["value"].to(v.dtype), v], dim=1)

        present = {"key": k.detach(), "value": v.detach()} if use_cache else None
        past_len = k.shape[1] - seq_len

        k = repeat_kv(k, self.num_key_value_groups).transpose(1, 2)
        v = repeat_kv(v, self.num_key_value_groups).transpose(1, 2)
        q = q.transpose(1, 2)

        if attention_mask is None and past_len == 0:
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            if attention_mask is not None:
                attention_mask = attention_mask[:, None, None, :].to(torch.bool)
                attn_bias = torch.zeros(batch, 1, seq_len, k.shape[-2], dtype=q.dtype, device=q.device)
                attn_bias = attn_bias.masked_fill(~attention_mask[:, :, :, -k.shape[-2] :], torch.finfo(q.dtype).min)
            else:
                attn_bias = None

            causal = torch.ones(seq_len, k.shape[-2], dtype=torch.bool, device=q.device).tril(diagonal=past_len)
            causal_bias = torch.zeros(seq_len, k.shape[-2], dtype=q.dtype, device=q.device)
            causal_bias = causal_bias.masked_fill(~causal, torch.finfo(q.dtype).min)
            attn_mask = causal_bias[None, None, :, :] if attn_bias is None else attn_bias + causal_bias[None, None, :, :]

            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        if self.g_proj is not None:
            gate = self.g_proj(hidden_states)
            if self.gated_attention == "headwise":
                gate = gate.unsqueeze(-1).expand(-1, -1, -1, self.head_dim).reshape(batch, seq_len, -1)
                attn_output = F.silu(attn_output) * gate
            elif self.gated_attention == "elementwise":
                attn_output = F.silu(attn_output) * gate
            elif self.gated_attention == "sigmoid":
                attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output), present


class GatedDeltaNet(nn.Module):
    def __init__(self, config: GWenConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.linear_num_key_heads
        self.head_dim = config.linear_key_head_dim
        self.value_dim = config.linear_value_head_dim
        self.qkv_dim = self.num_heads * self.head_dim
        self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        channels = 3 * self.qkv_dim

        self.in_proj_qkv = nn.Linear(config.hidden_size, channels, bias=config.attention_bias)
        self.in_proj_z = nn.Linear(config.hidden_size, self.qkv_dim, bias=config.attention_bias)
        self.in_proj_b = nn.Linear(config.hidden_size, self.num_heads, bias=config.attention_bias)
        self.in_proj_a = nn.Linear(config.hidden_size, self.num_heads, bias=config.attention_bias)
        self.conv_weight = nn.Parameter(torch.randn(channels, 1, config.linear_conv_kernel_dim) * 0.02)
        self.conv_bias = nn.Parameter(torch.zeros(channels))
        self.A_log = nn.Parameter(torch.empty(self.num_heads).uniform_(-0.1, 0.1))
        self.dt_bias = nn.Parameter(torch.zeros(self.num_heads))
        self.norm = RMSNorm(self.value_dim, config.rms_norm_eps)
        self.out_proj = nn.Linear(self.qkv_dim, config.hidden_size, bias=config.attention_bias)
        self.dropout = nn.Dropout(config.dropout)
        self._warned_missing_fla = False
        self._fla_head_first_kwarg = None
        self._fla_state_layout_kwarg = None
        if _HAS_FLA_GATED_DELTA:
            fla_params = signature(fla_chunk_gated_delta_rule).parameters
            if "head_first" in fla_params:
                self._fla_head_first_kwarg = "head_first"
            if "transpose_state_layout" in fla_params:
                self._fla_state_layout_kwarg = "transpose_state_layout"
            elif "state_v_first" in fla_params:
                self._fla_state_layout_kwarg = "state_v_first"

    def _recurrent_delta_rule(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        for t in range(q.shape[1]):
            decay_t = decay[:, t, :, None, None].float()
            beta_t = beta[:, t, :, None].float()
            k_t = k[:, t].float()
            v_t = v[:, t].float()
            q_t = q[:, t].float()

            # Gated delta rule in FLA's default state layout: [batch, heads, key_dim, value_dim].
            predicted_v = torch.einsum("bhk,bhkv->bhv", k_t, state)
            delta_v = (v_t - predicted_v) * beta_t
            state = decay_t * state + torch.einsum("bhk,bhv->bhkv", k_t, delta_v)
            outputs.append(torch.einsum("bhk,bhkv->bhv", q_t, state).to(q.dtype))
        return torch.stack(outputs, dim=1), state

    def _chunk_delta_rule(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._use_fla_kernel(q, state):
            return self._fla_chunk_delta_rule(q, k, v, decay, beta, state)

        chunks = []
        chunk_size = max(1, self.config.linear_chunk_size)
        for start in range(0, q.shape[1], chunk_size):
            end = min(start + chunk_size, q.shape[1])
            chunk, state = self._recurrent_delta_rule(
                q[:, start:end],
                k[:, start:end],
                v[:, start:end],
                decay[:, start:end],
                beta[:, start:end],
                state,
            )
            chunks.append(chunk)
        return torch.cat(chunks, dim=1), state

    def _use_fla_kernel(self, q: torch.Tensor, state: torch.Tensor) -> bool:
        backend = self.config.gdn_kernel_backend
        if backend == "torch":
            return False
        if backend == "fla" and not _HAS_FLA_GATED_DELTA:
            raise ImportError(
                "gdn_kernel_backend='fla' requires flash-linear-attention. "
                "Install it in the training environment, then rerun."
            )
        if not (self.training and q.is_cuda and q.shape[1] > 1):
            return False
        if _HAS_FLA_GATED_DELTA:
            return True
        if not self._warned_missing_fla:
            warnings.warn(
                "flash-linear-attention is not installed; GatedDeltaNet is using the slow PyTorch reference loop. "
                "Install flash-linear-attention or train with --linear_attention_backend full for speed.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_missing_fla = True
        return False

    def _fla_chunk_delta_rule(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_dtype = q.dtype
        # FLA keeps the g-derived A matrix in fp32 inside the Triton kernel.
        # v/beta must therefore also be fp32, otherwise tl.dot(A, v) fails with
        # "Both operands must be same dtype" under bf16/fp16 autocast.
        q = q.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        beta = beta.float().contiguous()
        g = torch.log(decay.float().clamp_min(1e-6)).contiguous()
        fla_kwargs = {
            "g": g,
            "beta": beta,
            "initial_state": state.float().contiguous(),
            "output_final_state": True,
            "use_qk_l2norm_in_kernel": False,
            "scale": 1.0,
        }
        if self._fla_head_first_kwarg is not None:
            fla_kwargs[self._fla_head_first_kwarg] = False
        if self._fla_state_layout_kwarg is not None:
            fla_kwargs[self._fla_state_layout_kwarg] = False

        try:
            output, final_state = fla_chunk_gated_delta_rule(q, k, v, **fla_kwargs)
        except TypeError as exc:
            unknown_scale = "scale" in str(exc)
            unknown_head_first = self._fla_head_first_kwarg is not None and self._fla_head_first_kwarg in str(exc)
            unknown_layout = self._fla_state_layout_kwarg is not None and self._fla_state_layout_kwarg in str(exc)
            if not (unknown_scale or unknown_head_first or unknown_layout):
                raise
            raise RuntimeError(
                "GWen GDN requires a recent flash-linear-attention chunk_gated_delta_rule "
                "with scale=1.0, head_first=False, and [B,H,K,V] state layout support."
            ) from exc
        return output.to(output_dtype), final_state.float()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        batch, seq_len, _ = hidden_states.shape
        conv_cache = past_key_value.get("conv") if past_key_value is not None else None
        state = past_key_value.get("state") if past_key_value is not None else None

        qkv = self.in_proj_qkv(hidden_states)
        qkv, new_conv_cache = causal_depthwise_conv1d(qkv, self.conv_weight, self.conv_bias, conv_cache)
        qkv = F.silu(qkv)
        q, k, v = qkv.split(self.qkv_dim, dim=-1)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_heads, self.value_dim)
        z = self.in_proj_z(hidden_states).view(batch, seq_len, self.num_heads, self.value_dim)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin, self.rotary_dim)
        q = F.normalize(q.float(), p=2, dim=-1).to(hidden_states.dtype)
        k = F.normalize(k.float(), p=2, dim=-1).to(hidden_states.dtype)

        a = self.in_proj_a(hidden_states) + self.dt_bias
        g = -torch.exp(self.A_log.float()).view(1, 1, -1) * F.softplus(a.float())
        decay = torch.exp(g).clamp(min=0.0, max=1.0).to(hidden_states.dtype)
        beta = torch.sigmoid(self.in_proj_b(hidden_states)).to(hidden_states.dtype)

        if state is None:
            state = torch.zeros(
                batch,
                self.num_heads,
                self.head_dim,
                self.value_dim,
                dtype=torch.float32,
                device=hidden_states.device,
            )
        else:
            state = state.float()

        if self.training and seq_len > 1:
            output, state = self._chunk_delta_rule(q, k, v, decay, beta, state)
        else:
            output, state = self._recurrent_delta_rule(q, k, v, decay, beta, state)
        output = self.norm(output * F.silu(z))
        output = output.reshape(batch, seq_len, self.qkv_dim)
        output = self.dropout(self.out_proj(output))
        present = {"state": state.detach(), "conv": new_conv_cache.detach()} if use_cache else None
        return output, present


class FeedForward(nn.Module):
    def __init__(self, config: GWenConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class GWenDecoderLayer(nn.Module):
    def __init__(self, config: GWenConfig, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = (
            FullAttention(config)
            if layer_type == "full_attention" or config.linear_attention_backend == "full"
            else GatedDeltaNet(config)
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        residual = hidden_states
        hidden_states, present = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present


class GWenModel(nn.Module):
    def __init__(self, config: GWenConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([GWenDecoderLayer(config, kind) for kind in config.layer_types])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        fa_rotary = int(config.head_dim * config.partial_rotary_factor)
        la_rotary = int(config.linear_key_head_dim * config.partial_rotary_factor)
        max_rotary = max(fa_rotary, la_rotary)
        cos, sin = precompute_freqs_cis(max_rotary, config.max_position_embeddings, config.rope_theta)
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Optional[Dict[str, torch.Tensor]]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Optional[Dict[str, torch.Tensor]]]]]:
        batch, seq_len = input_ids.shape
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        if position_ids is None:
            past_len = 0
            if past_key_values:
                for item in past_key_values:
                    if item is not None and "key" in item:
                        past_len = item["key"].shape[1]
                        break
            position_ids = torch.arange(past_len, past_len + seq_len, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch, -1)
        cos = self.freqs_cos[position_ids].to(hidden_states.device)
        sin = self.freqs_sin[position_ids].to(hidden_states.device)
        presents = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            past = past_key_values[idx] if past_key_values is not None else None
            hidden_states, present = layer(
                hidden_states,
                (cos, sin),
                attention_mask=attention_mask,
                past_key_value=past,
                use_cache=use_cache,
            )
            if use_cache:
                presents.append(present)

        return self.norm(hidden_states), presents


class GWenForCausalLM(nn.Module):
    def __init__(self, config: Optional[GWenConfig] = None):
        super().__init__()
        self.config = config or GWenConfig.gwen8k_hybrid()
        self.model = GWenModel(self.config)
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Optional[Dict[str, torch.Tensor]]]] = None,
        use_cache: bool = False,
        logits_to_keep: int = 0,
        **_: Dict,
    ) -> Dict[str, torch.Tensor]:
        hidden_states, presents = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        if logits_to_keep > 0:
            hidden_for_logits = hidden_states[:, -logits_to_keep:, :]
        else:
            hidden_for_logits = hidden_states
        logits = F.linear(hidden_for_logits, self.lm_head.weight)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": presents,
            "hidden_states": hidden_states,
        }

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.85,
        top_p: float = 0.9,
        top_k: int = 30,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        streamer=None,
        use_cache: bool = True,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        input_ids = input_ids if input_ids is not None else inputs
        if input_ids is None:
            raise ValueError("input_ids or inputs must be provided")
        input_ids = input_ids.repeat(num_return_sequences, 1)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat(num_return_sequences, 1)

        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        total_len = input_ids.shape[1]
        if streamer is not None:
            streamer.put(input_ids.cpu())

        for _ in range(max_new_tokens):
            step_input = input_ids[:, -1:] if past_key_values is not None else input_ids
            position_ids = torch.arange(total_len - step_input.shape[1], total_len, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)
            outputs = self(
                input_ids=step_input,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            logits = outputs["logits"][:, -1, :]
            if temperature and temperature > 0:
                logits = logits / temperature
            if repetition_penalty != 1.0:
                for row in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[row])
                    scores = logits[row, seen]
                    logits[row, seen] = torch.where(scores > 0, scores / repetition_penalty, scores * repetition_penalty)
            if top_k and top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1]
                logits = logits.masked_fill(logits < kth[:, None], -float("inf"))
            if top_p and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                mask = torch.cumsum(probs, dim=-1) > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                logits = logits.masked_fill(mask.scatter(1, sorted_idx, mask), -float("inf"))
            if do_sample:
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None:
                fill_token = pad_token_id if pad_token_id is not None else eos_token_id
                next_token = torch.where(finished[:, None], next_token.new_full(next_token.shape, fill_token), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
            past_key_values = outputs["past_key_values"] if use_cache else None
            total_len += 1
            if streamer is not None:
                streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all():
                    break
        if streamer is not None:
            streamer.end()
        return input_ids

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_param_breakdown(self) -> Dict[str, int]:
        embed = self.model.embed_tokens.weight.numel()
        total = self.get_num_params()
        lm_head = 0 if self.config.tie_word_embeddings else self.lm_head.weight.numel()
        return {"total": total, "embedding": embed, "lm_head": lm_head, "body": total - embed - lm_head}


def get_config(name: str) -> GWenConfig:
    if name not in CONFIG_PRESETS:
        raise KeyError(f"Unknown config '{name}'. Choose from {sorted(CONFIG_PRESETS)}")
    return CONFIG_PRESETS[name]()


def print_accel_info() -> None:
    print(f"  causal-conv1d: {'OK' if _HAS_CAUSAL_CONV1D else 'not installed'}")
    print(f"  flash-linear-attention: {'OK' if _HAS_FLA_GATED_DELTA else 'not installed'}")
    print(f"  torch cuda:    {'OK' if torch.cuda.is_available() else 'CPU only'}")


if __name__ == "__main__":
    for name in [
        "gwen8k_hybrid",
        "gwen8k_hybrid_128m",
        "gwen8k_hybrid_256m",
    ]:
        cfg = get_config(name)
        model = GWenForCausalLM(cfg)
        stats = model.get_param_breakdown()
        print(f"==========={name}==============")
        print(f"{name}: layers={cfg.num_hidden_layers} types={cfg.layer_types}")
        print(
            f"  params={stats['total']/1e6:.1f}M "
            f"embedding={stats['embedding']/1e6:.1f}M body={stats['body']/1e6:.1f}M"
        )
        if name == "gwen8k_hybrid":
            x = torch.randint(0, min(cfg.vocab_size, 1024), (2, 16))
            out = model(x, labels=x)
            print(f"  smoke loss={out['loss'].item():.4f}")
