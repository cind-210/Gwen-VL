"""
GWen text-only language model.

The model is a compact, readable PyTorch implementation of the Qwen3.5 text
backbone ideas: 3 linear-attention blocks followed by 1 full-attention block,
3D M-RoPE on full-attention layers, QK-Norm, SwiGLU, and tied token embeddings.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from inspect import signature
from typing import Dict, List, Optional, Tuple, Union

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
    vocab_size: int = 8192
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    intermediate_size: int = 3584
    hidden_act: str = "silu"
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02

    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 256
    attention_bias: bool = False
    attn_output_gate: bool = True
    gated_attention: str = "sigmoid"  # none, sigmoid, headwise, elementwise

    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    linear_chunk_size: int = 64
    linear_attention_backend: str = "gdn"
    gdn_kernel_backend: str = "auto"  # auto, fla, torch

    full_attention_interval: int = 4
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    use_cache: bool = True

    vision_model_name: str = "gongjy/siglip2-base-p32-256-ve"
    image_token_id: int = -1
    vision_start_token_id: int = -1
    vision_end_token_id: int = -1
    image_size: int = 256
    vision_patch_size: int = 32
    image_grid_size: int = 8
    vision_hidden_size: int = 768

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
            raise ValueError("full attention rotary dim must be even for M-RoPE")
        if int(self.head_dim * self.partial_rotary_factor) < 6:
            raise ValueError("full attention rotary dim must provide at least one pair for each M-RoPE axis")
        if self.image_grid_size != self.image_size // self.vision_patch_size:
            raise ValueError("image_grid_size must equal image_size // vision_patch_size")
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
            hidden_size=1024,
            num_hidden_layers=24,
            intermediate_size=3584,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=256,
            linear_num_key_heads=16,
            linear_num_value_heads=16,
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


def split_mrope_sections(pair_count: int) -> Tuple[int, int, int]:
    base = pair_count // 3
    remainder = pair_count % 3
    return base + (1 if remainder > 0 else 0), base + (1 if remainder > 1 else 0), base


def apply_axis_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pair_count: int,
) -> torch.Tensor:
    cos = cos[..., :pair_count].repeat_interleave(2, dim=-1).unsqueeze(2)
    sin = sin[..., :pair_count].repeat_interleave(2, dim=-1).unsqueeze(2)
    return x.float() * cos + rotate_half(x.float()) * sin


def apply_3d_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_t: torch.Tensor,
    sin_t: torch.Tensor,
    cos_h: torch.Tensor,
    sin_h: torch.Tensor,
    cos_w: torch.Tensor,
    sin_w: torch.Tensor,
    mrope_sections: Tuple[int, int, int],
    rotary_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if rotary_dim <= 0:
        return q, k
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    t_pairs, h_pairs, w_pairs = mrope_sections
    q_t, q_h, q_w = torch.split(q_rot, [2 * t_pairs, 2 * h_pairs, 2 * w_pairs], dim=-1)
    k_t, k_h, k_w = torch.split(k_rot, [2 * t_pairs, 2 * h_pairs, 2 * w_pairs], dim=-1)

    q_t = apply_axis_rotary(q_t, cos_t, sin_t, t_pairs).to(q.dtype)
    k_t = apply_axis_rotary(k_t, cos_t, sin_t, t_pairs).to(k.dtype)
    q_h = apply_axis_rotary(q_h, cos_h, sin_h, h_pairs).to(q.dtype)
    k_h = apply_axis_rotary(k_h, cos_h, sin_h, h_pairs).to(k.dtype)
    q_w = apply_axis_rotary(q_w, cos_w, sin_w, w_pairs).to(q.dtype)
    k_w = apply_axis_rotary(k_w, cos_w, sin_w, w_pairs).to(k.dtype)
    return torch.cat((q_t, q_h, q_w, q_pass), dim=-1), torch.cat((k_t, k_h, k_w, k_pass), dim=-1)


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
        self.mrope_sections = split_mrope_sections(self.rotary_dim // 2)

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
        position_embeddings: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
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
        cos_t, sin_t, cos_h, sin_h, cos_w, sin_w = position_embeddings
        q, k = apply_3d_rotary_pos_emb(q, k, cos_t, sin_t, cos_h, sin_h, cos_w, sin_w, self.mrope_sections, self.rotary_dim)

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
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
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
        position_embeddings: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
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
        max_axis_pairs = max(split_mrope_sections(fa_rotary // 2))
        cos, sin = precompute_freqs_cis(2 * max_axis_pairs, config.max_position_embeddings, config.rope_theta)
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Optional[Dict[str, torch.Tensor]]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Optional[Dict[str, torch.Tensor]]]]]:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided")
            inputs_embeds = self.embed_tokens(input_ids)
        batch, seq_len, _ = inputs_embeds.shape
        hidden_states = self.dropout(inputs_embeds)
        if position_ids is None:
            past_len = 0
            if past_key_values:
                for item in past_key_values:
                    if item is not None and "key" in item:
                        past_len = item["key"].shape[1]
                        break
            position_ids = torch.arange(past_len, past_len + seq_len, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(batch, -1)
        if position_ids.dim() == 2:
            position_ids = torch.stack((position_ids, position_ids, position_ids), dim=-1)
        cos_t = self.freqs_cos[position_ids[..., 0]].to(hidden_states.device)
        sin_t = self.freqs_sin[position_ids[..., 0]].to(hidden_states.device)
        cos_h = self.freqs_cos[position_ids[..., 1]].to(hidden_states.device)
        sin_h = self.freqs_sin[position_ids[..., 1]].to(hidden_states.device)
        cos_w = self.freqs_cos[position_ids[..., 2]].to(hidden_states.device)
        sin_w = self.freqs_sin[position_ids[..., 2]].to(hidden_states.device)
        presents = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            past = past_key_values[idx] if past_key_values is not None else None
            hidden_states, present = layer(
                hidden_states,
                (cos_t, sin_t, cos_h, sin_h, cos_w, sin_w),
                attention_mask=attention_mask,
                past_key_value=past,
                use_cache=use_cache,
            )
            if use_cache:
                presents.append(present)

        return self.norm(hidden_states), presents


class GWenForCausalLM(nn.Module):
    def __init__(self, config: Optional[GWenConfig] = None, vision_model_path: Optional[str] = None):
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
        self.visual = None
        self.vision_projector = None
        self.vision_model_path = vision_model_path or ""
        if vision_model_path is not None:
            self.load_vision_model(vision_model_path)

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
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Optional[Dict[str, torch.Tensor]]]] = None,
        use_cache: bool = False,
        logits_to_keep: int = 0,
        **_: Dict,
    ) -> Dict[str, torch.Tensor]:
        if pixel_values is not None:
            inputs_embeds = self.prepare_inputs_embeds(input_ids, pixel_values)
        if position_ids is None:
            position_ids = self.build_3d_position_ids(input_ids, past_key_values=past_key_values)
        hidden_states, presents = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
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

    def load_vision_model(self, vision_model_path: Optional[str] = None) -> None:
        from transformers import SiglipVisionModel

        self.vision_model_path = vision_model_path or self.config.vision_model_name
        if not self.vision_model_path:
            raise ValueError("vision_model_path or config.vision_model_name must be provided")
        self.visual = SiglipVisionModel.from_pretrained(self.vision_model_path)
        for param in self.visual.parameters():
            param.requires_grad = False
        self.config.vision_hidden_size = int(self.visual.config.hidden_size)
        self.vision_projector = GWenVisionProjector(self.config.vision_hidden_size, self.config.hidden_size)
        self.vision_projector.apply(self._init_weights)

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.visual is None or self.vision_projector is None:
            raise ValueError("Vision model is not loaded; pass vision_model_path when constructing GWenForCausalLM")
        self.visual.eval()
        with torch.no_grad():
            image_hidden = self.visual(pixel_values=pixel_values).last_hidden_state
        expected_tokens = self.config.image_grid_size * self.config.image_grid_size
        if image_hidden.shape[1] != expected_tokens:
            raise ValueError(f"SigLIP2 returned {image_hidden.shape[1]} tokens, expected {expected_tokens}")
        return self.vision_projector(image_hidden)

    def build_3d_position_ids(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Optional[Dict[str, torch.Tensor]]]] = None,
        rope_deltas: Optional[torch.Tensor] = None,
        return_rope_deltas: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch, seq_len = input_ids.shape
        past_len = 0
        if past_key_values:
            for item in past_key_values:
                if item is not None and "key" in item:
                    past_len = item["key"].shape[1]
                    break
        if past_len > 0:
            delta = 0 if rope_deltas is None else rope_deltas.to(input_ids.device).view(batch, 1)
            text_pos = torch.arange(past_len, past_len + seq_len, device=input_ids.device).unsqueeze(0)
            position_ids = text_pos.expand(batch, -1) + delta
            position_ids = torch.stack((position_ids, position_ids, position_ids), dim=-1)
            if return_rope_deltas:
                return position_ids, delta.squeeze(1)
            return position_ids

        text_pos = torch.arange(seq_len, device=input_ids.device)
        position_ids = torch.stack((text_pos, text_pos, text_pos), dim=-1).unsqueeze(0).expand(batch, -1, -1).clone()
        if self.config.image_token_id < 0:
            if return_rope_deltas:
                return position_ids, torch.zeros(batch, dtype=torch.long, device=input_ids.device)
            return position_ids

        image_mask = input_ids.eq(self.config.image_token_id)
        grid = self.config.image_grid_size
        row = torch.arange(grid, device=input_ids.device).repeat_interleave(grid)
        col = torch.arange(grid, device=input_ids.device).repeat(grid)
        temporal = torch.zeros_like(row)
        image_pos = torch.stack((temporal, row, col), dim=-1)
        expected_tokens = grid * grid
        deltas = torch.zeros(batch, dtype=torch.long, device=input_ids.device)
        for batch_idx in range(batch):
            indexes = torch.nonzero(image_mask[batch_idx], as_tuple=False).squeeze(-1)
            if indexes.numel() == 0:
                continue
            if indexes.numel() != expected_tokens:
                raise ValueError(f"Each image sample must contain {expected_tokens} <|image_pad|> tokens")
            if not torch.all(indexes.eq(torch.arange(indexes[0], indexes[0] + expected_tokens, device=input_ids.device))):
                raise ValueError("<|image_pad|> tokens must form one contiguous image block")

            image_start = int(indexes[0].item())
            current_pos = image_start
            position_ids[batch_idx, indexes] = image_pos + current_pos
            after_image = image_start + expected_tokens
            text_after = seq_len - after_image
            if text_after > 0:
                next_pos = current_pos + grid
                tail_pos = torch.arange(next_pos, next_pos + text_after, device=input_ids.device)
                position_ids[batch_idx, after_image:] = torch.stack((tail_pos, tail_pos, tail_pos), dim=-1)
            deltas[batch_idx] = int(position_ids[batch_idx].max().item()) + 1 - seq_len
        if return_rope_deltas:
            return position_ids, deltas
        return position_ids

    def prepare_inputs_embeds(self, input_ids: torch.Tensor, pixel_values: Optional[torch.Tensor]) -> torch.Tensor:
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is None:
            return inputs_embeds
        if self.config.image_token_id < 0:
            raise ValueError("config.image_token_id must be set before VLM forward")
        image_features = self.get_image_features(pixel_values).to(inputs_embeds.dtype)
        image_mask = input_ids.eq(self.config.image_token_id)
        expected_tokens = image_features.shape[1]
        token_counts = image_mask.sum(dim=1)
        if not torch.all(token_counts.eq(expected_tokens)):
            raise ValueError(f"Each sample must contain exactly {expected_tokens} <|image_pad|> tokens")
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[image_mask] = image_features.reshape(-1, image_features.shape[-1])
        return inputs_embeds

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
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
        if pixel_values is not None:
            pixel_values = pixel_values.repeat_interleave(num_return_sequences, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat(num_return_sequences, 1)

        past_key_values = kwargs.pop("past_key_values", None)
        rope_deltas = kwargs.pop("rope_deltas", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        total_len = input_ids.shape[1]
        if streamer is not None:
            streamer.put(input_ids.cpu())

        for _ in range(max_new_tokens):
            first_step = past_key_values is None
            step_input = input_ids if first_step else input_ids[:, -1:]
            step_pixel_values = pixel_values if first_step else None
            if first_step:
                position_ids, rope_deltas = self.build_3d_position_ids(step_input, return_rope_deltas=True)
            else:
                position_ids = self.build_3d_position_ids(
                    step_input,
                    past_key_values=past_key_values,
                    rope_deltas=rope_deltas,
                )
            outputs = self(
                input_ids=step_input,
                pixel_values=step_pixel_values,
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


class GWenVisionProjector(nn.Module):
    def __init__(self, vision_hidden_size: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(vision_hidden_size),
            nn.Linear(vision_hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
        x = torch.randint(0, min(cfg.vocab_size, 1024), (2, 16))
        out = model(x, labels=x)
        print(f"  smoke loss={out['loss'].item():.4f}")
