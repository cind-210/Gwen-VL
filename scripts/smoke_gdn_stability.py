# 这个代码的作用是检查GWen GatedDeltaNet的语义是否符合预期。
# 具体来说，它会测试GatedDeltaNet的chunk delta rule是否与recurrent delta rule一致。

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.model_gwen import GatedDeltaNet, GWenConfig, GWenForCausalLM


def tiny_config() -> GWenConfig:
    return GWenConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=4,
        intermediate_size=160,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_chunk_size=3,
        full_attention_interval=4,
        max_position_embeddings=64,
        linear_attention_backend="gdn",
        gdn_kernel_backend="torch",
        gated_attention="sigmoid",
        attn_output_gate=True,
    )


def check_gdn_chunk_matches_recurrent() -> None:
    torch.manual_seed(1)
    cfg = tiny_config()
    layer = GatedDeltaNet(cfg).eval()
    batch, seq_len = 2, 7
    q = torch.randn(batch, seq_len, cfg.linear_num_key_heads, cfg.linear_key_head_dim)
    k = torch.randn_like(q)
    v = torch.randn(batch, seq_len, cfg.linear_num_value_heads, cfg.linear_value_head_dim)
    q = torch.nn.functional.normalize(q, p=2, dim=-1)
    k = torch.nn.functional.normalize(k, p=2, dim=-1)
    decay = torch.rand(batch, seq_len, cfg.linear_num_key_heads) * 0.7 + 0.2
    beta = torch.rand(batch, seq_len, cfg.linear_num_key_heads)
    state = torch.randn(batch, cfg.linear_num_key_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim) * 0.01

    recurrent_out, recurrent_state = layer._recurrent_delta_rule(q, k, v, decay, beta, state.clone())
    chunk_out, chunk_state = layer._chunk_delta_rule(q, k, v, decay, beta, state.clone())

    torch.testing.assert_close(chunk_out, recurrent_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(chunk_state, recurrent_state, rtol=1e-5, atol=1e-5)


def check_cache_matches_full_forward() -> None:
    torch.manual_seed(2)
    cfg = tiny_config()
    model = GWenForCausalLM(cfg).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 12))

    with torch.inference_mode():
        full_logits = model(input_ids=input_ids, use_cache=False)["logits"]
        past = None
        step_logits = []
        for idx in range(input_ids.shape[1]):
            out = model(input_ids=input_ids[:, idx : idx + 1], past_key_values=past, use_cache=True)
            past = out["past_key_values"]
            step_logits.append(out["logits"])
        cached_logits = torch.cat(step_logits, dim=1)

    torch.testing.assert_close(cached_logits, full_logits, rtol=2e-4, atol=2e-4)


def check_forward_backward() -> None:
    torch.manual_seed(3)
    cfg = tiny_config()
    model = GWenForCausalLM(cfg).train()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(input_ids=input_ids, labels=input_ids)
    assert out["loss"] is not None
    out["loss"].backward()

    first_grad = model.model.embed_tokens.weight.grad
    assert first_grad is not None
    assert torch.isfinite(first_grad).all()


def check_layer_pattern() -> None:
    cfg = tiny_config()
    assert cfg.layer_types == ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    model = GWenForCausalLM(cfg)
    names = [layer.self_attn.__class__.__name__ for layer in model.model.layers]
    assert names == ["GatedDeltaNet", "GatedDeltaNet", "GatedDeltaNet", "FullAttention"]


def main() -> None:
    check_gdn_chunk_matches_recurrent()
    check_cache_matches_full_forward()
    check_forward_backward()
    check_layer_pattern()
    print("GDN smoke checks passed.")


if __name__ == "__main__":
    main()
