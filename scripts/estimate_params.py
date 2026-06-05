"""Estimate GWen parameter count for architecture design.

Edit the variables below, then run:

python scripts/estimate_params.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.model_gwen import GWenConfig, GWenForCausalLM


VOCAB_SIZE = 8192
HIDDEN_SIZE = 1024
NUM_HIDDEN_LAYERS = 16
INTERMEDIATE_SIZE = 3328

NUM_ATTENTION_HEADS = 8
NUM_KEY_VALUE_HEADS = 2
HEAD_DIM = 256

LINEAR_NUM_KEY_HEADS = 16
LINEAR_NUM_VALUE_HEADS = 16
LINEAR_KEY_HEAD_DIM = 128
LINEAR_VALUE_HEAD_DIM = 128

FULL_ATTENTION_INTERVAL = 4
PARTIAL_ROTARY_FACTOR = 0.25
MAX_POSITION_EMBEDDINGS = 8192


def main():
    config = GWenConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        intermediate_size=INTERMEDIATE_SIZE,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_key_value_heads=NUM_KEY_VALUE_HEADS,
        head_dim=HEAD_DIM,
        linear_num_key_heads=LINEAR_NUM_KEY_HEADS,
        linear_num_value_heads=LINEAR_NUM_VALUE_HEADS,
        linear_key_head_dim=LINEAR_KEY_HEAD_DIM,
        linear_value_head_dim=LINEAR_VALUE_HEAD_DIM,
        full_attention_interval=FULL_ATTENTION_INTERVAL,
        partial_rotary_factor=PARTIAL_ROTARY_FACTOR,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
    )
    model = GWenForCausalLM(config)
    stats = model.get_param_breakdown()
    rotary_dim = int(config.head_dim * config.partial_rotary_factor)
    print("GWen parameter estimate")
    print(f"vocab_size={config.vocab_size}")
    print(f"hidden_size={config.hidden_size}")
    print(f"num_hidden_layers={config.num_hidden_layers}")
    print(f"intermediate_size={config.intermediate_size}")
    print(f"num_attention_heads={config.num_attention_heads}")
    print(f"num_key_value_heads={config.num_key_value_heads}")
    print(f"head_dim={config.head_dim}")
    print(f"linear_num_key_heads={config.linear_num_key_heads}")
    print(f"linear_num_value_heads={config.linear_num_value_heads}")
    print(f"linear_key_head_dim={config.linear_key_head_dim}")
    print(f"linear_value_head_dim={config.linear_value_head_dim}")
    print(f"partial_rotary_factor={config.partial_rotary_factor}")
    print(f"rotary_dim={rotary_dim}")
    print(f"total_params={stats['total']:,} ({stats['total']/1e6:.2f}M)")
    print(f"embedding_params={stats['embedding']:,} ({stats['embedding']/1e6:.2f}M)")
    print(f"lm_head_params={stats['lm_head']:,} ({stats['lm_head']/1e6:.2f}M)")
    print(f"body_params={stats['body']:,} ({stats['body']/1e6:.2f}M)")


if __name__ == "__main__":
    main()
