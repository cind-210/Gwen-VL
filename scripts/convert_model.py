# 这个代码的作用是将GWen模型的检查点进行转换和LoRA合并。
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.model_lora import LoRAConfig, apply_lora_to_model, load_lora_state_dict, merge_lora
from model.model_gwen import CONFIG_PRESETS, GWenForCausalLM
from trainer.common import config_from_checkpoint, get_config, load_checkpoint, load_model_weights, load_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="gwen8k_hybrid", choices=sorted(CONFIG_PRESETS))
    parser.add_argument("--tokenizer_path", default="model/tokenizer_mini8k")
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--base_path", default=None)
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", default="pth", choices=["pth", "safetensors"])
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer_path)
    source_path = args.base_path or args.model_path
    source_ckpt = load_checkpoint(source_path, map_location="cpu") if source_path else None
    config = (
        config_from_checkpoint(source_ckpt, args.config, args.max_seq_len, len(tokenizer))
        if source_ckpt is not None
        else get_config(args.config, args.max_seq_len)
    )
    model = GWenForCausalLM(config)
    if args.base_path:
        load_model_weights(model, source_ckpt, torch.device("cpu"), strict=False)
    if args.model_path and not args.base_path:
        load_model_weights(model, source_ckpt, torch.device("cpu"), strict=False)
    if args.lora_path:
        ckpt = torch.load(args.lora_path, map_location="cpu", weights_only=False)
        lora_config = LoRAConfig.from_dict(ckpt.get("lora_config", {})) if isinstance(ckpt, dict) else LoRAConfig()
        apply_lora_to_model(model, lora_config)
        loaded = load_lora_state_dict(model, ckpt)
        merged = merge_lora(model)
        print(f"Loaded {loaded} LoRA params and merged {merged} modules.")

    state = model.state_dict()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if args.format == "safetensors":
        from safetensors.torch import save_file

        save_file(state, args.output)
    else:
        torch.save({"model_state_dict": state, "config": config.to_dict()}, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
