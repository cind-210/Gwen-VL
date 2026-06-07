"""Convert torch.compile-prefixed checkpoint keys back to normal module keys."""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove torch.compile _orig_mod prefixes from a Gwen checkpoint")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    converted = {}
    changed = 0
    for key, value in state.items():
        new_key = key.replace("model._orig_mod.", "model.")
        if new_key != key:
            changed += 1
        converted[new_key] = value

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        inference_ckpt = {"model_state_dict": converted}
        for key in ("config", "step", "epoch"):
            if key in ckpt:
                inference_ckpt[key] = ckpt[key]
        torch.save(inference_ckpt, args.output)
    else:
        torch.save(converted, args.output)

    print(f"converted_keys={changed} total_keys={len(converted)}")
    print(args.output)


if __name__ == "__main__":
    main()
