"""GWen image-only VLM CLI inference."""

from __future__ import annotations

import argparse
import os
import sys

import torch
from PIL import Image
from transformers import SiglipImageProcessor, TextStreamer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset import IMAGE_PAD, VISION_END, VISION_START
from model.model_gwen import CONFIG_PRESETS, GWenForCausalLM
from trainer.common import configure_vision_token_ids, config_from_checkpoint, load_checkpoint, load_model_weights, load_tokenizer


def build_prompt(tokenizer, prompt: str, image_token_count: int, has_image: bool) -> str:
    image_placeholder = VISION_START + (IMAGE_PAD * image_token_count) + VISION_END
    content = image_placeholder + prompt if has_image else prompt
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_image_pixels(image_processor, image_path: str, device: str, dtype: torch.dtype) -> torch.Tensor:
    if not image_path or not os.path.isfile(image_path):
        raise FileNotFoundError("\u56fe\u7247\u8def\u5f84\u4e0d\u5408\u6cd5")
    image = Image.open(image_path).convert("RGB")
    return image_processor(images=image, return_tensors="pt")["pixel_values"].to(device, dtype=dtype)


def generate_once(model, tokenizer, image_processor, config, args, dtype, prompt: str, image_path: str = "") -> str:
    pixel_values = None
    if image_path:
        pixel_values = load_image_pixels(image_processor, image_path, args.device, dtype)
    text = build_prompt(tokenizer, prompt, config.image_grid_size * config.image_grid_size, has_image=pixel_values is not None)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_seq_len).to(args.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generated = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=pixel_values,
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=args.repetition_penalty,
        streamer=streamer,
    )
    return tokenizer.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="GWen VLM inference")
    parser.add_argument("--config", default="gwen8k_hybrid", choices=sorted(CONFIG_PRESETS))
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default="model/tokenizer_mini8k")
    parser.add_argument("--vision_model_path", default="models/siglip2-base-p32-256-ve")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--vlm_rope_type", default="rope", choices=["mrope", "rope"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer_path)
    ckpt = load_checkpoint(args.model_path, map_location="cpu")
    config = config_from_checkpoint(
        ckpt,
        fallback_name=args.config,
        max_seq_len=args.max_seq_len,
        tokenizer_vocab_size=len(tokenizer),
    )
    configure_vision_token_ids(config, tokenizer)
    config.vision_model_name = args.vision_model_path
    config.vlm_rope_type = args.vlm_rope_type
    model = GWenForCausalLM(config, vision_model_path=args.vision_model_path)
    load_model_weights(model, ckpt, torch.device(args.device), strict=False)
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = model.to(args.device, dtype=dtype).eval()

    image_processor = SiglipImageProcessor.from_pretrained(args.vision_model_path)
    if args.prompt:
        image_path = input("\u8bf7\u8f93\u5165\u56fe\u7247\u8def\u5f84: ").strip()
        try:
            response = generate_once(model, tokenizer, image_processor, config, args, dtype, args.prompt, image_path)
        except FileNotFoundError:
            print("\u56fe\u7247\u8def\u5f84\u4e0d\u5408\u6cd5")
            return
        print()
        return

    while True:
        prompt = input("\nYou: ").strip()
        if not prompt:
            break
        image_path = input("\u8bf7\u8f93\u5165\u56fe\u7247\u8def\u5f84: ").strip()
        try:
            response = generate_once(model, tokenizer, image_processor, config, args, dtype, prompt, image_path)
        except FileNotFoundError:
            print("\u56fe\u7247\u8def\u5f84\u4e0d\u5408\u6cd5")
            continue
        print()


if __name__ == "__main__":
    main()
