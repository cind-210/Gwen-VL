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


def build_prompt(tokenizer, prompt: str, image_token_count: int) -> str:
    image_placeholder = VISION_START + (IMAGE_PAD * image_token_count) + VISION_END
    messages = [
        {
            "role": "user",
            "content": image_placeholder + prompt,
        }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    parser = argparse.ArgumentParser(description="GWen VLM inference")
    parser.add_argument("--config", default="gwen8k_hybrid", choices=sorted(CONFIG_PRESETS))
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default="model/tokenizer_mini8k")
    parser.add_argument("--vision_model_path", default="models/siglip2-base-p32-256-ve")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="请描述这张图片。")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=30)
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
    model = GWenForCausalLM(config, vision_model_path=args.vision_model_path)
    load_model_weights(model, ckpt, torch.device(args.device), strict=False)
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = model.to(args.device, dtype=dtype).eval()

    image_processor = SiglipImageProcessor.from_pretrained(args.vision_model_path)
    image = Image.open(args.image).convert("RGB")
    pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"].to(args.device, dtype=dtype)
    text = build_prompt(tokenizer, args.prompt, config.image_grid_size * config.image_grid_size)
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
        streamer=streamer,
    )
    response = tokenizer.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    print(f"\n{response}")


if __name__ == "__main__":
    main()
