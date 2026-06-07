"""GWen CLI chat/inference."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import torch
from transformers import TextStreamer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.model_gwen import CONFIG_PRESETS, GWenForCausalLM
from trainer.common import configure_vision_token_ids, config_from_checkpoint, load_checkpoint, load_model_weights, load_tokenizer


def init_model(args):
    tokenizer = load_tokenizer(args.tokenizer_path)
    ckpt = load_checkpoint(args.model_path, map_location="cpu") 
    config = config_from_checkpoint(
        ckpt,
        fallback_name=args.config,
        max_seq_len=args.max_seq_len,
        tokenizer_vocab_size=len(tokenizer),
    )
    configure_vision_token_ids(config, tokenizer)
    config.vlm_rope_type = args.vlm_rope_type
    config.rotary_dim = args.rotary_dim
    model = GWenForCausalLM(config)
    load_info = load_model_weights(model, ckpt, torch.device(args.device), strict=False)
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = model.to(args.device, dtype=dtype).eval()
    return model, tokenizer, config, load_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="gwen8k_hybrid", choices=sorted(CONFIG_PRESETS))
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default="model/tokenizer_mini8k")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6) # 温度参数，控制生成文本的随机程度
    parser.add_argument("--top_p", type=float, default=0.9) # top-p采样的概率阈值，top-p=0.9表示只在累积概率达到90%的词汇中进行采样，这种方法可以动态调整候选词汇的数量，通常比固定的top-k更能保持生成文本的多样性和质量
    parser.add_argument("--top_k", type=int, default=30) # top_k=0表示不使用top-k采样，直接在所有词汇上进行采样，top_k>0表示只在概率最高的k个词汇中进行采样
    parser.add_argument("--do_sample", type=bool, default=True) # 是否使用采样，默认为True表示使用采样，设置为False表示使用贪心解码（greedy decoding），即每次选择概率最高的词汇作为输出，这种方法通常会生成更确定性的文本，但可能缺乏多样性和创造
    parser.add_argument("--repetition_penalty", type=float, default=1.05) # 重复惩罚系数，默认是1.0，表示不使用重复惩罚，设置大于1.0的值可以增加对重复词汇的惩罚，从而减少生成文本中的重复内容
    parser.add_argument("--vlm_rope_type", default="rope", choices=["mrope", "rope"])
    parser.add_argument("--rotary_dim", type=int, default=64)
    parser.add_argument(
        "--system_prompt",
        default="",
        help="Optional ChatML system prompt. Useful for pinning model identity during evaluation.",
    )
    parser.add_argument("--historys", type=int, default=0)
    parser.add_argument("--show_speed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    prompts = [
        "你好，请问你是谁？",
        "你知道什么是人工智能吗？",
        "你有什么特长？",
        "用 Python 写一个斐波那契数列的函数。"
    ]
    model, tokenizer, config, load_info = init_model(args)
    stats = model.get_param_breakdown()
    print(
        f"GWen-{args.config} | {stats['total']/1e6:.1f}M params | "
        f"hidden={config.hidden_size} layers={config.num_hidden_layers} "
        f"vocab={len(tokenizer)} backend={config.linear_attention_backend} "
        f"gdn_kernel={config.gdn_kernel_backend} gate={config.gated_attention}"
    )
    missing = len(load_info.get("missing_keys", []))
    unexpected = len(load_info.get("unexpected_keys", []))
    skipped = len(load_info.get("skipped_shape_mismatch", {}))
    print(f"[Load] missing={missing} unexpected={unexpected} skipped_shape={skipped}")
    mode = int(input("[0] 自动测试\n[1] 手动输入\n"))
    prompt_iter = prompts if mode == 0 else iter(lambda: input("\nYou: "), "")
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    conversation = []

    for prompt in prompt_iter:
        torch.manual_seed(random.randint(0, 2**31 - 1))
        if mode == 0:
            print(f"\nYou: {prompt}")
        conversation = conversation[-args.historys:] if args.historys else []
        if args.system_prompt:
            conversation = [{"role": "system", "content": args.system_prompt}] + conversation
        conversation.append({"role": "user", "content": prompt})
        if "pretrain" in os.path.basename(args.model_path).lower():
            text = prompt
        else:
            text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_seq_len).to(args.device)
        print("GWen: ", end="", flush=True)
        t0 = time.time()
        generated = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=args.do_sample,
            repetition_penalty=args.repetition_penalty,
            streamer=streamer,
        )
        elapsed = max(time.time() - t0, 1e-6)
        response = tokenizer.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        if args.show_speed:
            print(f"\n[Speed] {(generated.shape[1] - inputs['input_ids'].shape[1]) / elapsed:.2f} tok/s")


if __name__ == "__main__":
    main()
