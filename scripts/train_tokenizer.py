"""MiniMind-style tokenizer training for GWen.

This script intentionally keeps the tokenizer recipe simple:
ByteLevel BPE + Qwen-style special tokens + a small sampled JSONL corpus.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Iterator, List

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoTokenizer


SPECIAL_TOKENS_NUM = 36 # 特殊token的数量

DEFAULT_MAX_SAMPLES = 10000 
# 10000是样本数量的默认值，表示在训练tokenizer时最多使用10000行数据。如果设置为-1，则表示使用所有行数据进行训练。
# 这个参数的作用是控制训练tokenizer时使用的数据量，以平衡训练时间和tokenizer质量之间的关系。

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<|audio_start|>",
    "<|audio_end|>",
    "<|audio_pad|>",
    "<tts_pad>",
    "<tts_text_bos>",
    "<tts_text_eod>",
    "<tts_text_bos_single>",
]

ADDITIONAL_TOKENS = [
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
]

CHAT_TEMPLATE = """{% for message in messages %}<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""


def build_reserved_tokens(special_tokens_num: int) -> List[str]:
    required = SPECIAL_TOKENS + ADDITIONAL_TOKENS
    if special_tokens_num < len(required):
        raise ValueError(f"--special_tokens_num must be at least {len(required)}")
    buffer_tokens = [f"<|buffer{i}|>" for i in range(1, special_tokens_num - len(required) + 1)]
    return required + buffer_tokens


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def extract_text(item: dict) -> str:
    if isinstance(item.get("text"), str):
        return item["text"]

    conversations = item.get("conversations") or item.get("messages")
    if isinstance(conversations, list):
        contents = []
        for message in conversations:
            if not isinstance(message, dict):
                continue
            content = message.get("content", message.get("value", ""))
            if isinstance(content, str) and content.strip():
                contents.append(content)
        if contents:
            return "\n".join(contents)

    return ""


def get_texts(paths: List[Path], max_samples: int, progress_interval: int) -> Iterator[str]:
    count = 0
    for path in paths:
        for item in read_jsonl(path):
            text = extract_text(item)
            if text:
                yield text
            count += 1
            if progress_interval > 0 and count % progress_interval == 0:
                print(f"loaded {count} jsonl rows", flush=True)
            if max_samples > 0 and count >= max_samples:
                return


def patch_tokenizer_json(tokenizer_dir: Path, real_special_tokens: List[str], reserved_tokens: List[str]) -> None:
    tokenizer_json_path = tokenizer_dir / "tokenizer.json"
    with tokenizer_json_path.open("r", encoding="utf-8") as handle:
        tokenizer_data = json.load(handle)
    real_special = set(real_special_tokens)
    for token_info in tokenizer_data.get("added_tokens", []):
        if token_info.get("content") not in real_special:
            token_info["special"] = False
    with tokenizer_json_path.open("w", encoding="utf-8") as handle:
        json.dump(tokenizer_data, handle, ensure_ascii=False, indent=2)

    added_tokens_decoder = {}
    token_to_id = {token["content"]: token["id"] for token in tokenizer_data.get("added_tokens", [])}
    for token in reserved_tokens:
        idx = token_to_id.get(token)
        if idx is None:
            continue
        added_tokens_decoder[str(idx)] = {
            "content": token,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": token in real_special,
        }

    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": added_tokens_decoder,
        "additional_special_tokens": [token for token in real_special_tokens if token != "<|endoftext|>"],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": True,
        "model_max_length": 8192,
        "pad_token": "<|endoftext|>",
        "spaces_between_special_tokens": False,
        "unk_token": "<|endoftext|>",
        "chat_template": CHAT_TEMPLATE,
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    with (tokenizer_dir / "tokenizer_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    with (tokenizer_dir / "special_tokens_map.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "bos_token": "<|im_start|>",
                "eos_token": "<|im_end|>",
                "unk_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
                "additional_special_tokens": config["additional_special_tokens"],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def train_tokenizer(args: argparse.Namespace) -> None:
    input_paths = [Path(item) for item in args.input]
    tokenizer_dir = Path(args.output)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    reserved_tokens = build_reserved_tokens(args.special_tokens_num)
    tokenizer = Tokenizer(models.BPE(unk_token="<|endoftext|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=reserved_tokens,
        min_frequency=args.min_frequency,
    )

    print(
        f"Training MiniMind-style BPE tokenizer: vocab_size={args.vocab_size}, "
        f"max_samples={args.max_samples}, input={','.join(map(str, input_paths))}",
        flush=True,
    )
    tokenizer.train_from_iterator(
        get_texts(input_paths, args.max_samples, args.progress_interval),
        trainer=trainer,
    )
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
    tokenizer.model.save(str(tokenizer_dir))
    patch_tokenizer_json(tokenizer_dir, SPECIAL_TOKENS, reserved_tokens)
    print(f"Tokenizer training completed: {tokenizer_dir}", flush=True)


def eval_tokenizer(tokenizer_dir: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": "你来自哪里？"},
        {"role": "assistant", "content": "我是 GWen，一个小型中文语言模型。"},
        {"role": "user", "content": "什么是人工智能？"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    print("-" * 100)
    print(prompt)
    print("-" * 100)
    print("tokenizer词表长度:", len(tokenizer))
    print("encoder长度:", len(ids))
    print("decoder一致性:", decoded == prompt)
    print(
        "special ids:",
        "pad", tokenizer.pad_token_id,
        "eos", tokenizer.eos_token_id,
        "im_start", tokenizer.convert_tokens_to_ids("<|im_start|>"),
        "im_end", tokenizer.convert_tokens_to_ids("<|im_end|>"),
    )
    print("-" * 100)
    print("压缩率测试（Chars/Tokens）：")
    test_texts = [
        "人工智能是计算机科学的一个分支，它研究如何让机器理解、学习和生成语言。",
        "Python 是一种高级编程语言，常用于数据科学、机器学习和 Web 开发。",
        "Large language models are trained on text to predict the next token in a sequence.",
    ]
    total = 0.0
    for idx, text in enumerate(test_texts, start=1):
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        ratio = len(text) / max(1, token_count)
        total += ratio
        print(f"样本 {idx} | 字符数: {len(text):4} | Tokens: {token_count:4} | 压缩率: {ratio:.2f}")
    print(f"平均压缩率: {total / len(test_texts):.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GWen tokenizer with the MiniMind recipe.")
    parser.add_argument("--input", nargs="+", default=["dataset/sft_t2t_mini.jsonl"])
    parser.add_argument("--output", default="model/tokenizer_mini8k")
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--special_tokens_num", type=int, default=SPECIAL_TOKENS_NUM)
    parser.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMPLES, help="-1 means use all rows.")
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument("--progress_interval", type=int, default=1000)
    parser.add_argument("--no_eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_tokenizer(args)
    if not args.no_eval:
        eval_tokenizer(args.output)


if __name__ == "__main__":
    main()
