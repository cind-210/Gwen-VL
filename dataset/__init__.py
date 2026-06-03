"""Datasets for GWen pretrain, SFT, DPO, and experimental RL."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


CHATML_START = "<|im_start|>"
CHATML_END = "<|im_end|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
DEFAULT_SYSTEM_PROMPTS = [
    "You are a helpful assistant.",
    "你是一个乐于助人的AI助手，你叫作GWen，是由ChengJun和Cind开发的。",
]


def read_jsonl(path: str) -> Iterable[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            with open(path, "r", encoding=enc) as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
            return
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"Cannot decode {path}")


def build_chatml(conversations: Sequence[Dict[str, str]], add_generation_prompt: bool = False) -> str:
    parts: List[str] = []
    for message in conversations:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"{CHATML_START}{role}\n{content}{CHATML_END}\n")
    if add_generation_prompt:
        parts.append(f"{CHATML_START}assistant\n")
    return "".join(parts)


def _encode(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False, verbose=False)


def normalize_conversations(item: Dict) -> List[Dict[str, str]]:
    """Normalize common SFT schemas to GWen role/content turns."""

    raw_messages = item.get("conversations") or item.get("messages")
    if not isinstance(raw_messages, list):
        return []

    role_map = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "model": "assistant",
        "system": "system",
        "tool": "tool",
    }
    conversations: List[Dict[str, str]] = []
    for message in raw_messages:
        if not isinstance(message, dict):
            continue
        raw_role = message.get("role", message.get("from", "user"))
        role = role_map.get(str(raw_role).lower())
        if role is None:
            continue
        content = message.get("content", message.get("value", message.get("text", "")))
        if content is None:
            continue
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        content = content.strip()
        if not content:
            continue
        conversations.append({"role": role, "content": content})
    return conversations


def has_assistant_turn(conversations: Sequence[Dict[str, str]]) -> bool:
    return any(message.get("role") == "assistant" for message in conversations)


def build_chatml_features(
    conversations: Sequence[Dict[str, str]],
    tokenizer,
    max_length: int,
    assistant_only_loss: bool = True,
) -> Dict[str, torch.Tensor]:
    input_ids: List[int] = []
    labels: List[int] = []
    for message in conversations:
        role = message.get("role", "user")
        content = message.get("content", "")
        prefix = _encode(tokenizer, f"{CHATML_START}{role}\n")
        content_ids = _encode(tokenizer, content)
        suffix = _encode(tokenizer, f"{CHATML_END}\n")
        ids = prefix + content_ids + suffix
        if assistant_only_loss and role == "assistant":
            label = [-100] * len(prefix) + content_ids + suffix
        elif assistant_only_loss:
            label = [-100] * len(ids)
        else:
            label = ids.copy()
        input_ids.extend(ids)
        labels.extend(label)

    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    pad_len = max_length - len(input_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids += [pad_id] * pad_len
    labels += [-100] * pad_len
    attention_mask = [1] * (max_length - pad_len) + [0] * pad_len
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def build_vlm_chatml_features(
    conversations: Sequence[Dict[str, str]],
    tokenizer,
    max_length: int,
    assistant_only_loss: bool = True,
) -> Dict[str, torch.Tensor]:
    return build_chatml_features(conversations, tokenizer, max_length, assistant_only_loss=assistant_only_loss)


def normalize_vlm_conversations(item: Dict, image_token_count: int) -> Tuple[List[Dict[str, str]], str]:
    raw_messages = item.get("conversations") or item.get("messages")
    if not isinstance(raw_messages, list):
        return [], ""

    image_path = str(item.get("image", item.get("image_path", "")) or "")
    image_placeholder = VISION_START + (IMAGE_PAD * image_token_count) + VISION_END
    role_map = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "model": "assistant",
        "system": "system",
        "tool": "tool",
    }
    conversations: List[Dict[str, str]] = []
    inserted_top_level_image = False
    for message in raw_messages:
        if not isinstance(message, dict):
            continue
        raw_role = message.get("role", message.get("from", "user"))
        role = role_map.get(str(raw_role).lower())
        if role is None:
            continue
        content = message.get("content", message.get("value", message.get("text", "")))
        parts: List[str] = []
        if isinstance(content, list):
            for piece in content:
                if not isinstance(piece, dict):
                    continue
                piece_type = str(piece.get("type", "")).lower()
                if piece_type == "image":
                    candidate = piece.get("image", piece.get("path", ""))
                    if candidate:
                        image_path = str(candidate)
                    parts.append(image_placeholder)
                elif piece_type == "text":
                    text = piece.get("text", "")
                    if text:
                        parts.append(str(text))
        else:
            text = "" if content is None else str(content)
            if image_path and role == "user" and not inserted_top_level_image:
                parts.append(image_placeholder)
                inserted_top_level_image = True
            parts.append(text)
        merged = "".join(parts).strip()
        if merged:
            conversations.append({"role": role, "content": merged})
    return conversations, image_path


class LazyPretrainDataset(Dataset):
    """MiniMind-style pretrain dataset: tokenize each JSONL text sample on demand."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 1024, add_eos: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_eos = add_eos
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.eos_id = tokenizer.eos_token_id
        self.data: List[str] = []
        for item in read_jsonl(data_path):
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                self.data.append(text)
        if not self.data:
            raise ValueError(f"No valid pretrain text found in {data_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = _encode(self.tokenizer, self.data[idx])
        if self.add_eos and self.eos_id is not None:
            ids.append(self.eos_id)
        ids = ids[: self.max_length]
        labels = ids.copy()
        pad_len = self.max_length - len(ids)
        ids = ids + [self.pad_id] * pad_len
        labels = labels + [-100] * pad_len
        attention_mask = [1] * (self.max_length - pad_len) + [0] * pad_len
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class PackedPretrainDataset(Dataset):
    """Pack JSONL text records into fixed-length next-token blocks."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 1024,
        add_eos: bool = True,
        cache_dir: str | None = None,
        use_cache: bool = True,
        build_cache: bool = True,
        logger=None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        cache_path = self._cache_path(data_path, tokenizer, max_length, add_eos, cache_dir)

        if use_cache and os.path.exists(cache_path):
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            self.blocks = payload["blocks"]
            if logger:
                logger.info(f"Loaded pretrain token cache: {cache_path} ({len(self.blocks)} blocks)")
            return

        if use_cache and not build_cache:
            self._wait_for_cache(cache_path, logger)
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            self.blocks = payload["blocks"]
            if logger:
                logger.info(f"Loaded pretrain token cache: {cache_path} ({len(self.blocks)} blocks)")
            return

        self.blocks = self._build_blocks(data_path, tokenizer, max_length, add_eos, logger)
        if use_cache:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            tmp_path = cache_path + ".tmp"
            torch.save({"blocks": self.blocks, "max_length": max_length, "data_path": data_path}, tmp_path)
            os.replace(tmp_path, cache_path)
            if logger:
                logger.info(f"Saved pretrain token cache: {cache_path}")

    @staticmethod
    def _cache_path(data_path: str, tokenizer, max_length: int, add_eos: bool, cache_dir: str | None) -> str:
        stat = os.stat(data_path)
        fingerprint_text = "GWen tokenizer fingerprint 你好<|im_start|><|im_end|>"
        tokenizer_fingerprint = ",".join(map(str, _encode(tokenizer, fingerprint_text)))
        key = "|".join(
            [
                os.path.abspath(data_path),
                str(stat.st_size),
                str(int(stat.st_mtime)),
                str(max_length),
                str(add_eos),
                str(len(tokenizer)),
                str(getattr(tokenizer, "vocab_size", "")),
                str(tokenizer.eos_token_id),
                str(tokenizer.pad_token_id),
                str(tokenizer.convert_tokens_to_ids(CHATML_START)),
                str(tokenizer.convert_tokens_to_ids(CHATML_END)),
                tokenizer_fingerprint,
            ]
        )
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
        root = cache_dir or os.path.join(os.path.dirname(data_path), ".cache")
        return os.path.join(root, f"pretrain_{digest}_seq{max_length}.pt")

    @staticmethod
    def _wait_for_cache(cache_path: str, logger=None, timeout: int = 86400) -> None:
        start = time.time()
        last_log = 0.0
        while not os.path.exists(cache_path):
            if time.time() - start > timeout:
                raise TimeoutError(f"Timed out waiting for pretrain cache: {cache_path}")
            if logger and time.time() - last_log > 30:
                logger.info(f"Waiting for rank0 to build pretrain cache: {cache_path}")
                last_log = time.time()
            time.sleep(2)

    @staticmethod
    def _build_blocks(data_path: str, tokenizer, max_length: int, add_eos: bool, logger=None) -> List[List[int]]:
        eos_id = tokenizer.eos_token_id
        stream: List[int] = []
        valid_lines = 0
        total_lines = 0
        start = time.time()
        last_log = start
        for item in read_jsonl(data_path):
            total_lines += 1
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                stream.extend(_encode(tokenizer, text))
                if add_eos and eos_id is not None:
                    stream.append(eos_id)
                valid_lines += 1
            if logger and time.time() - last_log > 10:
                logger.info(
                    f"Tokenizing pretrain data... lines={total_lines} valid={valid_lines} "
                    f"tokens={len(stream)/1e6:.1f}M elapsed={time.time()-start:.0f}s"
                )
                last_log = time.time()
        if len(stream) < 2:
            raise ValueError(f"No valid pretrain text found in {data_path}")
        block_size = max_length
        blocks = [stream[i : i + block_size] for i in range(0, len(stream) - 1, block_size)]
        if len(blocks[-1]) < 2:
            blocks.pop()
        if logger:
            logger.info(
                f"Built pretrain blocks: lines={total_lines} valid={valid_lines} "
                f"tokens={len(stream)/1e6:.1f}M blocks={len(blocks)} elapsed={time.time()-start:.0f}s"
            )
        return blocks

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        block = self.blocks[idx]
        ids = block[: self.max_length]
        labels = block[: self.max_length]
        pad_len = self.max_length - len(ids)
        ids = ids + [self.pad_id] * pad_len
        labels = labels + [-100] * pad_len
        attention_mask = [1] * (self.max_length - pad_len) + [0] * pad_len
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


PretrainDataset = LazyPretrainDataset


class SFTDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 1024,
        add_system_prob: float = 0.0,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_system_prob = add_system_prob
        self.data = []
        for item in read_jsonl(data_path):
            conversations = normalize_conversations(item)
            if conversations and has_assistant_turn(conversations):
                self.data.append(conversations)
        if not self.data:
            raise ValueError(f"No SFT conversations found in {data_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        conversations = list(self.data[idx])
        if self.add_system_prob > 0 and conversations and conversations[0].get("role") != "system":
            if random.random() < self.add_system_prob:
                conversations = [{"role": "system", "content": random.choice(DEFAULT_SYSTEM_PROMPTS)}] + conversations
        return build_chatml_features(conversations, self.tokenizer, self.max_length, assistant_only_loss=True)


class VLMSFTDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        image_processor,
        max_length: int = 1024,
        image_root: str = "",
        image_token_count: int = 64,
    ):
        from PIL import Image

        self.Image = Image
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.image_root = image_root
        self.image_token_count = image_token_count
        self.data = []
        for item in read_jsonl(data_path):
            conversations, image_path = normalize_vlm_conversations(item, image_token_count)
            if conversations and image_path and has_assistant_turn(conversations):
                self.data.append({"conversations": conversations, "image": image_path})
        if not self.data:
            raise ValueError(f"No single-image VLM conversations found in {data_path}")

    def __len__(self) -> int:
        return len(self.data)

    def _resolve_image_path(self, image_path: str) -> str:
        if os.path.isabs(image_path) or not self.image_root:
            return image_path
        return os.path.join(self.image_root, image_path)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        features = build_vlm_chatml_features(item["conversations"], self.tokenizer, self.max_length, assistant_only_loss=True)
        image_path = self._resolve_image_path(item["image"])
        image = self.Image.open(image_path).convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
        features["pixel_values"] = pixel_values
        return features


class DPODataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = [item for item in read_jsonl(data_path) if "chosen" in item and "rejected" in item]
        if not self.data:
            raise ValueError(f"No DPO pairs found in {data_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        chosen_messages = normalize_conversations({"conversations": item["chosen"]})
        rejected_messages = normalize_conversations({"conversations": item["rejected"]})
        chosen = build_chatml_features(chosen_messages, self.tokenizer, self.max_length, assistant_only_loss=True)
        rejected = build_chatml_features(rejected_messages, self.tokenizer, self.max_length, assistant_only_loss=True)
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_labels": chosen["labels"],
            "chosen_attention_mask": chosen["attention_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_labels": rejected["labels"],
            "rejected_attention_mask": rejected["attention_mask"],
        }


class RLAIFDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        for item in read_jsonl(data_path):
            conversations = normalize_conversations(item)
            if conversations:
                self.data.append(conversations)
        if not self.data:
            raise ValueError(f"No RLAIF conversations found in {data_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        conversations = self.data[idx]
        prompt = []
        for turn in conversations:
            if turn.get("role") == "assistant":
                break
            prompt.append(turn)
        text = build_chatml(prompt, add_generation_prompt=True)
        ids = _encode(self.tokenizer, text)[: self.max_length]
        return {"prompt": text, "prompt_ids": torch.tensor(ids, dtype=torch.long)}


class AgentRLDataset(Dataset):
    def __init__(self, data_path: str, tokenizer=None, max_length: int = 1024):
        self.data = [item for item in read_jsonl(data_path) if "prompt" in item and "gt" in item]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]
