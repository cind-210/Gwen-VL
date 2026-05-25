"""Shared training helpers for GWen."""

from __future__ import annotations

import math
import os
import random
import time
from contextlib import contextmanager, nullcontext
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from model.model_gwen import CONFIG_PRESETS, GWenConfig


_LAST_TOKENIZER_VOCAB_SIZE: Optional[int] = None
DEFAULT_TOKENIZER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "model", "tokenizer_mini8k")
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_config(name: str, max_seq_len: Optional[int] = None, vocab_size: Optional[int] = None) -> GWenConfig:
    if name not in CONFIG_PRESETS:
        raise KeyError(f"Unknown config {name}. Choose from {sorted(CONFIG_PRESETS)}")
    config = CONFIG_PRESETS[name]()
    if max_seq_len is not None:
        config.max_position_embeddings = max_seq_len
    effective_vocab_size = vocab_size if vocab_size is not None else _LAST_TOKENIZER_VOCAB_SIZE
    if effective_vocab_size is not None:
        config.vocab_size = int(effective_vocab_size)
    return config


def load_checkpoint(path: str, map_location: Union[str, torch.device] = "cpu") -> Dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    return ckpt if isinstance(ckpt, dict) else {"model_state_dict": ckpt}


def config_from_checkpoint(
    ckpt: Dict,
    fallback_name: str,
    max_seq_len: Optional[int] = None,
    tokenizer_vocab_size: Optional[int] = None,
) -> GWenConfig:
    if isinstance(ckpt.get("config"), dict):
        config = GWenConfig.from_dict(ckpt["config"])
        if max_seq_len is not None:
            config.max_position_embeddings = max_seq_len
    else:
        config = get_config(fallback_name, max_seq_len=max_seq_len, vocab_size=tokenizer_vocab_size)

    if tokenizer_vocab_size is not None and int(config.vocab_size) != int(tokenizer_vocab_size):
        raise ValueError(
            f"Tokenizer vocab size ({tokenizer_vocab_size}) does not match checkpoint/model config "
            f"vocab_size ({config.vocab_size}). Use the same tokenizer that was used for training."
        )
    return config


def configure_torch_speed() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def setup_distributed(device_arg: str = "") -> Dict:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device(device_arg or "cuda")
    else:
        device = torch.device("cpu")
    return {
        "distributed": distributed,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": device,
        "is_main": rank == 0,
    }


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


class Logger:
    def __init__(self, log_path: Optional[str] = None, enabled: bool = True):
        self.log_path = log_path
        self.enabled = enabled

    def info(self, message: str) -> None:
        if not self.enabled:
            return
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def banner(self, message: str) -> None:
        self.info("=" * 80)
        self.info(message)
        self.info("=" * 80)


def load_tokenizer(tokenizer_path: str = DEFAULT_TOKENIZER_PATH):
    from transformers import AutoTokenizer

    tokenizer_path = tokenizer_path or DEFAULT_TOKENIZER_PATH
    if tokenizer_path:
        normalized_path = tokenizer_path.replace("\\", "/")
        is_local_path = (
            os.path.isabs(tokenizer_path)
            or normalized_path.startswith(("./", "../", "/", "model/", "dataset/", "out/"))
            or os.path.exists(tokenizer_path)
        )
        if is_local_path and not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer path does not exist: {tokenizer_path}. "
                "Run scripts/train_tokenizer.py to build model/tokenizer_mini8k first."
            )
        if os.path.isdir(tokenizer_path) and not os.path.exists(os.path.join(tokenizer_path, "tokenizer.json")):
            raise FileNotFoundError(
                f"Tokenizer files were not found in {tokenizer_path}. "
                "Run scripts/train_tokenizer.py first, or pass model/tokenizer explicitly for the Qwen tokenizer."
            )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    required_tokens = ["<|im_start|>", "<|im_end|>"]
    vocab = tokenizer.get_vocab()
    missing_tokens = [token for token in required_tokens if token not in vocab]
    if missing_tokens:
        raise ValueError(
            f"Tokenizer at {tokenizer_path} is missing required ChatML tokens: {missing_tokens}. "
            "Retrain it with scripts/train_tokenizer.py or pass the matching tokenizer_path."
        )
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = "<|im_end|>"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    global _LAST_TOKENIZER_VOCAB_SIZE
    _LAST_TOKENIZER_VOCAB_SIZE = int(len(tokenizer))
    return tokenizer


def resolve_dtype(dtype: str, device: torch.device) -> Tuple[torch.dtype, str]:
    if dtype == "bf16":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bf16"
        if device.type == "cpu":
            return torch.bfloat16, "bf16"
        return torch.float16, "fp16"
    if dtype == "fp16":
        return torch.float16, "fp16"
    return torch.float32, "fp32"


def amp_context(dtype_name: str, dtype: torch.dtype, device: torch.device):
    if dtype_name == "fp32" or device.type == "cpu":
        return nullcontext()
    return autocast(device_type=device.type, dtype=dtype)


def make_scaler(dtype_name: str, device: torch.device) -> GradScaler:
    return GradScaler(device.type, enabled=(device.type == "cuda" and dtype_name == "fp16"))


def cosine_lr_lambda(step: int, total_steps: int, warmup_steps: int, min_ratio: float = 0.1) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))


def sync_mean(value: torch.Tensor, env: Dict) -> torch.Tensor:
    if env.get("distributed", False):
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value = value / env["world_size"]
    return value


@torch.no_grad()
def evaluate_loss(model, loader, env: Dict, dtype_name: str, dtype: torch.dtype, max_steps: int = 50) -> float:
    model_was_training = model.training
    model.eval()
    loss_sum = torch.zeros((), device=env["device"], dtype=torch.float32)
    loss_count = torch.zeros((), device=env["device"], dtype=torch.float32)
    for idx, batch in enumerate(loader):
        if max_steps > 0 and idx >= max_steps:
            break
        input_ids = batch["input_ids"].to(env["device"], non_blocking=True)
        labels = batch["labels"].to(env["device"], non_blocking=True)
        attention_mask = batch.get("attention_mask")
        attention_mask = attention_mask.to(env["device"], non_blocking=True) if attention_mask is not None else None
        with amp_context(dtype_name, dtype, env["device"]):
            loss = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)["loss"]
        loss_sum += loss.detach().float()
        loss_count += 1
    if env.get("distributed", False):
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
    if model_was_training:
        model.train()
    return (loss_sum / loss_count.clamp_min(1)).item()


def create_dataloader(dataset, batch_size: int, distributed: bool, shuffle: bool = True, num_workers: int = 0):
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": 4,
            }
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        **loader_kwargs,
    )
    return loader, sampler


def unwrap_model(model):
    model = model.module if isinstance(model, DDP) else model
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def wrap_ddp(model: torch.nn.Module, env: Dict) -> torch.nn.Module:
    if not env["distributed"]:
        return model
    kwargs = {}
    if env["device"].type == "cuda":
        kwargs = {"device_ids": [env["local_rank"]], "output_device": env["local_rank"]}
    return DDP(model, **kwargs)


@contextmanager
def maybe_no_sync(model: torch.nn.Module, enabled: bool):
    if enabled and hasattr(model, "no_sync"):
        with model.no_sync():
            yield
    else:
        yield


def save_checkpoint(
    path: str,
    model,
    config: GWenConfig,
    optimizer=None,
    scheduler=None,
    scaler=None,
    step: int = 0,
    epoch: int = 0,
    train_args: Optional[Dict] = None,
    extra: Optional[Dict] = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model_state = unwrap_model(model).state_dict()
    payload = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model_state.items()},
        "config": config.to_dict(),
        "step": step,
        "epoch": epoch,
        "train_args": train_args or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None and scaler.is_enabled():
        payload["scaler_state_dict"] = scaler.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_model_weights(model, checkpoint, device: torch.device, strict: bool = False) -> Dict:
    ckpt = load_checkpoint(checkpoint, map_location=device) if isinstance(checkpoint, (str, os.PathLike)) else checkpoint
    state = ckpt.get("model_state_dict", ckpt)
    target_model = unwrap_model(model)
    if not strict:
        target_state = target_model.state_dict()
        skipped = {}
        filtered = {}
        for key, value in state.items():
            if key in target_state and tuple(value.shape) != tuple(target_state[key].shape):
                skipped[key] = {"checkpoint": tuple(value.shape), "model": tuple(target_state[key].shape)}
                continue
            filtered[key] = value
        missing, unexpected = target_model.load_state_dict(filtered, strict=False)
        ckpt["missing_keys"] = list(missing)
        ckpt["unexpected_keys"] = list(unexpected)
        ckpt["skipped_shape_mismatch"] = skipped
    else:
        target_model.load_state_dict(state, strict=True)
    return ckpt


def load_resume(path: str, model, optimizer, scheduler, scaler, device: torch.device) -> Tuple[int, int]:
    ckpt = load_model_weights(model, path, device, strict=False)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and scaler.is_enabled() and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return int(ckpt.get("step", 0)), int(ckpt.get("epoch", 0))


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.1f}h"
