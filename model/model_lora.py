"""LoRA utilities for GWen."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Set

import torch
import torch.nn as nn


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "g_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
]


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: List[str] | None = None

    def __post_init__(self) -> None:
        if self.target_modules is None:
            self.target_modules = list(DEFAULT_TARGET_MODULES)
        if self.r <= 0:
            raise ValueError("LoRA rank r must be positive")

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "LoRAConfig":
        return cls(**data)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return self.base(x) + lora * self.scale

    def merged_linear(self) -> nn.Linear:
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        delta = (self.lora_B @ self.lora_A).to(self.base.weight.dtype) * self.scale
        merged.weight.data.copy_(self.base.weight.data + delta)
        if self.base.bias is not None:
            merged.bias.data.copy_(self.base.bias.data)
        return merged


def apply_lora_to_model(model: nn.Module, config: LoRAConfig) -> Set[str]:
    for param in model.parameters():
        param.requires_grad_(False)

    replaced: Set[str] = set()
    targets = set(config.target_modules or [])

    def replace(module: nn.Module, prefix: str = "") -> None:
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and name in targets:
                setattr(module, name, LoRALinear(child, config.r, config.alpha, config.dropout))
                replaced.add(full_name)
            else:
                replace(child, full_name)

    replace(model)
    return replaced


def iter_lora_modules(model: nn.Module) -> Iterable[tuple[nn.Module, str, LoRALinear]]:
    for module in model.modules():
        for name, child in module.named_children():
            if isinstance(child, LoRALinear):
                yield module, name, child


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


def save_lora_checkpoint(model: nn.Module, config: LoRAConfig, path: str) -> None:
    torch.save({"lora_config": config.to_dict(), "lora_state_dict": get_lora_state_dict(model)}, path)


def load_lora_state_dict(model: nn.Module, checkpoint: Dict[str, torch.Tensor] | Dict) -> int:
    state = checkpoint.get("lora_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model_state = model.state_dict()
    loaded = 0
    for key, value in state.items():
        if key in model_state:
            model_state[key].copy_(value.to(model_state[key].device, dtype=model_state[key].dtype))
            loaded += value.numel()
    return loaded


def merge_lora(model: nn.Module) -> int:
    merged = 0
    for parent, name, child in list(iter_lora_modules(model)):
        setattr(parent, name, child.merged_linear())
        merged += 1
    return merged


def unmerge_lora(model: nn.Module) -> None:
    raise RuntimeError("LoRA merge replaces adapter modules with nn.Linear and cannot be undone in-place")


def count_lora_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
