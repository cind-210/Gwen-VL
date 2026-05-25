"""
GWen 加速库检测 — 自动检测并利用 GPU 加速库。

检测顺序:
    causal-conv1d     → GatedDeltaNet 因果卷积加速 (来自 Mamba 生态)
    flash-linear-attention → GatedDeltaNet Delta Rule 递推加速
    flash-attn        → FullAttention Flash Attention v2/v3
    triton            → 自定义 kernel 后端
    torch.compile     → PyTorch 2.0 图编译

用法:
    from utils.accelerate import detect_acceleration, print_acceleration_info
    accel = detect_acceleration()
    print_acceleration_info(accel)
"""

import importlib
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class AccelerationStatus:
    """加速库可用性状态."""
    causal_conv1d: bool = False          # GDN 因果卷积
    flash_linear_attn: bool = False      # GDN 线性注意力
    flash_attn: bool = False             # Full Attention
    triton: bool = False                 # 自定义 kernel
    torch_compile: bool = False          # torch.compile
    torch_version: str = ""
    cuda_available: bool = False
    cuda_version: str = ""
    gpu_name: str = ""

    @property
    def gdn_optimized(self) -> bool:
        """GatedDeltaNet 是否有加速实现."""
        return self.causal_conv1d or self.flash_linear_attn

    @property
    def full_attn_optimized(self) -> bool:
        """FullAttention 是否有加速实现."""
        return self.flash_attn  # F.scaled_dot_product_attention 已经用了 Flash

    def summary(self) -> str:
        parts = []
        parts.append(f"GDN causal-conv1d:   {'✓' if self.causal_conv1d else '✗ (fallback: PyTorch conv1d)'}")
        parts.append(f"GDN delta-rule:      {'✓' if self.flash_linear_attn else '✗ (fallback: Python loop)'}")
        parts.append(f"Full Attention:      {'✓' if self.flash_attn else '✗ (fallback: PyTorch SDPA)'}")
        parts.append(f"Triton:              {'✓' if self.triton else '✗'}")
        parts.append(f"torch.compile:       {'✓' if self.torch_compile else '✗'}")
        return "\n".join(parts)


# 单例缓存
_accel_cache: Optional[AccelerationStatus] = None


def detect_acceleration() -> AccelerationStatus:
    """检测所有可用的加速库 (结果缓存)."""
    global _accel_cache
    if _accel_cache is not None:
        return _accel_cache

    import torch

    status = AccelerationStatus(
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
    )

    if status.cuda_available:
        status.cuda_version = torch.version.cuda or "unknown"
        try:
            status.gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            status.gpu_name = "unknown"

    # 1. causal-conv1d (Mamba 因果卷积，加速 GDN 的 Conv1d)
    try:
        importlib.import_module("causal_conv1d")
        status.causal_conv1d = True
    except ImportError:
        pass

    # 2. flash-linear-attention (加速 GDN 的 Delta Rule 递推)
    try:
        importlib.import_module("fla")
        status.flash_linear_attn = True
    except ImportError:
        pass

    # 3. flash-attn (加速 Full Attention)
    try:
        importlib.import_module("flash_attn")
        status.flash_attn = True
    except ImportError:
        pass

    # 4. triton
    try:
        importlib.import_module("triton")
        status.triton = True
    except ImportError:
        pass

    # 5. torch.compile (PyTorch >= 2.0)
    status.torch_compile = hasattr(torch, "compile")

    _accel_cache = status
    return status


def print_acceleration_info(status: AccelerationStatus = None):
    """打印加速库检测结果."""
    if status is None:
        status = detect_acceleration()

    lines = [
        "=" * 55,
        f"  PyTorch {status.torch_version}  |  CUDA {status.cuda_version}",
        f"  GPU: {status.gpu_name}",
        f"  {'─' * 51}",
        f"  {status.summary().replace(chr(10), chr(10) + '  ')}",
        f"  {'─' * 51}",
    ]

    install_hints = []
    if not status.causal_conv1d:
        install_hints.append("  pip install causal-conv1d          # +30% GDN speed")
    if not status.flash_linear_attn:
        install_hints.append("  pip install flash-linear-attention # +50% GDN speed")
    if not status.flash_attn:
        install_hints.append("  pip install flash-attn --no-build-isolation  # +20% FA speed")
    if not status.triton:
        install_hints.append("  pip install triton                 # custom kernel support")

    if install_hints:
        lines.append("  Recommended (install for faster training):")
        lines.extend(install_hints)
    else:
        lines.append("  All acceleration libraries available!")

    lines.append("=" * 55)
    text = "\n".join(lines)
    print(text)
    return text


# 便捷访问
ACCEL_AVAILABLE = detect_acceleration()
