"""Detección de dispositivo de cómputo (GPU NVIDIA/CUDA o CPU)."""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class DeviceInfo:
    device: str                 # "cuda" | "cpu"
    label: str                  # "GPU: NVIDIA ..." | "CPU MODE"
    gpu_name: str | None = None
    gpu_memory_gb: float | None = None
    cuda_version: str | None = None
    torch_version: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def detect_device(preferred: str = "auto") -> DeviceInfo:
    try:
        import torch
    except Exception:  # torch no instalado (p. ej. servicio API sin ML)
        return DeviceInfo("cpu", "CPU MODE", torch_version=None)
    cuda_ok = bool(torch.cuda.is_available())
    want = (preferred or "auto").lower()
    if want == "cpu" or not cuda_ok:
        return DeviceInfo("cpu", "CPU MODE", torch_version=torch.__version__)
    props = torch.cuda.get_device_properties(0)
    return DeviceInfo(
        "cuda", f"GPU: {props.name}", gpu_name=props.name,
        gpu_memory_gb=round(props.total_memory / 1024 ** 3, 2),
        cuda_version=torch.version.cuda, torch_version=torch.__version__)


def torch_device(preferred: str = "auto") -> str:
    return detect_device(preferred).device
