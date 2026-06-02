from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

from vla_zoo.core.errors import ConfigurationError, MissingDependencyError


@dataclass(frozen=True)
class CUDASmokeResult:
    device: str
    device_name: str
    torch_version: str
    dtype: str
    matrix_size: int
    iterations: int
    elapsed_ms: float
    memory_allocated_mib: float
    memory_reserved_mib: float
    sample: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        msg = (
            'CUDA smoke requires torch. Install GPU dependencies with: '
            'pip install "vla_zoo[openvla]"'
        )
        raise MissingDependencyError(msg) from exc
    return torch


def _torch_dtype(torch: Any, dtype: str) -> Any:
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return dtype_map[dtype.lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(dtype_map))
        raise ConfigurationError(
            f"Unsupported CUDA smoke dtype {dtype!r}; use one of {supported}"
        ) from exc


def run_cuda_smoke(
    *,
    device: str = "cuda:0",
    dtype: str = "float16",
    matrix_size: int = 512,
    iterations: int = 8,
) -> CUDASmokeResult:
    """Run a small torch CUDA matmul to verify that the GPU path actually executes."""

    if matrix_size <= 0:
        raise ConfigurationError("matrix_size must be positive")
    if iterations <= 0:
        raise ConfigurationError("iterations must be positive")

    torch = _import_torch()
    if not bool(torch.cuda.is_available()):
        raise ConfigurationError("torch is installed but torch.cuda.is_available() is false")

    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ConfigurationError(f"CUDA smoke requires a CUDA device, got {device!r}")

    tensor_dtype = _torch_dtype(torch, dtype)
    torch.cuda.set_device(torch_device)
    device_name = str(torch.cuda.get_device_name(torch_device))

    with torch.no_grad():
        a = torch.randn((matrix_size, matrix_size), device=torch_device, dtype=tensor_dtype) * 0.01
        b = torch.randn((matrix_size, matrix_size), device=torch_device, dtype=tensor_dtype) * 0.01
        _ = a @ b
        torch.cuda.synchronize(torch_device)

        start = perf_counter()
        output = a
        for _ in range(iterations):
            output = a @ b
        torch.cuda.synchronize(torch_device)
        elapsed_ms = (perf_counter() - start) * 1000.0
        sample = float(output[0, 0].detach().float().cpu().item())

    memory_allocated_mib = float(torch.cuda.memory_allocated(torch_device) / (1024**2))
    memory_reserved_mib = float(torch.cuda.memory_reserved(torch_device) / (1024**2))

    return CUDASmokeResult(
        device=device,
        device_name=device_name,
        torch_version=str(getattr(torch, "__version__", "unknown")),
        dtype=dtype,
        matrix_size=matrix_size,
        iterations=iterations,
        elapsed_ms=elapsed_ms,
        memory_allocated_mib=memory_allocated_mib,
        memory_reserved_mib=memory_reserved_mib,
        sample=sample,
    )
