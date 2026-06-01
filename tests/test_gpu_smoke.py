from __future__ import annotations

import pytest

from vla_zoo.core.errors import ConfigurationError
from vla_zoo.runtime import gpu


class _FakeDevice:
    type = "cuda"


class _FakeTensor:
    def __matmul__(self, other: object) -> _FakeTensor:
        return self

    def __mul__(self, other: object) -> _FakeTensor:
        return self

    def __rmul__(self, other: object) -> _FakeTensor:
        return self

    def __getitem__(self, key: object) -> _FakeTensor:
        return self

    def detach(self) -> _FakeTensor:
        return self

    def float(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def item(self) -> float:
        return 1.25


class _FakeNoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def set_device(device: object) -> None:
        return None

    @staticmethod
    def get_device_name(device: object) -> str:
        return "Fake CUDA"

    @staticmethod
    def synchronize(device: object) -> None:
        return None

    @staticmethod
    def memory_allocated(device: object) -> int:
        return 1024 * 1024

    @staticmethod
    def memory_reserved(device: object) -> int:
        return 2 * 1024 * 1024


class _FakeTorch:
    __version__ = "2.fake"
    cuda = _FakeCuda()
    float16 = object()
    bfloat16 = object()
    float32 = object()

    @staticmethod
    def device(value: str) -> _FakeDevice:
        return _FakeDevice()

    @staticmethod
    def no_grad() -> _FakeNoGrad:
        return _FakeNoGrad()

    @staticmethod
    def randn(shape: tuple[int, int], *, device: object, dtype: object) -> _FakeTensor:
        return _FakeTensor()


def test_run_cuda_smoke_with_fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu, "_import_torch", lambda: _FakeTorch)

    result = gpu.run_cuda_smoke(device="cuda:0", dtype="float16", matrix_size=8, iterations=2)

    assert result.device == "cuda:0"
    assert result.device_name == "Fake CUDA"
    assert result.torch_version == "2.fake"
    assert result.memory_allocated_mib == 1.0
    assert result.memory_reserved_mib == 2.0
    assert result.sample == 1.25


def test_run_cuda_smoke_rejects_bad_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu, "_import_torch", lambda: _FakeTorch)

    with pytest.raises(ConfigurationError, match="Unsupported CUDA smoke dtype"):
        gpu.run_cuda_smoke(dtype="int8")
