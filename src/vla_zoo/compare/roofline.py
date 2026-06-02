"""First-order roofline latency floors for VLA inference, joined with recorded probes.

Inspired by VLA-Perf (NVIDIA, arXiv:2602.18397), which predicts VLA inference latency
analytically with the roofline model ``latency = max(FLOPs / peak_compute, Bytes /
memory_bandwidth)``. At batch size one, VLA inference is dominated by the memory-bound
term: the model weights must be streamed from HBM at least once per forward pass, so the
hardware lower bound is ``weight_bytes / memory_bandwidth``.

This module computes that **single-forward roofline floor** from a model's declared
parameter count and dtype, on a chosen hardware profile, and contrasts it with vla_zoo's
**recorded** runtime probes. The floor is a theoretical hardware lower bound, not an
achievable target: the recorded latency additionally pays for multi-token / multi-step
decoding and Python / framework overhead, so the gap is best read as optimization
headroom -- exactly the regime VLA-Perf quantifies. Real-time bands follow VLA-Perf's
10-100 ms visual-ingestion target.

Everything here is a first-order analytical estimate, not a measurement, and makes no
policy-quality or task-success claim. Hardware peak figures are nominal vendor specs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Schema identifier for the JSON artifact this module emits.
ROOFLINE_SCHEMA_VERSION = "vla-zoo-roofline/v1"

ROOFLINE_NOTE = (
    "First-order roofline floor (VLA-Perf style): the single-forward memory-bound hardware "
    "lower bound weight_bytes/bandwidth at batch 1. NOT an achievable latency -- recorded "
    "p50 also pays for multi-step decode + framework overhead, so measured/floor is "
    "optimization headroom. Hardware peaks are nominal vendor specs. No policy-quality claim."
)


@dataclass(frozen=True)
class HardwareProfile:
    """A GPU's nominal peak compute and memory bandwidth (vendor spec sheet)."""

    name: str
    peak_tflops_fp16: float
    memory_bandwidth_gbps: float
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "peak_tflops_fp16": self.peak_tflops_fp16,
            "memory_bandwidth_gbps": self.memory_bandwidth_gbps,
            "note": self.note,
        }


@dataclass(frozen=True)
class ModelComputeProfile:
    """A model's roofline-relevant size: parameter count and bytes per stored parameter."""

    name: str
    params_billion: float
    bytes_per_param: float
    measured_memory_gb: float | None = None
    note: str = ""

    @property
    def weight_bytes(self) -> float:
        return self.params_billion * 1e9 * self.bytes_per_param

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "params_billion": self.params_billion,
            "bytes_per_param": self.bytes_per_param,
            "weight_gb": self.weight_bytes / 1e9,
            "measured_memory_gb": self.measured_memory_gb,
            "note": self.note,
        }


#: Nominal vendor specs (fp16 tensor TFLOPS dense; HBM/GDDR bandwidth GB/s).
HARDWARE_PROFILES: dict[str, HardwareProfile] = {
    "rtx_4070_ti_super": HardwareProfile(
        "GPU", 88.0, 672.0, "Local card the recorded probes ran on (16 GB)."
    ),
    "rtx_4090": HardwareProfile("GPU", 165.0, 1008.0, "Consumer flagship (24 GB)."),
    "jetson_agx_orin": HardwareProfile(
        "Jetson AGX Orin", 42.0, 204.8, "Edge module (fp16 dense; int8 sparse is higher)."
    ),
    "a100_80gb": HardwareProfile("GPU 80GB", 312.0, 2039.0, "Datacenter (HBM2e)."),
}

#: Declared sizes for the adapters vla_zoo has measured. params/dtype are published facts;
#: measured_memory_gb comes from the recorded deployment table and is context, not the floor.
MODEL_PROFILES: dict[str, ModelComputeProfile] = {
    "smolvla": ModelComputeProfile(
        "smolvla", 0.45, 2.0, 0.97, "SmolVLA-450M, bf16 (recorded ~0.97 GB)."
    ),
    "openvla": ModelComputeProfile(
        "openvla", 7.0, 0.5, 4.6, "OpenVLA-7b, bitsandbytes nf4 4-bit (recorded ~4.6 GB)."
    ),
    "pi0": ModelComputeProfile(
        "pi0", 3.3, 2.0, 8.9, "pi0_base (PaliGemma-3b + action expert), bf16; latency blocked."
    ),
}


@dataclass(frozen=True)
class RooflineEstimate:
    """The roofline floor for one model on one hardware profile."""

    model: str
    hardware: str
    weight_gb: float
    memory_bound_ms: float
    compute_bound_ms: float
    forward_floor_ms: float
    bound_by: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "hardware": self.hardware,
            "weight_gb": self.weight_gb,
            "memory_bound_ms": self.memory_bound_ms,
            "compute_bound_ms": self.compute_bound_ms,
            "forward_floor_ms": self.forward_floor_ms,
            "bound_by": self.bound_by,
        }


def estimate_roofline(model: ModelComputeProfile, hardware: HardwareProfile) -> RooflineEstimate:
    """Compute the single-forward, batch-1 roofline floor for a model on hardware."""

    memory_ms = model.weight_bytes / (hardware.memory_bandwidth_gbps * 1e9) * 1000.0
    # Compute term: one matmul pass over the weights at batch 1 is ~2 FLOPs per parameter.
    flops = 2.0 * model.params_billion * 1e9
    compute_ms = flops / (hardware.peak_tflops_fp16 * 1e12) * 1000.0
    floor = max(memory_ms, compute_ms)
    bound_by = "memory" if memory_ms >= compute_ms else "compute"
    return RooflineEstimate(
        model=model.name,
        hardware=hardware.name,
        weight_gb=model.weight_bytes / 1e9,
        memory_bound_ms=memory_ms,
        compute_bound_ms=compute_ms,
        forward_floor_ms=floor,
        bound_by=bound_by,
    )


def realtime_band(measured_p50_ms: float | None) -> str:
    """Classify a measured p50 latency against the VLA-Perf 10-100 ms visual target."""

    if measured_p50_ms is None:
        return "unknown"
    if measured_p50_ms <= 100.0:
        return "real-time (<=100 ms)"
    if measured_p50_ms <= 1000.0:
        return "usable (sub-second)"
    return "slow (>1 s)"


@dataclass(frozen=True)
class RooflineComparison:
    """A model's roofline floor next to its recorded p50 and the resulting headroom."""

    estimate: RooflineEstimate
    measured_p50_ms: float | None
    headroom_x: float | None
    band: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.estimate.to_dict(),
            "measured_p50_ms": self.measured_p50_ms,
            "headroom_x": self.headroom_x,
            "realtime_band": self.band,
        }


@dataclass(frozen=True)
class RooflineReport:
    """Roofline floors vs recorded probes across models on one hardware profile."""

    hardware: HardwareProfile
    comparisons: tuple[RooflineComparison, ...]
    note: str = ROOFLINE_NOTE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ROOFLINE_SCHEMA_VERSION,
            "hardware": self.hardware.to_dict(),
            "comparisons": [c.to_dict() for c in self.comparisons],
            "note": self.note,
        }


def build_roofline_report(
    measured_p50_by_model: dict[str, float | None],
    *,
    hardware: HardwareProfile,
    profiles: dict[str, ModelComputeProfile] | None = None,
) -> RooflineReport:
    """Join measured p50 latencies with roofline floors for every known model profile.

    ``measured_p50_by_model`` maps a model name to its recorded p50 (or ``None`` for a
    blocked/unmeasured model). Models present in ``profiles`` but absent from the mapping
    are still shown with their floor and an unknown band; models without a profile are
    skipped (their roofline-relevant size is not declared here).
    """

    table = profiles if profiles is not None else MODEL_PROFILES
    comparisons: list[RooflineComparison] = []
    names = list(table.keys())
    for extra in measured_p50_by_model:
        if extra not in table and extra in MODEL_PROFILES:
            names.append(extra)
    for name in names:
        profile = table.get(name) or MODEL_PROFILES.get(name)
        if profile is None:
            continue
        estimate = estimate_roofline(profile, hardware)
        measured = measured_p50_by_model.get(name)
        headroom = (measured / estimate.forward_floor_ms) if measured else None
        comparisons.append(
            RooflineComparison(
                estimate=estimate,
                measured_p50_ms=measured,
                headroom_x=headroom,
                band=realtime_band(measured),
            )
        )
    return RooflineReport(hardware=hardware, comparisons=tuple(comparisons))


def _fmt(value: float | None, suffix: str = "") -> str:
    return f"{value:.1f}{suffix}" if value is not None else "—"


def format_roofline_markdown(report: RooflineReport) -> str:
    """Render the roofline-vs-measured comparison as a runtime-centric Markdown report."""

    hw = report.hardware
    lines = [
        "# VLA roofline floor vs recorded latency",
        "",
        f"- Hardware: **{hw.name}** "
        f"({hw.memory_bandwidth_gbps:g} GB/s, {hw.peak_tflops_fp16:g} TFLOPS fp16 nominal)",
        "- Floor = single-forward, batch-1 memory-bound lower bound `weight_bytes / bandwidth`.",
        "",
        "| Model | Weights | Floor (ms) | Bound by | Measured p50 (ms) | Headroom "
        "| Real-time band |",
        "|---|---:|---:|:--:|---:|---:|:--|",
    ]
    for comp in report.comparisons:
        est = comp.estimate
        headroom = f"{comp.headroom_x:.0f}×" if comp.headroom_x is not None else "—"
        lines.append(
            f"| {est.model} | {est.weight_gb:.2f} GB | {est.forward_floor_ms:.2f} "
            f"| {est.bound_by} | {_fmt(comp.measured_p50_ms)} | {headroom} | {comp.band} |"
        )
    lines += ["", f"> {report.note}", ""]
    return "\n".join(lines)
