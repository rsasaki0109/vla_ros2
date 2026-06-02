from __future__ import annotations

import json
from pathlib import Path

from vla_zoo.compare.roofline import (
    HARDWARE_PROFILES,
    MODEL_PROFILES,
    ROOFLINE_SCHEMA_VERSION,
    HardwareProfile,
    ModelComputeProfile,
    build_roofline_report,
    estimate_roofline,
    format_roofline_markdown,
    realtime_band,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_estimate_roofline_is_memory_bound_at_batch_one() -> None:
    model = ModelComputeProfile("toy", params_billion=7.0, bytes_per_param=0.5)
    hw = HardwareProfile("test", peak_tflops_fp16=88.0, memory_bandwidth_gbps=672.0)
    est = estimate_roofline(model, hw)
    # 7e9 * 0.5 bytes / 672e9 B/s = ~5.21 ms; compute term is far smaller at batch 1
    assert abs(est.memory_bound_ms - 5.208) < 0.05
    assert est.bound_by == "memory"
    assert est.forward_floor_ms == est.memory_bound_ms


def test_weight_bytes_scales_with_params_and_dtype() -> None:
    bf16 = ModelComputeProfile("a", 1.0, 2.0)
    nf4 = ModelComputeProfile("b", 1.0, 0.5)
    assert bf16.weight_bytes == 2e9
    assert nf4.weight_bytes == 0.5e9


def test_realtime_band_thresholds() -> None:
    assert realtime_band(50.0) == "real-time (<=100 ms)"
    assert realtime_band(400.0) == "usable (sub-second)"
    assert realtime_band(2000.0) == "slow (>1 s)"
    assert realtime_band(None) == "unknown"


def test_build_report_joins_measured_and_computes_headroom() -> None:
    report = build_roofline_report(
        {"smolvla": 381.9, "openvla": 1996.8, "pi0": None},
        hardware=HARDWARE_PROFILES["rtx_4070_ti_super"],
    )
    by_model = {c.estimate.model: c for c in report.comparisons}
    assert set(by_model) == {"smolvla", "openvla", "pi0"}
    # headroom = measured / floor, large because recorded path is unoptimized
    assert by_model["smolvla"].headroom_x is not None
    assert by_model["smolvla"].headroom_x > 100
    assert by_model["openvla"].headroom_x > by_model["smolvla"].headroom_x
    # blocked model still shows its floor, with no measured latency and unknown band
    assert by_model["pi0"].measured_p50_ms is None
    assert by_model["pi0"].headroom_x is None
    assert by_model["pi0"].band == "unknown"


def test_report_to_dict_carries_schema_and_note() -> None:
    report = build_roofline_report(
        {"smolvla": 381.9}, hardware=HARDWARE_PROFILES["rtx_4070_ti_super"]
    )
    payload = report.to_dict()
    assert payload["schema_version"] == ROOFLINE_SCHEMA_VERSION
    assert "not an achievable latency" in payload["note"].lower()
    assert payload["hardware"]["name"] == "GPU"


def test_markdown_lists_floor_measured_and_headroom() -> None:
    report = build_roofline_report(
        {"smolvla": 381.9, "openvla": 1996.8},
        hardware=HARDWARE_PROFILES["rtx_4070_ti_super"],
    )
    markdown = format_roofline_markdown(report)
    assert "roofline floor vs recorded latency" in markdown.lower()
    assert "Headroom" in markdown
    assert "×" in markdown  # headroom multiplier rendered


def test_known_model_profiles_match_recorded_memory() -> None:
    # the declared dtype/params should be consistent with the recorded VRAM footprint
    assert MODEL_PROFILES["smolvla"].measured_memory_gb == 0.97
    assert MODEL_PROFILES["openvla"].bytes_per_param == 0.5  # nf4 4-bit


def test_recorded_roofline_artifact_is_valid() -> None:
    path = REPO_ROOT / "docs" / "assets" / "roofline" / "vla_roofline.json"
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == ROOFLINE_SCHEMA_VERSION
    models = {c["model"]: c for c in payload["comparisons"]}
    assert models["smolvla"]["realtime_band"] == "usable (sub-second)"
    assert models["openvla"]["realtime_band"] == "slow (>1 s)"
    assert models["smolvla"]["headroom_x"] > 100
