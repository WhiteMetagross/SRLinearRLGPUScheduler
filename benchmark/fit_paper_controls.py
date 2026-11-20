"""Fit every deterministic quota artifact reported in the paper.

Inputs must be one train only quota series produced by the common simulator.
The script writes no validation or test dependent state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_train(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("test_outcomes_accessed") is not False:
        raise ValueError("Quota source must prove test_outcomes_accessed=false.")
    reports = payload.get("reports", [])
    if len(reports) != 1 or reports[0].get("split") != "train":
        raise ValueError("Quota source must contain exactly one training report.")
    report = reports[0]
    arrays = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in report["series"].items()
    }
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("Training quota arrays must be equal and nonempty.")
    if np.any(arrays["capacity"] <= 0.0):
        raise ValueError("Training capacity must be positive.")
    return report, arrays


def _features(arrays: dict[str, np.ndarray]) -> np.ndarray:
    capacity = arrays["capacity"]
    time = arrays["time"]
    phase_day = 2.0 * np.pi * ((time % 86400.0) / 86400.0)
    phase_week = 2.0 * np.pi * ((time % (7.0 * 86400.0)) / (7.0 * 86400.0))
    return np.stack(
        (
            arrays["hp_used"] / capacity,
            arrays["base_quota"] / capacity,
            arrays["spot_used"] / capacity,
            arrays["hp_queue_gpu_demand"] / capacity,
            np.zeros(len(time), dtype=np.float64),
            np.sin(phase_day),
            np.cos(phase_day),
            np.sin(phase_week),
            np.cos(phase_week),
        ),
        axis=1,
    )


def _lower(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability, method="lower"))


def _base_artifact(
    *,
    policy_type: str,
    source: Path,
    qualification: str,
    risk_limit: float,
) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[1]
    try:
        source_label = str(source.resolve().relative_to(repository_root))
    except ValueError:
        source_label = str(source)
    return {
        "schema_version": "1.0",
        "policy_type": policy_type,
        "qualification": qualification,
        "fit_split": "train",
        "risk_limit": float(risk_limit),
        "source_report": source_label,
        "source_report_sha256": _sha256(source),
        "test_outcomes_accessed": False,
    }


def fit_paper_controls(
    source: Path,
    output_dir: Path,
    *,
    risk_limit: float,
    minimum_residual: float,
    maximum_residual: float,
    qualification: str,
) -> dict[str, Path]:
    """Fit fixed, diurnal, online quantile, headroom, CLQ, and guard policies."""
    if not 0.0 < risk_limit < 1.0:
        raise ValueError("risk_limit must lie in (0, 1).")
    if not -1.0 <= minimum_residual <= maximum_residual <= 1.0:
        raise ValueError("Residual bounds must satisfy -1 <= min <= max <= 1.")
    report, arrays = _load_train(source)
    evaluated_hours = float(report.get("evaluated_hours", 0.0))
    if qualification == "publication" and evaluated_hours < 1000.0:
        raise ValueError("Publication quota fit requires at least 1000 training hours.")
    capacity = arrays["capacity"]
    target = (arrays["oracle_safe_quota"] - arrays["base_quota"]) / capacity
    target = np.clip(target, -1.0, 1.0)
    outputs: dict[str, Path] = {}

    static_grid = np.linspace(minimum_residual, maximum_residual, 121)
    static_risk = np.asarray([np.mean(value > target) for value in static_grid])
    feasible = static_grid[static_risk <= risk_limit + 1e-12]
    static_value = float(feasible.max()) if len(feasible) else float(static_grid[0])
    static = _base_artifact(
        policy_type="fixed_residual",
        source=source,
        qualification=qualification,
        risk_limit=risk_limit,
    )
    static.update({
        "residual_fraction": static_value,
        "train_risk_tick_rate": float(np.mean(static_value > target)),
    })
    outputs["fixed_residual"] = output_dir / "non_rl_static_residual.json"
    _write(outputs["fixed_residual"], static)

    hour = ((arrays["time"] // 3600) % 24).astype(np.int64)
    fallback = _lower(target, risk_limit)
    residual_by_hour = [
        float(np.clip(
            _lower(target[hour == value], risk_limit)
            if np.any(hour == value)
            else fallback,
            minimum_residual,
            maximum_residual,
        ))
        for value in range(24)
    ]
    diurnal = _base_artifact(
        policy_type="hour_of_day_residual",
        source=source,
        qualification=qualification,
        risk_limit=risk_limit,
    )
    diurnal.update({
        "residual_by_hour": residual_by_hour,
        "minimum_residual": minimum_residual,
        "maximum_residual": maximum_residual,
    })
    outputs["diurnal"] = output_dir / "non_rl_hour_of_day.json"
    _write(outputs["diurnal"], diurnal)

    online_quantile = _base_artifact(
        policy_type="expanding_quantile_residual",
        source=source,
        qualification=qualification,
        risk_limit=risk_limit,
    )
    online_quantile.update({
        "initial_residuals": target.tolist(),
        "horizon_seconds": int(report["horizon_seconds"]),
        "minimum_residual": minimum_residual,
        "maximum_residual": maximum_residual,
    })
    outputs["online_quantile"] = output_dir / "non_rl_expanding_quantile.json"
    _write(outputs["online_quantile"], online_quantile)

    x = _features(arrays)
    feature_mean = x.mean(axis=0)
    feature_scale = x.std(axis=0)
    feature_scale = np.where(feature_scale > 1e-10, feature_scale, 1.0)
    z = (x - feature_mean) / feature_scale
    design = np.column_stack((z, np.ones(len(z))))
    penalty = np.eye(design.shape[1]) * 1e-2
    penalty[-1, -1] = 0.0
    solution = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    raw_prediction = design @ solution
    safety_offset = _lower(target - raw_prediction, risk_limit)
    adjusted = np.clip(
        raw_prediction + safety_offset,
        minimum_residual,
        maximum_residual,
    )
    linear = _base_artifact(
        policy_type="contextual_linear_residual",
        source=source,
        qualification=qualification,
        risk_limit=risk_limit,
    )
    linear.update({
        "coefficients": solution[:-1].tolist(),
        "intercept": float(solution[-1]),
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "safety_offset": safety_offset,
        "minimum_residual": minimum_residual,
        "maximum_residual": maximum_residual,
        "train_risk_tick_rate": float(np.mean(adjusted > target)),
        "feature_names": [
            "hp_used_fraction",
            "base_quota_fraction",
            "spot_used_fraction",
            "hp_queue_fraction",
            "spot_queue_count_proxy",
            "day_sin",
            "day_cos",
            "week_sin",
            "week_cos",
        ],
    })
    outputs["contextual_linear"] = output_dir / "non_rl_contextual_linear.json"
    _write(outputs["contextual_linear"], linear)

    future_hp_rise = (
        capacity - arrays["oracle_safe_quota"] - arrays["hp_used"]
    ) / capacity
    headroom_calibration_risk = risk_limit / 2.0
    headroom_margin = float(np.quantile(
        future_hp_rise,
        1.0 - headroom_calibration_risk,
        method="higher",
    ))
    margin = float(np.quantile(
        future_hp_rise,
        1.0 - risk_limit,
        method="higher",
    ))
    hp_queue_fraction = arrays["hp_queue_gpu_demand"] / capacity
    positive_hp_queue = hp_queue_fraction[hp_queue_fraction > 0.0]
    emergency_threshold = (
        float(np.quantile(positive_hp_queue, 0.50, method="higher"))
        if len(positive_hp_queue)
        else 1.0
    )
    headroom = _base_artifact(
        policy_type="current_free_headroom",
        source=source,
        qualification=qualification,
        risk_limit=risk_limit,
    )
    headroom.update({
        "margin_fraction": max(0.0, headroom_margin),
        "calibration_risk": headroom_calibration_risk,
        "minimum_residual": minimum_residual,
        "maximum_residual": maximum_residual,
    })
    outputs["headroom"] = output_dir / "fixed_headroom_quota.json"
    _write(outputs["headroom"], headroom)

    guarded = _base_artifact(
        policy_type="guarded_base_quota",
        source=source,
        qualification=qualification,
        risk_limit=risk_limit,
    )
    guarded.update({
        "hp_rise_margin_fraction": max(0.0, margin),
        "backlog_bonus_fraction": 0.01,
        "emergency_hp_queue_fraction": emergency_threshold,
        "minimum_residual": minimum_residual,
        "maximum_residual": maximum_residual,
    })
    outputs["guarded_base"] = output_dir / "non_rl_guarded_base.json"
    _write(outputs["guarded_base"], guarded)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--risk-limit", type=float, default=0.02)
    parser.add_argument("--minimum-residual", type=float, default=-0.15)
    parser.add_argument("--maximum-residual", type=float, default=0.15)
    parser.add_argument(
        "--qualification",
        choices=("smoke_only", "publication"),
        default="smoke_only",
    )
    args = parser.parse_args()
    outputs = fit_paper_controls(
        args.train_report.resolve(),
        args.output_dir.resolve(),
        risk_limit=args.risk_limit,
        minimum_residual=args.minimum_residual,
        maximum_residual=args.maximum_residual,
        qualification=args.qualification,
    )
    print(json.dumps({
        "status": "completed",
        "outputs": {key: str(value) for key, value in outputs.items()},
        "test_outcomes_accessed": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
