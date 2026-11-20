#!/usr/bin/env python3
"""Fit the supported contextual quota control from training data only."""

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


def _load_training_series(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("test_outcomes_accessed") is not False:
        raise ValueError("Training source must prove that test outcomes were not accessed.")
    reports = payload.get("reports", [])
    if len(reports) != 1 or reports[0].get("split") != "train":
        raise ValueError("Training source must contain exactly one training report.")
    arrays = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in reports[0]["series"].items()
    }
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("Training arrays must have equal nonzero length.")
    return arrays


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


def fit_contextual_control(
    source: Path,
    output: Path,
    *,
    risk_limit: float,
    minimum_residual: float,
    maximum_residual: float,
    qualification: str,
) -> dict[str, Any]:
    if not 0.0 < risk_limit < 1.0:
        raise ValueError("Risk limit must lie in (0, 1).")
    arrays = _load_training_series(source)
    capacity = arrays["capacity"]
    target = np.clip(
        (arrays["oracle_safe_quota"] - arrays["base_quota"]) / capacity,
        -1.0,
        1.0,
    )
    features = _features(arrays)
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale = np.where(feature_scale > 1e-10, feature_scale, 1.0)
    standardized = (features - feature_mean) / feature_scale
    design = np.column_stack((standardized, np.ones(len(standardized))))
    penalty = np.eye(design.shape[1]) * 1e-2
    penalty[-1, -1] = 0.0
    solution = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    raw_prediction = design @ solution
    safety_offset = float(np.quantile(
        target - raw_prediction, risk_limit, method="lower"
    ))
    adjusted = np.clip(
        raw_prediction + safety_offset, minimum_residual, maximum_residual
    )
    future_hp_rise = (
        capacity - arrays["oracle_safe_quota"] - arrays["hp_used"]
    ) / capacity
    native_safe_margin = float(np.quantile(
        future_hp_rise, 1.0 - risk_limit, method="higher"
    ))
    try:
        source_label = str(source.resolve().relative_to(Path(__file__).parents[1]))
    except ValueError:
        source_label = source.name
    artifact = {
        "schema_version": "1.0",
        "policy_type": "contextual_linear_residual",
        "qualification": qualification,
        "fit_split": "train",
        "risk_limit": float(risk_limit),
        "source_report": source_label,
        "source_report_sha256": _sha256(source),
        "test_outcomes_accessed": False,
        "coefficients": solution[:-1].tolist(),
        "intercept": float(solution[-1]),
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "safety_offset": safety_offset,
        "minimum_residual": float(minimum_residual),
        "maximum_residual": float(maximum_residual),
        "train_risk_tick_rate": float(np.mean(adjusted > target)),
        "native_safe_margin_fraction": max(0.0, native_safe_margin),
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
    }
    _write(output, artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--risk-limit", type=float, default=0.02)
    parser.add_argument("--minimum-residual", type=float, default=-0.15)
    parser.add_argument("--maximum-residual", type=float, default=0.15)
    parser.add_argument(
        "--qualification", choices=("smoke_only", "publication"), default="smoke_only"
    )
    args = parser.parse_args()
    artifact = fit_contextual_control(
        args.train_report.resolve(),
        args.output.resolve(),
        risk_limit=args.risk_limit,
        minimum_residual=args.minimum_residual,
        maximum_residual=args.maximum_residual,
        qualification=args.qualification,
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "train_risk_tick_rate": artifact["train_risk_tick_rate"],
        "test_outcomes_accessed": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
