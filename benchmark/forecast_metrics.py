"""Metrics for causal HP demand forecasts and implied safe quota."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np


def pinball_loss(prediction: np.ndarray, target: np.ndarray, quantile: float) -> float:
    error = np.asarray(target) - np.asarray(prediction)
    return float(np.mean(np.maximum(float(quantile) * error, (float(quantile) - 1.0) * error)))


def _point_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    horizon = error.shape[1]
    return {
        "mae_gpu": float(np.mean(np.abs(error))),
        "rmse_gpu": float(np.sqrt(np.mean(np.square(error)))),
        "bias_gpu": float(np.mean(error)),
        "per_horizon": {
            str(index + 1): {
                "mae_gpu": float(np.mean(np.abs(error[:, index]))),
                "rmse_gpu": float(np.sqrt(np.mean(np.square(error[:, index])))),
                "bias_gpu": float(np.mean(error[:, index])),
            }
            for index in range(horizon)
        },
        "window_max_absolute_error_gpu": {
            str(window): float(np.max(np.abs(error[:, :window])))
            for window in (1, 2, horizon)
            if window <= horizon
        },
    }


def _quantile_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    quantile: float,
    capacity: float,
) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    shortfall = np.maximum(truth - pred, 0.0)
    excess = np.maximum(pred - truth, 0.0)
    predicted_quota = np.clip(float(capacity) - pred, 0.0, float(capacity))
    oracle_quota = np.clip(float(capacity) - truth, 0.0, float(capacity))
    return {
        "pinball_loss_gpu": pinball_loss(pred, truth, quantile),
        "empirical_coverage": float(np.mean(truth <= pred)),
        "coverage_gap": float(np.mean(truth <= pred) - quantile),
        "mean_excess_gpu": float(np.mean(excess)),
        "mean_underprediction_shortfall_gpu": float(np.mean(shortfall)),
        "maximum_underprediction_shortfall_gpu": float(np.max(shortfall)),
        "safe_quota_mae_gpu": float(np.mean(np.abs(predicted_quota - oracle_quota))),
        "unsafe_quota_overexposure_gpu": float(
            np.mean(np.maximum(predicted_quota - oracle_quota, 0.0))
        ),
        "unused_safe_quota_gpu": float(
            np.mean(np.maximum(oracle_quota - predicted_quota, 0.0))
        ),
    }


def aggregate_forecast_records(
    records: Iterable[dict[str, Any]],
    *,
    capacity: float,
) -> dict[str, Any]:
    materialized = list(records)
    point_scored = [
        record
        for record in materialized
        if bool(record.get("point_scored", record.get("scored")))
    ]
    quantile_scored = [
        record
        for record in materialized
        if bool(record.get("quantile_scored", record.get("scored")))
    ]
    if not point_scored:
        raise ValueError("No scored forecast origins are available.")
    point = np.asarray([record["pool_point"] for record in point_scored], dtype=np.float64)
    conservative = np.asarray(
        [record["target_sum_organization_max"] for record in point_scored],
        dtype=np.float64,
    )
    simultaneous = np.asarray(
        [record["target_simultaneous_pool_max"] for record in point_scored],
        dtype=np.float64,
    )
    result: dict[str, Any] = {
        "scored_origins": len(point_scored),
        "point_scored_origins": len(point_scored),
        "quantile_scored_origins": len(quantile_scored),
        "forecast_values": int(point.size),
        "target_definition": "sum_of_per_organization_hourly_maxima",
        "point": _point_metrics(point, conservative),
        "point_against_simultaneous_pool_peak": _point_metrics(point, simultaneous),
    }

    grouped: dict[tuple[str, float], list[np.ndarray]] = defaultdict(list)
    grouped_targets: dict[tuple[str, float], list[np.ndarray]] = defaultdict(list)
    for record in quantile_scored:
        for family in ("intrinsic_pool_quantiles", "conformal_pool_quantiles"):
            for raw_quantile, values in record.get(family, {}).items():
                key = (family, float(raw_quantile))
                grouped[key].append(np.asarray(values, dtype=np.float64))
                grouped_targets[key].append(
                    np.asarray(record["target_sum_organization_max"], dtype=np.float64)
                )
    for (family, quantile), predictions in sorted(grouped.items()):
        family_result = result.setdefault(family, {})
        family_result[f"q{int(round(100 * quantile))}"] = _quantile_metrics(
            np.asarray(predictions),
            np.asarray(grouped_targets[(family, quantile)]),
            quantile,
            capacity,
        )
    return result
