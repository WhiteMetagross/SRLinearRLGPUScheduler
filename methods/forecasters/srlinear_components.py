"""Causal feature and ridge initialization components for SRLinear.

SRLinear keeps one exact weekly seasonal skip connection and learns only a
ridge-regularized correction from other causal seasonal anchors, recent levels,
recent slopes, and calendar features.  A second head forecasts aggregate demand;
These private helpers build seasonal anchors and initialize the learnable model.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class _Design:
    features: np.ndarray
    base: np.ndarray
    residual: np.ndarray
    positions: np.ndarray
    channels: int


@dataclass(frozen=True)
class _RidgeHead:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    alpha: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        standardized = (features - self.feature_mean) / self.feature_scale
        return standardized @ self.coefficients + self.intercept


def _validate_structure(
    *,
    horizon: int,
    periods: tuple[int, ...],
    base_period: int,
    recent_windows: tuple[int, ...],
) -> None:
    if horizon <= 0:
        raise ValueError("SRLinear horizon must be positive.")
    if base_period not in periods:
        raise ValueError("SRLinear base period must be one of the seasonal periods.")
    if any(period <= horizon - 1 for period in periods):
        raise ValueError("SRLinear seasonal periods must exceed the forecast horizon.")
    if any(window <= 0 for window in recent_windows):
        raise ValueError("SRLinear recent windows must be positive.")


def _positions(
    *,
    length: int,
    history_required: int,
    horizon: int,
    stride: int,
    start: int | None = None,
) -> np.ndarray:
    first = max(int(history_required), int(start or 0))
    stop = int(length) - int(horizon) + 1
    if first >= stop:
        return np.empty(0, dtype=np.int64)
    return np.arange(first, stop, int(stride), dtype=np.int64)


def _calendar_columns(target_times: pd.DatetimeIndex) -> tuple[np.ndarray, ...]:
    hour = target_times.hour.to_numpy(dtype=np.float64)
    weekday = target_times.dayofweek.to_numpy(dtype=np.float64)
    weekend = (weekday >= 5.0).astype(np.float64)
    return (
        np.sin(2.0 * np.pi * hour / 24.0),
        np.cos(2.0 * np.pi * hour / 24.0),
        np.sin(2.0 * np.pi * weekday / 7.0),
        np.cos(2.0 * np.pi * weekday / 7.0),
        weekend,
    )


def _build_design(
    values: np.ndarray,
    timestamps: pd.DatetimeIndex,
    *,
    positions: np.ndarray,
    horizon_step: int,
    periods: tuple[int, ...],
    base_period: int,
    recent_windows: tuple[int, ...],
    residual_lookback: int,
    calendar_offset: pd.Timedelta,
    include_target: bool,
) -> _Design:
    """Build causal organization-pooled features for one direct horizon."""

    values = np.asarray(values, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("SRLinear values require shape [time, organization].")
    if len(timestamps) != values.shape[0]:
        raise ValueError("SRLinear values and timestamps do not align.")
    if positions.size == 0:
        raise ValueError("SRLinear design has no valid positions.")
    if positions.min() < max(
        max(periods), max(recent_windows), int(residual_lookback)
    ):
        raise ValueError("SRLinear design position lacks required causal history.")
    if positions.max() + horizon_step >= len(values) and include_target:
        raise ValueError("SRLinear target crosses available history.")
    if positions.max() + horizon_step - min(periods) >= len(values):
        raise ValueError("SRLinear seasonal anchor crosses available history.")

    base = values[positions + int(horizon_step) - int(base_period)]
    seasonal_deltas = [
        values[positions + int(horizon_step) - int(period)] - base
        for period in periods
        if period != base_period
    ]
    last = values[positions - 1]
    cumulative = np.vstack(
        (np.zeros((1, values.shape[1]), dtype=np.float64), np.cumsum(values, axis=0))
    )
    recent_means = [
        (cumulative[positions] - cumulative[positions - int(window)]) / float(window)
        for window in recent_windows
    ]
    recent_slopes = [last - values[positions - int(window)] for window in recent_windows]

    if positions.max() < len(timestamps):
        target_times = pd.DatetimeIndex(timestamps[positions]) + calendar_offset
        target_times = target_times + pd.to_timedelta(int(horizon_step), unit="h")
    else:
        raise ValueError("SRLinear inference target timestamps must be supplied in history.")
    calendar = _calendar_columns(target_times)
    weekend = calendar[-1][:, None]

    local_residuals = [
        values[positions - lag] - base
        for lag in range(int(residual_lookback), 0, -1)
    ]
    feature_columns: list[np.ndarray] = []
    feature_columns.extend(seasonal_deltas)
    feature_columns.extend(delta * weekend for delta in seasonal_deltas)
    feature_columns.append(last - base)
    feature_columns.extend(mean - base for mean in recent_means)
    feature_columns.extend(recent_slopes)
    feature_columns.extend(local_residuals)
    for column in calendar:
        feature_columns.append(np.broadcast_to(column[:, None], base.shape))
    features = np.stack(feature_columns, axis=-1)

    if include_target:
        residual = values[positions + int(horizon_step)] - base
    else:
        residual = np.zeros_like(base)
    return _Design(
        features=features.reshape(-1, features.shape[-1]),
        base=base.reshape(-1),
        residual=residual.reshape(-1),
        positions=positions,
        channels=values.shape[1],
    )


def _build_inference_design(
    values: np.ndarray,
    *,
    origin: pd.Timestamp,
    calendar_timestamp: pd.Timestamp,
    horizon_step: int,
    periods: tuple[int, ...],
    base_period: int,
    recent_windows: tuple[int, ...],
    residual_lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one strictly causal feature row per organization at the origin."""

    values = np.asarray(values, dtype=np.float64)
    position = len(values)
    if position < max(max(periods), max(recent_windows), int(residual_lookback)):
        raise ValueError("SRLinear inference lacks required causal history.")
    base = values[position + int(horizon_step) - int(base_period)][None, :]
    seasonal_deltas = [
        values[position + int(horizon_step) - int(period)][None, :] - base
        for period in periods
        if period != base_period
    ]
    last = values[position - 1][None, :]
    recent_means = [values[position - int(window) : position].mean(axis=0, keepdims=True) for window in recent_windows]
    recent_slopes = [last - values[position - int(window)][None, :] for window in recent_windows]
    target_time = pd.DatetimeIndex(
        [pd.Timestamp(calendar_timestamp) + pd.Timedelta(hours=int(horizon_step))]
    )
    calendar = _calendar_columns(target_time)
    weekend = calendar[-1][:, None]

    local_residuals = [
        values[position - lag][None, :] - base
        for lag in range(int(residual_lookback), 0, -1)
    ]
    feature_columns: list[np.ndarray] = []
    feature_columns.extend(seasonal_deltas)
    feature_columns.extend(delta * weekend for delta in seasonal_deltas)
    feature_columns.append(last - base)
    feature_columns.extend(mean - base for mean in recent_means)
    feature_columns.extend(recent_slopes)
    feature_columns.extend(local_residuals)
    for column in calendar:
        feature_columns.append(np.broadcast_to(column[:, None], base.shape))
    features = np.stack(feature_columns, axis=-1).reshape(values.shape[1], -1)
    return features, base.reshape(-1)


def _fit_ridge(features: np.ndarray, targets: np.ndarray, alpha: float) -> _RidgeHead:
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    if features.ndim != 2 or len(features) != len(targets):
        raise ValueError("SRLinear ridge features and targets do not align.")
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale = np.where(feature_scale > 1e-8, feature_scale, 1.0)
    standardized = (features - feature_mean) / feature_scale
    target_mean = float(targets.mean())
    centered_target = targets - target_mean
    sample_count = max(1, len(targets))
    gram = standardized.T @ standardized / sample_count
    right = standardized.T @ centered_target / sample_count
    regularized = gram + float(alpha) * np.eye(gram.shape[0], dtype=np.float64)
    try:
        coefficients = np.linalg.solve(regularized, right)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(regularized, right, rcond=None)[0]
    return _RidgeHead(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coefficients=coefficients,
        intercept=target_mean,
        alpha=float(alpha),
    )


def _select_head(
    train: _Design,
    validation: _Design,
    *,
    alphas: tuple[float, ...],
    means: np.ndarray,
    scales: np.ndarray,
) -> tuple[_RidgeHead, np.ndarray, float]:
    if not alphas:
        raise ValueError("SRLinear requires at least one ridge alpha.")
    means = np.asarray(means, dtype=np.float64).reshape(-1)
    scales = np.asarray(scales, dtype=np.float64).reshape(-1)
    if train.channels != len(means) or train.channels != len(scales):
        raise ValueError("SRLinear scale width does not match design channels.")
    validation_base = validation.base.reshape(-1, validation.channels)
    validation_truth = (
        validation.base + validation.residual
    ).reshape(-1, validation.channels)
    best: tuple[float, float, _RidgeHead, np.ndarray] | None = None
    for alpha in alphas:
        head = _fit_ridge(train.features, train.residual, float(alpha))
        residual_prediction = head.predict(validation.features).reshape(
            -1, validation.channels
        )
        prediction = np.clip(
            (validation_base + residual_prediction) * scales[None, :] + means[None, :],
            0.0,
            None,
        )
        truth = validation_truth * scales[None, :] + means[None, :]
        mae = float(np.abs(prediction - truth).mean())
        candidate = (mae, float(alpha), head, prediction)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    return best[2], best[3], best[0]
