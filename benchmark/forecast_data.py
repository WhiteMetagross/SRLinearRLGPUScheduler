"""Causal data extraction for forecast benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class SnapshotHistory:
    minute_history: pd.DataFrame
    recorder_anchor: pd.Timestamp
    capacity: float
    source_metadata: dict[str, Any]
    source_sha256: str


@dataclass(frozen=True)
class OriginData:
    origin: pd.Timestamp
    causal_history: pd.DataFrame
    target_columns: tuple[str, ...]
    target_by_organization: np.ndarray
    target_sum_organization_max: np.ndarray
    target_simultaneous_pool_max: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("status") != "FROZEN":
        raise ValueError("Forecast benchmark requires a frozen environment protocol.")
    if raw.get("protocol_version") != "5.0.1":
        raise ValueError("Forecast benchmark expects environment protocol 5.0.1.")
    return raw


def load_snapshot_history(path: Path) -> SnapshotHistory:
    """Read the trusted local frozen snapshot and retain recorder history only."""

    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or payload.get("version") != "3.1":
        raise ValueError("Forecast source must be a schema 3.1 warmup snapshot.")
    metadata = payload.get("metadata")
    simulator = payload.get("sim")
    if not isinstance(metadata, dict) or simulator is None:
        raise ValueError("Forecast source snapshot is incomplete.")
    if metadata.get("forecast_provider_id") != "native_gfs":
        raise ValueError("Forecast history must come from Native GFS.")
    recorder = getattr(simulator, "recorder", None)
    history = getattr(recorder, "_org_request_wide_cache", None)
    if not isinstance(history, pd.DataFrame) or history.empty:
        raise ValueError("Snapshot has no materialized organization request history.")
    if not isinstance(history.index, pd.DatetimeIndex):
        raise TypeError("Snapshot request history requires a DatetimeIndex.")
    if not history.index.is_monotonic_increasing or not history.index.is_unique:
        raise ValueError("Snapshot request history index must be ordered and unique.")
    if history.isna().any().any():
        raise ValueError("Snapshot request history contains missing values.")
    values = history.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Snapshot request history must be finite and nonnegative.")

    cluster = getattr(simulator, "cluster", None)
    vc_dict = getattr(cluster, "vc_dict", None)
    if not isinstance(vc_dict, dict) or not vc_dict:
        raise ValueError("Snapshot has no cluster capacity mapping.")
    capacity = float(sum(float(vc.total_gpus) for vc in vc_dict.values()))
    anchor = pd.Timestamp(getattr(recorder, "train_start_timestamp"))
    result = SnapshotHistory(
        minute_history=history.copy(deep=True),
        recorder_anchor=anchor,
        capacity=capacity,
        source_metadata=dict(metadata),
        source_sha256=sha256_file(path),
    )
    del payload, simulator, recorder, history
    gc.collect()
    return result


def split_origins(
    protocol: dict[str, Any],
    *,
    split: str,
    recorder_anchor: pd.Timestamp,
    cadence_hours: int,
) -> tuple[pd.Timestamp, ...]:
    if split == "test":
        raise ValueError("Final test split is locked during forecast development.")
    try:
        bounds = protocol["split_bounds"][split]
    except KeyError as exc:
        raise ValueError(f"Unknown forecast split {split!r}.") from exc
    start = recorder_anchor + pd.Timedelta(seconds=int(bounds["score_start"]))
    end = recorder_anchor + pd.Timedelta(seconds=int(bounds["score_end"]))
    cadence = pd.Timedelta(hours=int(cadence_hours))
    if cadence <= pd.Timedelta(0):
        raise ValueError("Origin cadence must be positive.")
    return tuple(pd.date_range(start=start, end=end, freq=cadence, inclusive="left"))


def _causally_seen_columns(frame: pd.DataFrame, origin: pd.Timestamp) -> list[str]:
    history = frame.loc[frame.index < origin]
    if history.empty:
        return []
    seen = (history > 0.0).any(axis=0)
    return [str(column) for column in frame.columns[seen.to_numpy()]]


def _hourly_target(
    frame: pd.DataFrame,
    *,
    origin: pd.Timestamp,
    horizon: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    end = origin + pd.Timedelta(hours=int(horizon))
    future = frame.loc[(frame.index >= origin) & (frame.index < end)]
    if len(future) != int(horizon) * 60:
        raise ValueError(
            f"Origin {origin} requires {int(horizon) * 60} future minute rows, got {len(future)}."
        )
    offset = pd.Timedelta(minutes=int(origin.minute), seconds=int(origin.second))
    by_org = future.resample("1h", origin="start_day", offset=offset).max()
    simultaneous = future.sum(axis=1).resample(
        "1h", origin="start_day", offset=offset
    ).max()
    if len(by_org) != horizon or len(simultaneous) != horizon:
        raise ValueError("Future target aggregation produced an incomplete horizon.")
    return by_org, simultaneous.to_numpy(dtype=np.float64)


def build_origin_data(
    snapshot: SnapshotHistory,
    *,
    origin: pd.Timestamp,
    horizon: int,
    minimum_history_hours: int,
) -> OriginData:
    frame = snapshot.minute_history
    if origin <= frame.index.min() or origin > frame.index.max():
        raise ValueError(f"Origin {origin} lies outside snapshot history support.")
    columns = _causally_seen_columns(frame, origin)
    if not columns:
        raise ValueError(f"No organizations were causally observed before {origin}.")
    causal = frame.loc[frame.index < origin, columns]
    required_minutes = int(minimum_history_hours) * 60
    if len(causal) < required_minutes:
        raise ValueError(
            f"Origin {origin} has {len(causal)} history minutes, requires {required_minutes}."
        )
    if causal.index.max() >= origin:
        raise AssertionError("Causal history includes the forecast origin or future.")

    target_by_org, simultaneous = _hourly_target(
        frame,
        origin=origin,
        horizon=horizon,
    )
    target_values = target_by_org.to_numpy(dtype=np.float64)
    return OriginData(
        origin=origin,
        causal_history=causal,
        target_columns=tuple(str(column) for column in target_by_org.columns),
        target_by_organization=target_values,
        target_sum_organization_max=target_values.sum(axis=1),
        target_simultaneous_pool_max=simultaneous,
    )


def align_prediction(
    *,
    prediction: np.ndarray,
    prediction_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
) -> np.ndarray:
    """Align a causal forecast to the realized schema without exposing new names."""

    values = np.asarray(prediction, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(prediction_columns):
        raise ValueError("Prediction shape does not match prediction columns.")
    target_index = {column: index for index, column in enumerate(target_columns)}
    aligned = np.zeros((values.shape[0], len(target_columns)), dtype=np.float64)
    for source_index, column in enumerate(prediction_columns):
        if column in target_index:
            aligned[:, target_index[column]] = values[:, source_index]
    return aligned
