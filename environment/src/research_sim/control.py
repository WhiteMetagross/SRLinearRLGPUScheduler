"""Version 4 causal quota and Spot admission control contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple
import math

import numpy as np

from .types import (
    AdmissionJobView,
    ControlDecisionContext,
    InvalidActionError,
)


CONTROL_SCHEMA_VERSION = "4.0"
ADMISSION_FEATURE_NAMES = (
    "queue_wait_squashed",
    "gpu_per_worker_fraction",
    "worker_fraction_of_nodes",
    "total_gpu_demand_fraction",
    "guarantee_fraction",
    "estimated_duration_squashed",
)
ADMISSION_FEATURE_DIM = len(ADMISSION_FEATURE_NAMES)
CONTROL_GLOBAL_FEATURE_NAMES = (
    "hp_used_fraction",
    "spot_used_fraction",
    "free_fraction",
    "base_quota_fraction",
    "spot_queue_count_squashed",
    "spot_queue_demand_squashed",
    "hp_queue_count_squashed",
    "hp_queue_demand_squashed",
    "eviction_failure_rate",
    "day_sin_scaled",
    "day_cos_scaled",
    "score_progress",
)
CONTROL_GLOBAL_FEATURE_DIM = len(CONTROL_GLOBAL_FEATURE_NAMES)


def _squash_nonnegative(value: float, scale: float) -> float:
    value = max(0.0, float(value))
    scale = max(float(scale), 1.0)
    return value / (value + scale)


def native_admission_key(job: Any) -> Tuple[float, int, float]:
    """Exact stable ordering used by the maintained GFS Spot queue scan."""
    return (
        -float(job["gpu_request"]),
        int(job["worker_num"]),
        float(job["submit_time"]),
    )


def build_admission_job_view(
    job: Any,
    *,
    time: int,
    total_nodes: int,
    total_capacity: int,
    max_gpu_per_worker: int,
    max_guarantee_hours: int,
    estimated_duration_seconds: int,
) -> AdmissionJobView:
    """Build a bounded causal queue record without trace-known duration."""
    worker_num = int(job["worker_num"])
    gpu_request = float(job["gpu_request"])
    total_demand = worker_num * gpu_request
    wait = max(0.0, float(time) - float(job["submit_time"]))
    guarantee = max(0, int(job.get("guarantee_request", 1)))
    estimated_duration = max(0, int(estimated_duration_seconds))
    features = (
        _squash_nonnegative(wait, 3600.0),
        float(np.clip(gpu_request / max(1, max_gpu_per_worker), 0.0, 1.0)),
        float(np.clip(worker_num / max(1, total_nodes), 0.0, 1.0)),
        float(np.clip(total_demand / max(1, total_capacity), 0.0, 1.0)),
        float(np.clip(guarantee / max(1, max_guarantee_hours), 0.0, 1.0)),
        _squash_nonnegative(estimated_duration, 3600.0),
    )
    return AdmissionJobView(
        job_id=int(job["job_name"]),
        pool=str(job.get("vc_name", job.get("gpu_model", ""))),
        submit_time=int(job["submit_time"]),
        worker_num=worker_num,
        gpu_request=gpu_request,
        total_gpu_demand=total_demand,
        guarantee_request=guarantee,
        estimated_duration_seconds=estimated_duration,
        features=features,
    )


@dataclass(frozen=True)
class UnifiedControlAction:
    """Bounded v4 action applied for one GFS quota interval."""

    quota_delta_fraction: float
    priority_weights: Tuple[float, ...]

    @classmethod
    def from_array(cls, action: Sequence[float]) -> "UnifiedControlAction":
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (1 + ADMISSION_FEATURE_DIM,):
            raise InvalidActionError(
                f"Control action must have shape {(1 + ADMISSION_FEATURE_DIM,)}, got {values.shape}."
            )
        if not np.all(np.isfinite(values)):
            raise InvalidActionError("Control action must contain only finite values.")
        if np.any(values < -1.0) or np.any(values > 1.0):
            raise InvalidActionError("Every normalized control action component must be within [-1, 1].")
        return cls(
            quota_delta_fraction=float(values[0]),
            priority_weights=tuple(float(v) for v in values[1:]),
        )


class WeightedAdmissionRanker:
    """Ranks sanitized queue records with fixed weights and native tie breaks."""

    ranker_id = "weighted_causal_v1"

    def __init__(self) -> None:
        self._weights = np.zeros(ADMISSION_FEATURE_DIM, dtype=np.float64)

    @property
    def weights(self) -> Tuple[float, ...]:
        return tuple(float(v) for v in self._weights)

    def set_weights(self, weights: Sequence[float]) -> None:
        values = np.asarray(weights, dtype=np.float64)
        if values.shape != (ADMISSION_FEATURE_DIM,):
            raise InvalidActionError(
                f"Priority weights must have shape {(ADMISSION_FEATURE_DIM,)}, got {values.shape}."
            )
        if not np.all(np.isfinite(values)) or np.any(np.abs(values) > 1.0):
            raise InvalidActionError("Priority weights must be finite and within [-1, 1].")
        self._weights = values.copy()

    def order(self, jobs: Sequence[Any], views: Mapping[int, AdmissionJobView]) -> Tuple[int, ...]:
        expected = {int(job["job_name"]) for job in jobs}
        if set(views) != expected:
            raise InvalidActionError("Admission views must cover the queue exactly.")

        def key(job: Any) -> Tuple[float, float, int, float]:
            job_id = int(job["job_name"])
            score = float(np.dot(self._weights, np.asarray(views[job_id].features)))
            native = native_admission_key(job)
            return (-score, native[0], native[1], native[2])

        return tuple(int(job["job_name"]) for job in sorted(jobs, key=key))


def apply_quota_delta(
    context: ControlDecisionContext,
    action: UnifiedControlAction,
    *,
    max_delta_fraction: float,
) -> Dict[str, Dict[int, int]]:
    """Apply one normalized residual to every pool's causal base quota."""
    if not math.isfinite(max_delta_fraction) or not 0.0 < max_delta_fraction <= 1.0:
        raise InvalidActionError("max_delta_fraction must be finite and within (0, 1].")
    adjusted: Dict[str, Dict[int, int]] = {}
    for pool, base_by_hour in context.base_quota_by_pool.items():
        capacity = int(context.capacity_by_pool[pool])
        delta = action.quota_delta_fraction * max_delta_fraction * capacity
        adjusted[pool] = {
            int(hour): int(np.clip(np.rint(float(base) + delta), 0, capacity))
            for hour, base in base_by_hour.items()
        }
    return adjusted


def build_control_observation(
    context: ControlDecisionContext,
    *,
    queue_size: int,
) -> Dict[str, np.ndarray]:
    """Create the fixed-shape v4 observation from a causal control context."""
    total_capacity = max(1, sum(context.capacity_by_pool.values()))
    hp_used = sum(context.hp_used_by_pool.values())
    spot_used = sum(context.spot_used_by_pool.values())
    free = max(0.0, total_capacity - hp_used - spot_used)
    base_quota = sum(
        min(hours.values()) if hours else 0
        for hours in context.base_quota_by_pool.values()
    )
    spot_queue_demand = sum(job.total_gpu_demand for job in context.admission_jobs)
    eviction_rates = tuple(context.eviction_failure_rate_by_pool.values())
    eviction_rate = float(np.mean(eviction_rates)) if eviction_rates else 0.0
    day_fraction = (context.time % 86400) / 86400.0
    global_features = np.asarray((
        np.clip(hp_used / total_capacity, 0.0, 1.0),
        np.clip(spot_used / total_capacity, 0.0, 1.0),
        np.clip(free / total_capacity, 0.0, 1.0),
        np.clip(base_quota / total_capacity, 0.0, 1.0),
        _squash_nonnegative(len(context.admission_jobs), queue_size),
        _squash_nonnegative(spot_queue_demand, total_capacity),
        _squash_nonnegative(context.hp_queue_count, queue_size),
        _squash_nonnegative(context.hp_queue_gpu_demand, total_capacity),
        np.clip(eviction_rate, 0.0, 1.0),
        0.5 + 0.5 * math.sin(2.0 * math.pi * day_fraction),
        0.5 + 0.5 * math.cos(2.0 * math.pi * day_fraction),
        np.clip(context.score_progress, 0.0, 1.0),
    ), dtype=np.float32)
    queue_features = np.zeros((queue_size, ADMISSION_FEATURE_DIM), dtype=np.float32)
    queue_mask = np.zeros(queue_size, dtype=bool)
    for index, job in enumerate(context.admission_jobs[:queue_size]):
        queue_features[index] = np.asarray(job.features, dtype=np.float32)
        queue_mask[index] = True
    return {
        "global_features": global_features,
        "queue_features": queue_features,
        "queue_mask": queue_mask,
    }
