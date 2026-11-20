"""Censoring-aware publication metrics derived only from replay events."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Iterable

import numpy as np


PUBLICATION_METRICS_VERSION = "1.1"
RESTRICTED_HORIZON_SECONDS = 3600


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    data = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    if data.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    return {
        "count": int(data.size),
        "mean": float(np.mean(data)),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
    }


def _add_interval(
    daily: list[dict[str, Any]],
    start: int,
    end: int,
    score_start: int,
    score_end: int,
    field: str,
    rate: float,
) -> None:
    left = max(int(start), int(score_start))
    right = min(int(end), int(score_end))
    while left < right:
        index = (left - score_start) // 86400
        day_end = min(right, score_start + (index + 1) * 86400)
        daily[index][field] += float(day_end - left) * float(rate) / 3600.0
        left = day_end


def derive_publication_metrics(
    events: list[dict[str, Any]],
    spec: Any,
    *,
    total_capacity_gpu: float,
) -> dict[str, Any]:
    """Compute cohort, occupancy, queue, and daily metrics without future features."""
    if not events:
        raise ValueError("Publication metrics require a nonempty event stream.")
    if total_capacity_gpu <= 0 or not math.isfinite(total_capacity_gpu):
        raise ValueError("total_capacity_gpu must be finite and positive.")
    event_horizon_end = int(events[-1]["time"])
    observation_end = min(int(spec.score_end), event_horizon_end)
    observation_end = max(int(spec.score_start), observation_end)
    days = int(math.ceil((spec.score_end - spec.score_start) / 86400.0))
    anchor = datetime.fromisoformat(spec.calendar_anchor)
    split_start = anchor + timedelta(seconds=int(spec.score_start))
    daily: list[dict[str, Any]] = [
        {
            "day_index": index,
            "date": (split_start + timedelta(days=index)).date().isoformat(),
            "useful_spot_gpu_hours": 0.0,
            "lost_spot_gpu_hours": 0.0,
            "hp_allocated_gpu_hours": 0.0,
            "spot_allocated_gpu_hours": 0.0,
            "hp_queued_gpu_hours": 0.0,
            "spot_queued_gpu_hours": 0.0,
            "hp_arrivals": 0,
            "spot_arrivals": 0,
            "hp_completions": 0,
            "spot_completions": 0,
            "spot_evictions": 0,
            "lease_expirations": 0,
            "spot_lease_expirations": 0,
            "hp_lease_expirations": 0,
            "lease_turnover_overhead_gpu_hours": 0.0,
            "hp_gang_cohort": 0,
            "hp_gang_violations": 0,
            "observed_hours": max(
                0.0,
                (
                    min(observation_end, spec.score_start + (index + 1) * 86400)
                    - max(spec.score_start, spec.score_start + index * 86400)
                ) / 3600.0,
            ),
        }
        for index in range(days)
    ]

    jobs: dict[Any, dict[str, Any]] = {}
    running: dict[Any, tuple[str, float]] = {}
    queued: dict[Any, tuple[str, float]] = {}
    last_time = int(events[0]["time"])

    def event_day(time_value: int) -> int | None:
        if spec.score_start <= time_value < spec.score_end:
            return int((time_value - spec.score_start) // 86400)
        return None

    for event in events:
        time_value = int(event["time"])
        if time_value < last_time:
            raise ValueError("Publication metric event times must be monotonic.")
        hp_running = sum(value[1] for value in running.values() if value[0] == "HP")
        spot_running = sum(value[1] for value in running.values() if value[0] == "Spot")
        hp_queued = sum(value[1] for value in queued.values() if value[0] == "HP")
        spot_queued = sum(value[1] for value in queued.values() if value[0] == "Spot")
        _add_interval(daily, last_time, time_value, spec.score_start, spec.score_end,
                      "hp_allocated_gpu_hours", hp_running)
        _add_interval(daily, last_time, time_value, spec.score_start, spec.score_end,
                      "spot_allocated_gpu_hours", spot_running)
        _add_interval(daily, last_time, time_value, spec.score_start, spec.score_end,
                      "hp_queued_gpu_hours", hp_queued)
        _add_interval(daily, last_time, time_value, spec.score_start, spec.score_end,
                      "spot_queued_gpu_hours", spot_queued)
        last_time = time_value

        details = event["details"]
        event_type = event["event_type"]
        job_id = event.get("job_id")
        day = event_day(time_value)
        if event_type == "arrival":
            job_type = str(details["job_type"])
            demand = float(details["worker_num"]) * float(details["gpu_request"])
            jobs[job_id] = {
                "job_type": job_type,
                "demand": demand,
                "arrival": int(details["executed_arrival_time"]),
                "first_start": None,
                "completion": None,
                "queue_since": time_value,
                "total_queue_seconds": 0.0,
                "hp_cohort_eligible": bool(details.get("hp_cohort_eligible", False)),
                "hp_outcome": None,
            }
            queued[job_id] = (job_type, demand)
            if day is not None:
                daily[day]["hp_arrivals" if job_type == "HP" else "spot_arrivals"] += 1
        elif event_type == "start" and details.get("complete_start", False):
            job = jobs.get(job_id)
            if job is None:
                raise ValueError(f"Start references unknown job {job_id!r}.")
            if job["first_start"] is None:
                job["first_start"] = time_value
            if job["queue_since"] is not None:
                job["total_queue_seconds"] += time_value - int(job["queue_since"])
                job["queue_since"] = None
            queued.pop(job_id, None)
            running[job_id] = (job["job_type"], job["demand"])
        elif event_type == "preemption":
            job = jobs.get(job_id)
            if job is None or job["job_type"] != "Spot":
                raise ValueError(f"Preemption references unknown non-Spot job {job_id!r}.")
            running.pop(job_id, None)
            queued[job_id] = ("Spot", job["demand"])
            job["queue_since"] = time_value
            if day is not None:
                daily[day]["spot_evictions"] += 1
                daily[day]["lost_spot_gpu_hours"] += float(details["lost_gpu_seconds"]) / 3600.0
        elif event_type == "lease_expiration":
            job = jobs.get(job_id)
            if job is None:
                raise ValueError(f"Lease expiration references unknown job {job_id!r}.")
            running.pop(job_id, None)
            queued[job_id] = (job["job_type"], job["demand"])
            job["queue_since"] = time_value
            if day is not None:
                daily[day]["lease_expirations"] += 1
                daily[day][f"{job['job_type'].lower()}_lease_expirations"] += 1
                daily[day]["lease_turnover_overhead_gpu_hours"] += (
                    float(details.get("overhead_seconds", 0.0)) * job["demand"] / 3600.0
                )
        elif event_type == "completion":
            job = jobs.get(job_id)
            if job is None:
                raise ValueError(f"Completion references unknown job {job_id!r}.")
            job["completion"] = time_value
            running.pop(job_id, None)
            queued.pop(job_id, None)
            if day is not None:
                field = "hp_completions" if job["job_type"] == "HP" else "spot_completions"
                daily[day][field] += 1
        elif event_type == "credit" and day is not None:
            daily[day]["useful_spot_gpu_hours"] += (
                float(details["credited_gpu_seconds"]) / 3600.0
            )
        elif event_type == "deadline_outcome":
            job = jobs.get(job_id)
            if job is not None:
                job["hp_outcome"] = details["outcome"]

    _add_interval(daily, last_time, observation_end, spec.score_start, spec.score_end,
                  "hp_allocated_gpu_hours", sum(v[1] for v in running.values() if v[0] == "HP"))
    _add_interval(daily, last_time, observation_end, spec.score_start, spec.score_end,
                  "spot_allocated_gpu_hours", sum(v[1] for v in running.values() if v[0] == "Spot"))
    _add_interval(daily, last_time, observation_end, spec.score_start, spec.score_end,
                  "hp_queued_gpu_hours", sum(v[1] for v in queued.values() if v[0] == "HP"))
    _add_interval(daily, last_time, observation_end, spec.score_start, spec.score_end,
                  "spot_queued_gpu_hours", sum(v[1] for v in queued.values() if v[0] == "Spot"))

    cohort: dict[str, dict[str, Any]] = {}
    for job_type in ("HP", "Spot"):
        rows = [
            job for job in jobs.values()
            if job["job_type"] == job_type
            and spec.score_start <= job["arrival"] < spec.score_end
        ]
        completed = [job for job in rows if job["completion"] is not None]
        started = [job for job in rows if job["first_start"] is not None]
        restricted_rows = [
            job for job in rows
            if job["arrival"] + RESTRICTED_HORIZON_SECONDS <= event_horizon_end
        ]
        restricted_jct = [
            min(
                RESTRICTED_HORIZON_SECONDS,
                max(0, int(job["completion"]) - job["arrival"])
                if job["completion"] is not None
                else RESTRICTED_HORIZON_SECONDS,
            )
            for job in restricted_rows
        ]
        restricted_wait = [
            min(
                RESTRICTED_HORIZON_SECONDS,
                max(0, int(job["first_start"]) - job["arrival"])
                if job["first_start"] is not None
                else RESTRICTED_HORIZON_SECONDS,
            )
            for job in restricted_rows
        ]
        cohort[job_type.lower()] = {
            "arrivals": len(rows),
            "observed_completions": len(completed),
            "observed_completion_rate": len(completed) / len(rows) if rows else None,
            "restricted_1h_eligible_arrivals": len(restricted_rows),
            "restricted_1h_censored_arrivals": len(rows) - len(restricted_rows),
            "completion_within_1h_rate": (
                sum(
                    1 for job in restricted_rows
                    if job["completion"] is not None
                    and int(job["completion"]) - job["arrival"] <= RESTRICTED_HORIZON_SECONDS
                ) / len(restricted_rows)
                if restricted_rows else None
            ),
            "first_start_within_1h_rate": (
                sum(
                    1 for job in restricted_rows
                    if job["first_start"] is not None
                    and int(job["first_start"]) - job["arrival"] <= RESTRICTED_HORIZON_SECONDS
                ) / len(restricted_rows)
                if restricted_rows else None
            ),
            "completed_only_jct_seconds": _summary(
                int(job["completion"]) - job["arrival"] for job in completed
            ),
            "started_only_initial_queue_seconds": _summary(
                int(job["first_start"]) - job["arrival"] for job in started
            ),
            "completed_only_total_queue_seconds": _summary(
                job["total_queue_seconds"] for job in completed
            ),
            "restricted_1h_time_to_completion_seconds": _summary(restricted_jct),
            "restricted_1h_time_to_first_start_seconds": _summary(restricted_wait),
        }

    for job in jobs.values():
        if not job["hp_cohort_eligible"]:
            continue
        day = event_day(job["arrival"])
        if day is not None:
            daily[day]["hp_gang_cohort"] += 1
            if job["hp_outcome"] == "missed":
                daily[day]["hp_gang_violations"] += 1

    declared_score_hours = (spec.score_end - spec.score_start) / 3600.0
    score_hours = (observation_end - spec.score_start) / 3600.0
    hp_allocated = math.fsum(row["hp_allocated_gpu_hours"] for row in daily)
    spot_allocated = math.fsum(row["spot_allocated_gpu_hours"] for row in daily)
    useful_spot = math.fsum(row["useful_spot_gpu_hours"] for row in daily)
    lost_spot = math.fsum(row["lost_spot_gpu_hours"] for row in daily)
    hp_gang_cohort = sum(int(row["hp_gang_cohort"]) for row in daily)
    hp_gang_violations = sum(int(row["hp_gang_violations"]) for row in daily)
    spot_evictions = sum(int(row["spot_evictions"]) for row in daily)
    lease_expirations = sum(int(row["lease_expirations"]) for row in daily)
    spot_lease_expirations = sum(int(row["spot_lease_expirations"]) for row in daily)
    hp_lease_expirations = sum(int(row["hp_lease_expirations"]) for row in daily)
    lease_turnover_overhead = math.fsum(
        row["lease_turnover_overhead_gpu_hours"] for row in daily
    )
    for row in daily:
        denominator = row["hp_gang_cohort"]
        row["hp_gang_violation_rate"] = (
            row["hp_gang_violations"] / denominator if denominator else None
        )
        row["idle_gpu_hours"] = (
            total_capacity_gpu * row["observed_hours"]
            - row["hp_allocated_gpu_hours"]
            - row["spot_allocated_gpu_hours"]
        )
    capacity_hours = total_capacity_gpu * score_hours
    return {
        "schema_version": PUBLICATION_METRICS_VERSION,
        "restricted_horizon_seconds": RESTRICTED_HORIZON_SECONDS,
        "cohort_definition": "Jobs with arrival in [score_start, score_end).",
        "censoring_note": (
            "Completed-only JCT, queue, and observed completion summaries are diagnostic. "
            "Restricted 1h metrics include only arrivals with a fully observed 1h horizon."
        ),
        "total_capacity_gpu": float(total_capacity_gpu),
        "declared_score_hours": declared_score_hours,
        "score_hours": score_hours,
        "observation_complete": observation_end >= spec.score_end,
        "observation_end": observation_end,
        "event_horizon_end": event_horizon_end,
        "useful_spot_gpu_hours": useful_spot,
        "lost_spot_gpu_hours": lost_spot,
        "spot_evictions": spot_evictions,
        "lease_expirations": lease_expirations,
        "spot_lease_expirations": spot_lease_expirations,
        "hp_lease_expirations": hp_lease_expirations,
        "lease_turnover_overhead_gpu_hours": lease_turnover_overhead,
        "hp_gang_cohort_count": hp_gang_cohort,
        "hp_gang_violation_count": hp_gang_violations,
        "hp_gang_violation_rate": (
            hp_gang_violations / hp_gang_cohort if hp_gang_cohort else None
        ),
        "hp_allocated_gpu_hours": hp_allocated,
        "spot_allocated_gpu_hours": spot_allocated,
        "idle_gpu_hours": capacity_hours - hp_allocated - spot_allocated,
        "hp_mean_allocation_fraction": hp_allocated / capacity_hours if capacity_hours else None,
        "spot_mean_allocation_fraction": spot_allocated / capacity_hours if capacity_hours else None,
        "total_mean_allocation_fraction": (
            (hp_allocated + spot_allocated) / capacity_hours if capacity_hours else None
        ),
        "hp_queued_gpu_hours": math.fsum(row["hp_queued_gpu_hours"] for row in daily),
        "spot_queued_gpu_hours": math.fsum(row["spot_queued_gpu_hours"] for row in daily),
        "cohorts": cohort,
        "daily": daily,
    }
