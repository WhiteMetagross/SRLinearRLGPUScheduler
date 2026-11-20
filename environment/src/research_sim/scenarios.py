"""
scenarios.py: Past-fitted future high-priority gang scenario generator and
class-specific causal duration estimators.
Fitted on past chronological training blocks only.
Preserves bursts and task shapes via conditional block bootstrap.
Never reads future trace arrivals beyond the decision time t_k.
"""

import math
import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Set
import numpy as np
import pandas as pd
from .types import ClusterStateSnapshot, QueuedHPJob


@dataclass(frozen=True)
class ScenarioJob:
    job_id: int
    submit_time: int       # Arrival time a_i in sim seconds relative to origin
    worker_num: int        # w_i
    gpu_request: int       # g_i (whole cards)
    gpu_model: str         # m_i
    duration: int          # Estimated duration \hat{d}_i in seconds
    is_queued: bool = False # True if already queued at t_k


@dataclass(frozen=True)
class Scenario:
    scenario_id: int
    seed: int
    start_time: int        # Decision time t_k
    horizon_seconds: int   # H_s
    jobs: Tuple[ScenarioJob, ...] # Queued jobs + sampled future arrivals
    provenance: Dict[str, Any] = field(default_factory=dict)


def derive_scenario_seed(
    shared_scenario_seed: int,
    split_id: str,
    replay_second: int,
    offered_job_id: int,
    offer_ordinal: int,
    scenario_index: int
) -> int:
    """
    Derives scenario random seeds from a stable policy-independent key.
    Includes shared scenario seed, split ID, replay second, offered job ID,
    offer ordinal, and scenario index.
    Does NOT include policy-specific run or episode IDs.
    Does NOT use Python hash().
    """
    key_str = f"{shared_scenario_seed}:{split_id}:{replay_second}:{offered_job_id}:{offer_ordinal}:{scenario_index}"
    h = hashlib.sha256(key_str.encode("utf-8")).hexdigest()
    return int(h[:15], 16)


def build_observed_completion_history(
    completion_events: List[Dict[str, Any]],
    jobs_df: pd.DataFrame,
    fit_cutoff: int
) -> pd.DataFrame:
    """
    Creates an observed completion history input built from executed completion events
    strictly before fit_cutoff.
    Verifies that:
    - Each completed job has a unique executed completion event strictly before fit_cutoff.
    - Each completed job joins cleanly with jobs_df by job_id/job_name.
    - Preserves arrival and completion times, job class, worker count, GPU request, model,
      trace row ID, and event ID.
    Raises ValueError with diagnostics on duplicates or missing joins.
    """
    type_col = "job_type" if "job_type" in jobs_df.columns else ("type" if "type" in jobs_df.columns else None)
    if type_col is None:
        raise ValueError("Input jobs_df missing required job type column ('job_type' or 'type').")

    id_col = "job_name" if "job_name" in jobs_df.columns else ("job_id" if "job_id" in jobs_df.columns else None)
    if id_col is None:
        raise ValueError("Input jobs_df missing required job id column ('job_name' or 'job_id').")

    # Map job_id to (index, Series)
    jobs_map = {}
    for idx, row in jobs_df.iterrows():
        jid = int(row[id_col])
        if jid not in jobs_map:
            jobs_map[jid] = (idx, row)

    seen_completed_ids: Set[int] = set()
    records: List[Dict[str, Any]] = []

    for ev_idx, ev in enumerate(completion_events):
        ev_type = ev.get("event_type")
        ev_time = ev.get("time")
        if ev_type != "completion":
            continue
        if ev_time is None or ev_time >= fit_cutoff:
            continue

        jid = ev.get("job_id")
        if jid is None:
            details = ev.get("details", {})
            jid = details.get("job_id")
        if jid is None:
            raise ValueError(f"Completion event at index {ev_idx} missing job_id: {ev}")
        jid = int(jid)

        if jid in seen_completed_ids:
            raise ValueError(
                f"Duplicate completion event detected for job_id {jid} before cutoff {fit_cutoff}. "
                f"Previous event already recorded; second event at time {ev_time}."
            )
        seen_completed_ids.add(jid)

        if jid not in jobs_map:
            raise ValueError(
                f"Completed job_id {jid} from completion event at time {ev_time} "
                f"not found in jobs_df (available rows: {len(jobs_df)})."
            )

        row_idx, job_row = jobs_map[jid]
        dur = float(job_row["duration"]) if "duration" in job_row else float(ev_time - float(job_row["submit_time"]))
        records.append({
            "job_id": jid,
            "job_type": str(job_row[type_col]),
            "arrival_time": float(job_row["submit_time"]),
            "completion_time": float(ev_time),
            "duration": dur,
            "worker_num": int(job_row["worker_num"]),
            "gpu_request": float(job_row["gpu_request"]),
            "gpu_model": str(job_row.get("gpu_model", "A100-SXM4-80GB")),
            "trace_row_id": int(row_idx),
            "event_id": ev.get("call_id", ev.get("seq", ev_idx)),
        })

    return pd.DataFrame(records)


def build_right_censored_service_history(
    running_jobs: List[Dict[str, Any]],
    preemption_events: List[Dict[str, Any]],
    fit_cutoff: int,
) -> pd.DataFrame:
    """Build one right censored retained service age per job running at cutoff.

    Spot attempt time lost to checkpoint rollback is removed. Recorded restart
    overhead is intentionally excluded from service age and is added back when
    estimating the current job's residual requirement.
    """
    spot_service_before_attempt: Dict[int, int] = {}
    spot_overhead: Dict[int, int] = {}
    for ev in preemption_events:
        if isinstance(ev, dict):
            event_type = ev.get("event_type")
            event_time = ev.get("time")
            job_id = ev.get("job_id")
            details = ev.get("details", {})
        else:
            event_type = getattr(ev, "event_type", None)
            event_time = getattr(ev, "time", None)
            job_id = getattr(ev, "job_id", None)
            details = getattr(ev, "details", {})
        if event_type != "preemption" or event_time is None or int(event_time) >= fit_cutoff or job_id is None:
            continue
        job_id = int(job_id)
        start_time = int(details.get("start_time", event_time))
        rollback = int(round(float(details.get("rollback_runtime_seconds", 0.0))))
        retained = max(0, int(event_time) - start_time - rollback)
        spot_service_before_attempt[job_id] = spot_service_before_attempt.get(job_id, 0) + retained
        spot_overhead[job_id] = spot_overhead.get(job_id, 0) + int(round(float(details.get("overhead", 0.0))))

    rows: List[Dict[str, Any]] = []
    for job in running_jobs:
        if job.get("status") != "run" or int(job.get("submit_time", fit_cutoff)) >= fit_cutoff:
            continue
        job_id = int(job["job_name"])
        job_type = str(job.get("type", job.get("job_type", "")))
        start_time = int(job.get("start_time", fit_cutoff))
        current_attempt_age = max(0, fit_cutoff - start_time)
        retained_age = current_attempt_age
        overhead = 0
        if job_type == "Spot":
            retained_age += spot_service_before_attempt.get(job_id, 0)
            overhead = spot_overhead.get(job_id, 0)
        if retained_age <= 0:
            continue
        rows.append({
            "job_id": job_id,
            "job_type": job_type,
            "gpu_model": str(job.get("gpu_model", job.get("vc_name", "A100-SXM4-80GB"))),
            "worker_num": int(job.get("worker_num", 1)),
            "gpu_request": float(job.get("gpu_request", 1.0)),
            "duration": int(retained_age),
            "service_age_seconds": int(retained_age),
            "additional_work_seconds": int(overhead),
            "censored": True,
            "fit_cutoff": int(fit_cutoff),
        })
    return pd.DataFrame(rows)


@dataclass
class ClassSpecificDurationEstimator:
    """
    Causal, class-specific duration estimator.
    Fits high-priority duration estimates only from completed HP jobs,
    and proposed Spot duration estimates only from completed Spot jobs.
    Uses shape key (gpu_model, worker_num, gpu_request) with class-specific median fallback.
    """
    hp_estimates: Dict[Tuple[str, int, int], int] = field(default_factory=dict)
    hp_fallback: Optional[int] = None
    spot_estimates: Dict[Tuple[str, int, int], int] = field(default_factory=dict)
    spot_fallback: Optional[int] = None
    hp_sample_count: int = 0
    hp_censored_count: int = 0
    spot_sample_count: int = 0
    spot_censored_count: int = 0
    sample_count_per_key: Dict[str, int] = field(default_factory=dict)
    hp_fallback_count: int = 0
    spot_fallback_count: int = 0
    fit_cutoff: Optional[int] = None
    estimator_version: str = "2.0"
    estimator_hash: str = ""
    remaining_samples: Dict[Tuple[Any, ...], List[Tuple[int, bool]]] = field(default_factory=dict)
    remaining_quantile: float = 0.5
    min_remaining_risk_set: int = 5
    _remaining_cache: Dict[Tuple[str, str, int, int, int], Optional[int]] = field(default_factory=dict)
    _remaining_curve_cache: Dict[Tuple[Any, ...], Tuple[List[int], List[int], List[float]]] = field(default_factory=dict)

    def estimate_hp_duration(
        self,
        worker_num: int,
        gpu_request: int,
        gpu_model: str = "A100-SXM4-80GB"
    ) -> int:
        key = (gpu_model, int(worker_num), int(gpu_request))
        if key in self.hp_estimates:
            return self.hp_estimates[key]
        # Secondary fallback across models for exact (worker_num, gpu_request)
        for (m, w, g), val in self.hp_estimates.items():
            if w == worker_num and g == gpu_request:
                return val
        if self.hp_fallback is not None and self.hp_fallback > 0:
            self.hp_fallback_count += 1
            return self.hp_fallback
        raise ValueError(
            f"No defensible HP duration evidence for shape (model={gpu_model}, w={worker_num}, g={gpu_request}) "
            f"before cutoff {self.fit_cutoff}."
        )

    def estimate_spot_duration(
        self,
        worker_num: int,
        gpu_request: int,
        gpu_model: str = "A100-SXM4-80GB"
    ) -> int:
        key = (gpu_model, int(worker_num), int(gpu_request))
        if key in self.spot_estimates:
            return self.spot_estimates[key]
        for (m, w, g), val in self.spot_estimates.items():
            if w == worker_num and g == gpu_request:
                return val
        if self.spot_fallback is not None and self.spot_fallback > 0:
            self.spot_fallback_count += 1
            return self.spot_fallback
        raise ValueError(
            f"No defensible Spot duration evidence for shape (model={gpu_model}, w={worker_num}, g={gpu_request}) "
            f"before cutoff {self.fit_cutoff}."
        )

    def estimate_duration(
        self,
        worker_num: int,
        gpu_request: int,
        is_spot: bool = False,
        gpu_model: str = "A100-SXM4-80GB"
    ) -> int:
        if is_spot:
            return self.estimate_spot_duration(worker_num, gpu_request, gpu_model)
        return self.estimate_hp_duration(worker_num, gpu_request, gpu_model)

    @staticmethod
    def _conditional_remaining_quantile(
        samples: List[Tuple[int, bool]],
        service_age_seconds: int,
        quantile: float,
        min_risk_set: int,
    ) -> Optional[int]:
        curve = ClassSpecificDurationEstimator._build_remaining_curve(samples)
        return ClassSpecificDurationEstimator._query_remaining_curve(
            curve, service_age_seconds, quantile, min_risk_set
        )

    @staticmethod
    def _build_remaining_curve(
        samples: List[Tuple[int, bool]],
    ) -> Tuple[List[int], List[int], List[float]]:
        counts_by_time: Dict[int, List[int]] = {}
        durations = []
        for raw_time, observed in samples:
            duration = max(0, int(raw_time))
            counts = counts_by_time.setdefault(duration, [0, 0])
            counts[0] += 1
            counts[1] += int(bool(observed))
            durations.append(duration)
        risk_count = len(samples)
        survival = 1.0
        event_times: List[int] = []
        survival_after_events: List[float] = []
        for event_time in sorted(counts_by_time):
            total_count, event_count = counts_by_time[event_time]
            if event_count and risk_count > 0:
                survival *= 1.0 - (event_count / risk_count)
                event_times.append(event_time)
                survival_after_events.append(survival)
            risk_count -= total_count
        durations.sort()
        return durations, event_times, survival_after_events

    @staticmethod
    def _query_remaining_curve(
        curve: Tuple[List[int], List[int], List[float]],
        service_age_seconds: int,
        quantile: float,
        min_risk_set: int,
    ) -> Optional[int]:
        from bisect import bisect_right

        durations, event_times, survival_after_events = curve
        age = max(0, int(service_age_seconds))
        at_risk_count = len(durations) - bisect_right(durations, age)
        if at_risk_count < min_risk_set:
            return None
        first_future_event = bisect_right(event_times, age)
        if first_future_event >= len(event_times):
            return None
        survival_at_age = 1.0 if first_future_event == 0 else survival_after_events[first_future_event - 1]
        target_survival = survival_at_age * (1.0 - quantile)

        low = first_future_event
        high = len(event_times)
        while low < high:
            middle = (low + high) // 2
            if survival_after_events[middle] <= target_survival:
                high = middle
            else:
                low = middle + 1
        if low >= len(event_times):
            return None
        return max(1, int(event_times[low] - age))

    def estimate_remaining(
        self,
        worker_num: int,
        gpu_request: int,
        service_age_seconds: int,
        is_spot: bool = False,
        gpu_model: str = "A100-SXM4-80GB",
        additional_work_seconds: int = 0,
    ) -> Optional[int]:
        """Estimate conditional residual work from past completions and censoring.

        Return None when no pool has enough at risk observations or a finite
        conditional quantile. Callers must fail closed for verifier checks.
        """
        job_type = "Spot" if is_spot else "HP"
        worker_num = int(worker_num)
        gpu_request = int(gpu_request)
        age = max(0, int(service_age_seconds))
        cache_key = (job_type, str(gpu_model), worker_num, gpu_request, age)
        if cache_key not in self._remaining_cache:
            pool_keys = (
                (job_type, str(gpu_model), worker_num, gpu_request),
                (job_type, str(gpu_model), None, None),
                (job_type, None, None, None),
            )
            estimate = None
            for key in pool_keys:
                samples = self.remaining_samples.get(key)
                if not samples:
                    continue
                if key not in self._remaining_curve_cache:
                    self._remaining_curve_cache[key] = self._build_remaining_curve(samples)
                estimate = self._query_remaining_curve(
                    self._remaining_curve_cache[key],
                    age,
                    self.remaining_quantile,
                    self.min_remaining_risk_set,
                )
                if estimate is not None:
                    break
            if len(self._remaining_cache) >= 8192:
                self._remaining_cache.clear()
            self._remaining_cache[cache_key] = estimate
        estimate = self._remaining_cache[cache_key]
        if estimate is None:
            return None
        return max(1, int(estimate) + max(0, int(additional_work_seconds)))

    def fit(
        self,
        observed_completions: pd.DataFrame,
        all_jobs_df: pd.DataFrame,
        fit_cutoff: int,
        gpu_model_filter: Optional[str] = None,
        right_censored_service: Optional[pd.DataFrame] = None,
    ) -> None:
        self.fit_cutoff = fit_cutoff
        self.hp_estimates.clear()
        self.spot_estimates.clear()
        self.remaining_samples.clear()
        self._remaining_cache.clear()
        self._remaining_curve_cache.clear()
        self.sample_count_per_key.clear()
        self.hp_fallback_count = 0
        self.spot_fallback_count = 0

        type_col = "job_type" if "job_type" in all_jobs_df.columns else ("type" if "type" in all_jobs_df.columns else "job_type")

        # Split observed completions by class
        hp_completed = observed_completions[observed_completions["job_type"] == "HP"].copy() if not observed_completions.empty else pd.DataFrame()
        spot_completed = observed_completions[observed_completions["job_type"] == "Spot"].copy() if not observed_completions.empty else pd.DataFrame()

        if gpu_model_filter and not hp_completed.empty and "gpu_model" in hp_completed.columns:
            hp_completed = hp_completed[hp_completed["gpu_model"] == gpu_model_filter]
        if gpu_model_filter and not spot_completed.empty and "gpu_model" in spot_completed.columns:
            spot_completed = spot_completed[spot_completed["gpu_model"] == gpu_model_filter]

        censored = right_censored_service.copy() if right_censored_service is not None else pd.DataFrame()
        if not censored.empty:
            required_censored = {"job_type", "gpu_model", "worker_num", "gpu_request", "service_age_seconds"}
            missing = required_censored.difference(censored.columns)
            if missing:
                raise ValueError(f"Right censored service table missing columns: {sorted(missing)}")
            if "fit_cutoff" in censored.columns and (censored["fit_cutoff"] > fit_cutoff).any():
                raise ValueError("Right censored service table contains a cutoff after fit_cutoff.")
            if gpu_model_filter and "gpu_model" in censored.columns:
                censored = censored[censored["gpu_model"] == gpu_model_filter]

        self.hp_sample_count = len(hp_completed)
        self.spot_sample_count = len(spot_completed)

        for job_type, completed in (("HP", hp_completed), ("Spot", spot_completed)):
            for _, row in completed.iterrows():
                request = float(row["gpu_request"])
                if request < 1.0 or not request.is_integer():
                    continue
                self._add_remaining_sample(
                    job_type, str(row["gpu_model"]), int(row["worker_num"]),
                    int(request), int(round(float(row["duration"]))), True
                )
        if not censored.empty:
            for _, row in censored.iterrows():
                job_type = str(row["job_type"])
                if job_type not in ("HP", "Spot"):
                    continue
                request = float(row["gpu_request"])
                if request < 1 or not request.is_integer():
                    continue
                self._add_remaining_sample(
                    job_type, str(row["gpu_model"]), int(row["worker_num"]),
                    int(request), int(row["service_age_seconds"]), False
                )

        # Count censored rows in all_jobs_df arriving before fit_cutoff
        completed_ids = set(observed_completions["job_id"].tolist()) if not observed_completions.empty else set()
        id_col = "job_name" if "job_name" in all_jobs_df.columns else "job_id"
        arrived_before_cutoff = all_jobs_df[all_jobs_df["submit_time"] < fit_cutoff]

        hp_arrived = arrived_before_cutoff[arrived_before_cutoff[type_col] == "HP"]
        spot_arrived = arrived_before_cutoff[arrived_before_cutoff[type_col] == "Spot"]
        if gpu_model_filter and "gpu_model" in all_jobs_df.columns:
            hp_arrived = hp_arrived[hp_arrived["gpu_model"] == gpu_model_filter]
            spot_arrived = spot_arrived[spot_arrived["gpu_model"] == gpu_model_filter]

        self.hp_censored_count = max(0, len(hp_arrived) - sum(1 for jid in hp_arrived[id_col] if jid in completed_ids))
        self.spot_censored_count = max(0, len(spot_arrived) - sum(1 for jid in spot_arrived[id_col] if jid in completed_ids))

        # 1. High Priority Fitting
        if not hp_completed.empty and "duration" in hp_completed.columns:
            hp_grouped = hp_completed.groupby(["gpu_model", "worker_num", "gpu_request"])
            for (m, w, g), group in hp_grouped:
                val = max(1, int(round(group["duration"].median())))
                self.hp_estimates[(str(m), int(w), int(g))] = val
                key_str = f"HP:{m}:{w}:{g}"
                self.sample_count_per_key[key_str] = len(group)
            self.hp_fallback = max(1, int(round(hp_completed["duration"].median())))
        else:
            self.hp_fallback = None

        # 2. Spot Fitting (strictly from completed Spot jobs, never reuse HP median)
        if not spot_completed.empty and "duration" in spot_completed.columns:
            spot_grouped = spot_completed.groupby(["gpu_model", "worker_num", "gpu_request"])
            for (m, w, g), group in spot_grouped:
                val = max(1, int(round(group["duration"].median())))
                self.spot_estimates[(str(m), int(w), int(g))] = val
                key_str = f"Spot:{m}:{w}:{g}"
                self.sample_count_per_key[key_str] = len(group)
            self.spot_fallback = max(1, int(round(spot_completed["duration"].median())))
        else:
            self.spot_fallback = None

        raw_meta = json.dumps({
            "hp_estimates": {f"{k[0]}:{k[1]}:{k[2]}": v for k, v in sorted(self.hp_estimates.items())},
            "hp_fallback": self.hp_fallback,
            "spot_estimates": {f"{k[0]}:{k[1]}:{k[2]}": v for k, v in sorted(self.spot_estimates.items())},
            "spot_fallback": self.spot_fallback,
            "hp_samples": self.hp_sample_count,
            "spot_samples": self.spot_sample_count,
            "remaining_samples": {
                "|".join(str(v) for v in key): values
                for key, values in sorted(self.remaining_samples.items(), key=lambda item: str(item[0]))
            },
            "remaining_quantile": self.remaining_quantile,
            "min_remaining_risk_set": self.min_remaining_risk_set,
            "fit_cutoff": self.fit_cutoff,
            "version": self.estimator_version
        }, sort_keys=True)
        self.estimator_hash = hashlib.sha256(raw_meta.encode("utf-8")).hexdigest()

    def _add_remaining_sample(
        self,
        job_type: str,
        gpu_model: str,
        worker_num: int,
        gpu_request: int,
        duration_seconds: int,
        observed: bool,
    ) -> None:
        sample = (max(0, int(duration_seconds)), bool(observed))
        keys = (
            (job_type, gpu_model, int(worker_num), int(gpu_request)),
            (job_type, gpu_model, None, None),
            (job_type, None, None, None),
        )
        for key in keys:
            self.remaining_samples.setdefault(key, []).append(sample)


class PastFittedScenarioGenerator:
    """
    Fits empirical arrival blocks and duration distributions from past training data.
    Unconditional historical block bootstrap: samples complete past HP blocks of width H_s.
    Separates HP arrival block fitting from class-specific duration fitting.
    Never reads future trace arrivals beyond the decision time t_k.
    """
    def __init__(
        self,
        horizon_seconds: int = 1800,
        L_H: int = 3600,
        model_version: str = "2.0",
        random_seed: int = 42,
        hp_arrival_cohort: str = "gang_only",
    ):
        if hp_arrival_cohort not in {"gang_only", "whole_card_all"}:
            raise ValueError("hp_arrival_cohort must be gang_only or whole_card_all.")
        self.horizon_seconds = horizon_seconds
        self.L_H = L_H
        self.model_version = model_version
        self.random_seed = random_seed
        self.hp_arrival_cohort = hp_arrival_cohort
        self.is_fitted = False
        self.fit_range: Optional[Tuple[int, int]] = None
        self.fit_cutoff: Optional[int] = None
        self.historical_blocks: List[List[Dict[str, Any]]] = []
        self.duration_estimator: Optional[ClassSpecificDurationEstimator] = None
        self.duration_estimates: Dict[Tuple[int, int], int] = {}
        self.default_duration: int = 600
        self.sample_count: int = 0
        self.censored_count: int = 0
        self.fit_hash: str = ""
        self.estimator_version: str = "3.0_conditional_survival"
        self.fit_report: Dict[str, Any] = {}

    def fit(
        self,
        jobs_df: pd.DataFrame,
        fit_start: int,
        fit_end: int,
        gpu_model_filter: Optional[str] = None,
        completion_events: Optional[List[Dict[str, Any]]] = None,
        observed_completions: Optional[pd.DataFrame] = None,
        require_observed_completions: bool = False,
        right_censored_service: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Fits generator strictly on past training data [fit_start, fit_end).
        Requires:
        - HP jobs only for arrival blocks
        - worker_num > 1 for gang_only, or >= 1 for whole_card_all
        - integer gpu_request >= 1.0
        - submit_time in [fit_start, fit_end)
        - complete historical blocks of width horizon_seconds (empty blocks retained)
        - duration fitting strictly from observed completions strictly before fit_end
        """
        if fit_start < 0 or fit_start >= fit_end:
            raise ValueError(f"Invalid fit interval: fit_start ({fit_start}) must be >= 0 and < fit_end ({fit_end}).")

        # Clear all prior fit state before refitting
        self.is_fitted = False
        self.historical_blocks = []
        self.duration_estimates = {}
        self.duration_estimator = None
        self.fit_report = {}
        self.sample_count = 0
        self.censored_count = 0
        self.fit_hash = ""

        self.fit_range = (fit_start, fit_end)
        self.fit_cutoff = fit_end

        type_col = "job_type" if "job_type" in jobs_df.columns else ("type" if "type" in jobs_df.columns else None)
        if type_col is None:
            raise ValueError("Input jobs_df missing required job type column ('job_type' or 'type').")

        type_mask = (jobs_df[type_col] == "HP")

        # Filter: HP only, integer gpu_request >= 1.0, submit_time in [fit_start, fit_end)
        base_mask = (
            type_mask &
            (jobs_df["gpu_request"] >= 1.0) &
            (jobs_df["gpu_request"] % 1.0 == 0.0) &
            (jobs_df["submit_time"] >= fit_start) &
            (jobs_df["submit_time"] < fit_end)
        )
        if gpu_model_filter and "gpu_model" in jobs_df.columns:
            base_mask = base_mask & (jobs_df["gpu_model"] == gpu_model_filter)

        # Preserve the original gang only protocol unless the revised,
        # explicitly versioned cohort includes all whole card HP arrivals.
        gang_mask = base_mask & (jobs_df["worker_num"] > 1)
        gang_df = jobs_df[gang_mask].copy()
        arrival_mask = (gang_mask if self.hp_arrival_cohort == "gang_only"
                        else base_mask & (jobs_df["worker_num"] >= 1))
        arrival_df = jobs_df[arrival_mask].copy()

        assert (gang_df[type_col] == "Spot").sum() == 0, "Filtered HP gang cohort must have 0 Spot jobs!"
        if not gang_df.empty:
            assert (gang_df[type_col] == "HP").all(), "Filtered HP gang cohort must contain HP jobs only!"
            assert (gang_df["worker_num"] > 1).all(), "Filtered HP gang cohort must have worker_num > 1!"
            assert ((gang_df["gpu_request"] >= 1.0) & (gang_df["gpu_request"] % 1.0 == 0.0)).all(), "Filtered HP gang cohort must have integer gpu_request >= 1.0!"

        # Stable sort by submit_time, then job_name / index
        sort_cols = ["submit_time"]
        if "job_name" in arrival_df.columns:
            sort_cols.append("job_name")
        arrival_df = arrival_df.sort_values(by=sort_cols)

        # Partition into complete chronological blocks of width horizon_seconds (empty blocks retained)
        blocks: List[List[Dict[str, Any]]] = []
        cur_t = fit_start
        while cur_t + self.horizon_seconds <= fit_end:
            nxt_t = cur_t + self.horizon_seconds
            block_jobs = arrival_df[(arrival_df["submit_time"] >= cur_t) & (arrival_df["submit_time"] < nxt_t)]
            job_dicts = []
            for _, row in block_jobs.iterrows():
                job_dicts.append({
                    "job_name": int(row.get("job_name", row.name)),
                    "offset": int(row["submit_time"] - cur_t),
                    "worker_num": int(row["worker_num"]),
                    "gpu_request": int(row["gpu_request"]),
                    "gpu_model": str(row.get("gpu_model", "A100-SXM4-80GB")),
                })
            blocks.append(job_dicts)
            cur_t = nxt_t

        self.historical_blocks = blocks

        # -------------------------------------------------------------
        # Duration Fitting: Strictly from observed completions before fit_cutoff
        # -------------------------------------------------------------
        if observed_completions is not None:
            comp_df = observed_completions
        elif completion_events is not None:
            comp_df = build_observed_completion_history(completion_events, jobs_df, fit_cutoff=fit_end)
        elif "observed_completion_time" in jobs_df.columns:
            comp_df = jobs_df[jobs_df["observed_completion_time"] < fit_end].copy()
            if "job_id" not in comp_df.columns and "job_name" in comp_df.columns:
                comp_df["job_id"] = comp_df["job_name"]
        else:
            if require_observed_completions:
                raise ValueError(
                    "Causal duration fit requires executed completion events or an observed completion table. "
                    "Final trace duration cannot prove completion before the cutoff."
                )
            # Compatibility path for isolated synthetic unit tests only. Production callers must set
            # require_observed_completions=True and provide executed completion evidence.
            if "duration" in jobs_df.columns:
                mask_completed = (jobs_df["submit_time"] + jobs_df["duration"]) <= fit_end
                comp_df = jobs_df[mask_completed].copy()
                if "job_id" not in comp_df.columns and "job_name" in comp_df.columns:
                    comp_df["job_id"] = comp_df["job_name"]
            else:
                comp_df = pd.DataFrame()

        if not comp_df.empty:
            required_cols = {"job_id", "job_type", "duration"}
            missing = required_cols.difference(comp_df.columns)
            if missing:
                raise ValueError(f"Observed completion table missing columns: {sorted(missing)}")
            if require_observed_completions and "completion_time" not in comp_df.columns:
                raise ValueError(
                    "Causal observed completion rows must include completion_time so the fit cutoff can be checked."
                )
            if "completion_time" in comp_df.columns:
                bad = comp_df[comp_df["completion_time"] >= fit_end]
                if not bad.empty:
                    raise ValueError(
                        f"Observed completion table contains {len(bad)} completion(s) at or after cutoff {fit_end}."
                    )

        self.duration_estimator = ClassSpecificDurationEstimator(fit_cutoff=fit_end, estimator_version=self.estimator_version)
        self.duration_estimator.fit(
            observed_completions=comp_df,
            all_jobs_df=jobs_df,
            fit_cutoff=fit_end,
            gpu_model_filter=gpu_model_filter,
            right_censored_service=right_censored_service,
        )

        self.sample_count = self.duration_estimator.hp_sample_count
        self.censored_count = self.duration_estimator.hp_censored_count
        self.default_duration = self.duration_estimator.hp_fallback or 600

        # Maintain legacy duration_estimates dictionary for (worker_num, gpu_request)
        for (m, w, g), val in self.duration_estimator.hp_estimates.items():
            if (w, g) not in self.duration_estimates:
                self.duration_estimates[(w, g)] = val

        raw_fit_str = json.dumps({
            "fit_range": (fit_start, fit_end),
            "retained_blocks": len(blocks),
            "historical_blocks": blocks,
            "retained_hp_gang_rows": len(gang_df),
            "retained_hp_arrival_rows": len(arrival_df),
            "hp_arrival_cohort": self.hp_arrival_cohort,
            "sample_count": self.sample_count,
            "censored_count": self.censored_count,
            "dur_hash": self.duration_estimator.estimator_hash,
            "horizon_seconds": self.horizon_seconds
        }, sort_keys=True)
        self.fit_hash = hashlib.sha256(raw_fit_str.encode("utf-8")).hexdigest()

        self.fit_report = {
            "total_input_rows": int(len(jobs_df)),
            "input_hp_rows": int((jobs_df[type_col] == "HP").sum()),
            "input_spot_rows": int((jobs_df[type_col] == "Spot").sum()),
            "input_by_type": {str(k): int(v) for k, v in jobs_df[type_col].value_counts().items()},
            "input_by_model": {str(k): int(v) for k, v in jobs_df["gpu_model"].value_counts().items()} if "gpu_model" in jobs_df.columns else {},
            "retained_hp_gang_rows": int(len(gang_df)),
            "retained_hp_arrival_rows": int(len(arrival_df)),
            "retained_hp_single_worker_rows": int((arrival_df["worker_num"] == 1).sum()),
            "hp_arrival_cohort": self.hp_arrival_cohort,
            "retained_by_model": {str(k): int(v) for k, v in gang_df["gpu_model"].value_counts().items()} if "gpu_model" in gang_df.columns else {},
            "retained_blocks": int(len(blocks)),
            "fit_range": (fit_start, fit_end),
            "fit_cutoff": fit_end,
            "gpu_model_filter": gpu_model_filter,
            "sample_count": int(self.sample_count),
            "censored_count": int(self.censored_count),
            "spot_sample_count": int(self.duration_estimator.spot_sample_count),
            "spot_censored_count": int(self.duration_estimator.spot_censored_count),
            "right_censored_service_count": int(len(right_censored_service)) if right_censored_service is not None else 0,
            "remaining_time_quantile": float(self.duration_estimator.remaining_quantile),
            "remaining_time_min_risk_set": int(self.duration_estimator.min_remaining_risk_set),
            "fit_hash": self.fit_hash,
            "estimator_hash": self.duration_estimator.estimator_hash
        }

        self.is_fitted = True

    def estimate_hp_duration(self, worker_num: int, gpu_request: int, gpu_model: str = "A100-SXM4-80GB") -> int:
        if self.duration_estimator is not None:
            return self.duration_estimator.estimate_hp_duration(worker_num, gpu_request, gpu_model)
        return self.duration_estimates.get((worker_num, gpu_request), self.default_duration)

    def estimate_spot_duration(self, worker_num: int, gpu_request: int, gpu_model: str = "A100-SXM4-80GB") -> int:
        if self.duration_estimator is not None:
            return self.duration_estimator.estimate_spot_duration(worker_num, gpu_request, gpu_model)
        raise ValueError("Duration estimator not fitted for Spot jobs.")

    def estimate_remaining(
        self,
        worker_num: int,
        gpu_request: int,
        service_age_seconds: int,
        is_spot: bool = False,
        gpu_model: str = "A100-SXM4-80GB",
        additional_work_seconds: int = 0,
    ) -> Optional[int]:
        if self.duration_estimator is None:
            return None
        return self.duration_estimator.estimate_remaining(
            worker_num=worker_num,
            gpu_request=gpu_request,
            service_age_seconds=service_age_seconds,
            is_spot=is_spot,
            gpu_model=gpu_model,
            additional_work_seconds=additional_work_seconds,
        )

    def estimate_duration(
        self,
        worker_num: int,
        gpu_request: int,
        is_spot: bool = False,
        gpu_model: str = "A100-SXM4-80GB"
    ) -> int:
        if is_spot:
            return self.estimate_spot_duration(worker_num, gpu_request, gpu_model)
        return self.estimate_hp_duration(worker_num, gpu_request, gpu_model)

    def sample_scenarios(
        self,
        snapshot: ClusterStateSnapshot,
        K: int,
        seed: Optional[int] = None,
        offered_job_id: Optional[int] = None,
        offer_ordinal: int = 0,
        split_id: str = "train"
    ) -> List[Scenario]:
        """
        Samples K common future scenarios conditional on decision snapshot.
        Includes all currently queued whole-card HP jobs.
        Generates each scenario deterministically from stable policy-independent seeds.
        """
        if not self.is_fitted:
            raise RuntimeError("PastFittedScenarioGenerator must be fitted before sampling.")
        if not self.historical_blocks:
            raise ValueError("Historical block population is empty. Cannot sample future scenarios.")

        t_k = snapshot.time
        n_blocks = len(self.historical_blocks)
        base_seed = seed if seed is not None else self.random_seed
        job_id_val = int(offered_job_id) if offered_job_id is not None else 0

        # Convert snapshot queued HP jobs to ScenarioJob (marked is_queued=True)
        queued_scenario_jobs = [
            ScenarioJob(
                job_id=qj.job_id,
                submit_time=qj.submit_time,
                worker_num=qj.worker_num,
                gpu_request=qj.gpu_request,
                gpu_model=qj.gpu_model,
                duration=self.estimate_hp_duration(qj.worker_num, qj.gpu_request, qj.gpu_model),
                is_queued=True
            )
            for qj in snapshot.queued_hp_jobs
        ]

        scenarios: List[Scenario] = []

        for k in range(K):
            k_seed = derive_scenario_seed(
                shared_scenario_seed=base_seed,
                split_id=split_id,
                replay_second=t_k,
                offered_job_id=job_id_val,
                offer_ordinal=offer_ordinal,
                scenario_index=k
            )
            k_rng = np.random.default_rng(k_seed)

            # Sample a historical block via block bootstrap
            sampled_jobs: List[ScenarioJob] = list(queued_scenario_jobs)
            block_idx = int(k_rng.integers(0, n_blocks))
            block = self.historical_blocks[block_idx]
            for idx, bj in enumerate(block):
                arr_time = t_k + bj["offset"]
                dur = self.estimate_hp_duration(bj["worker_num"], bj["gpu_request"], bj["gpu_model"])
                sampled_jobs.append(
                    ScenarioJob(
                        job_id=-(k * 100000 + idx + 1),
                        submit_time=arr_time,
                        worker_num=bj["worker_num"],
                        gpu_request=bj["gpu_request"],
                        gpu_model=bj["gpu_model"],
                        duration=dur,
                        is_queued=False
                    )
                )

            # Assert zero Spot identifiers and integer whole-card constraints in sampled jobs
            for j in sampled_jobs:
                if not j.is_queued:
                    assert j.job_id < 0, f"Sampled scenario job has non-negative ID: {j.job_id}"
                    assert j.worker_num >= (2 if self.hp_arrival_cohort == "gang_only" else 1), (
                        f"Sampled arrival job violates cohort {self.hp_arrival_cohort}: {j.worker_num}"
                    )
                assert j.gpu_request >= 1.0 and (j.gpu_request % 1.0 == 0.0), f"Sampled scenario job must have integer gpu_request >= 1.0, got {j.gpu_request}"
                assert getattr(j, "is_spot", False) is False, "Sampled scenario job must not be Spot!"

            # Sort jobs stably by submit_time, then job_id
            sampled_jobs.sort(key=lambda j: (j.submit_time, j.job_id))

            provenance = {
                "fit_range": self.fit_range,
                "fit_cutoff": self.fit_cutoff,
                "fit_hash": self.fit_hash,
                "duration_model_hash": self.duration_estimator.estimator_hash if self.duration_estimator else None,
                "sample_count": self.sample_count,
                "censored_count": self.censored_count,
                "model_version": self.model_version,
                "hp_arrival_cohort": self.hp_arrival_cohort,
                "estimator_version": self.estimator_version,
                "sample_seed": k_seed,
                "sampled_block_index": block_idx,
                "relative_arrivals": [bj["offset"] for bj in block],
                "queued_job_ids": [qj.job_id for qj in snapshot.queued_hp_jobs],
                "t_k": t_k,
                "K": K,
                "k_index": k
            }

            scenarios.append(
                Scenario(
                    scenario_id=k,
                    seed=k_seed,
                    start_time=t_k,
                    horizon_seconds=self.horizon_seconds,
                    jobs=tuple(sampled_jobs),
                    provenance=provenance
                )
            )

        return scenarios
