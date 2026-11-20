"""
metrics.py: Event ledger, credit tracking, HP gang deadline verification, and episode metrics.
Reconciles transition rewards with episode totals, enforces [score_start, score_end) boundaries,
and treats zero-denominator cohorts honestly.
"""

import math
import heapq
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import numpy as np

from .types import SimulationEvent, EpisodeSpec
from .measurement import (
    SCHEMA_VERSION,
    PHASE_WARMUP,
    PHASE_SCORE,
    PHASE_FOLLOWUP,
    get_phase,
    count_intersecting_quota_windows,
)


def sanitize_primitive(obj: Any) -> Any:
    """Recursively converts objects to strictly JSON-serializable primitives and rejects non-finite floats."""
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"Non-finite float {obj} encountered in event details")
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if not math.isfinite(val):
            raise ValueError(f"Non-finite numpy float {val} encountered in event details")
        return val
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_primitive(v) for v in obj]
    if isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            if isinstance(k, (int, np.integer)):
                clean_k = int(k)
            elif isinstance(k, str):
                clean_k = k
            else:
                clean_k = str(k)
            res[clean_k] = sanitize_primitive(v)
        return res
    if hasattr(obj, "item"):
        return sanitize_primitive(obj.item())
    return str(obj)


@dataclass
class EpisodeMetrics:
    schema_version: str = SCHEMA_VERSION
    run_id: str = ""
    episode_id: str = ""
    status: str = "completed"
    terminal_reason: str = ""
    useful_spot_gpu_seconds_score: float = 0.0
    useful_spot_gpu_hours_score_raw: float = 0.0
    initial_reward_gpu_hours: float = 0.0
    step_reward_gpu_hours_sum: float = 0.0
    hp_gang_cohort_count: int = 0
    hp_gang_violation_count: int = 0
    hp_gang_violation_rate: Optional[float] = None
    hp_gang_censored_count: int = 0
    spot_eviction_events_score: int = 0
    unique_spot_jobs_evicted_score: int = 0
    spot_lost_gpu_seconds_score: float = 0.0
    solver_calls_score: int = 0
    quota_windows_score: int = 0
    calls_per_window: Optional[float] = None
    decision_count: int = 0

    # All-phase totals
    useful_spot_gpu_seconds_all_phase: float = 0.0
    useful_spot_gpu_hours_all_phase_raw: float = 0.0
    spot_eviction_events_all_phase: int = 0
    unique_spot_jobs_evicted_all_phase: int = 0
    spot_lost_gpu_seconds_all_phase: float = 0.0
    solver_calls_all_phase: int = 0
    completed_hp_jobs: int = 0
    completed_spot_jobs: int = 0
    remaining_queued_jobs: int = 0
    remaining_running_jobs: int = 0

    # Display fields
    useful_spot_gpu_hours_score_display: float = 0.0
    hp_gang_violation_rate_display: Optional[float] = None
    calls_per_window_display: Optional[float] = None
    eviction_events_per_1000_spot_starts_display: Optional[float] = None

    # Bounds and rules metadata
    rule_version: str = "1.0"
    score_start: int = 0
    score_end: int = 0
    followup_end: int = 0
    terminated: bool = False
    truncated: bool = False

    # Compatibility properties for prior tests and call sites
    @property
    def useful_spot_gpu_seconds(self) -> float:
        return self.useful_spot_gpu_seconds_score

    @property
    def useful_spot_gpu_hours(self) -> float:
        return self.useful_spot_gpu_hours_score_display

    @property
    def hp_gang_arrivals_cohort(self) -> int:
        return self.hp_gang_cohort_count

    @property
    def hp_gang_violations(self) -> int:
        return self.hp_gang_violation_count

    @property
    def total_solver_calls(self) -> int:
        return self.solver_calls_score

    @property
    def calls_per_300s_window(self) -> float:
        return self.calls_per_window_display or 0.0

    @property
    def num_quota_windows(self) -> int:
        return self.quota_windows_score

    @property
    def spot_evictions_count(self) -> int:
        return self.spot_eviction_events_score

    @property
    def spot_lost_gpu_seconds(self) -> float:
        return self.spot_lost_gpu_seconds_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "status": self.status,
            "terminal_reason": self.terminal_reason,
            "useful_spot_gpu_seconds_score": self.useful_spot_gpu_seconds_score,
            "useful_spot_gpu_hours_score_raw": self.useful_spot_gpu_hours_score_raw,
            "initial_reward_gpu_hours": self.initial_reward_gpu_hours,
            "step_reward_gpu_hours_sum": self.step_reward_gpu_hours_sum,
            "hp_gang_cohort_count": self.hp_gang_cohort_count,
            "hp_gang_violation_count": self.hp_gang_violation_count,
            "hp_gang_violation_rate": self.hp_gang_violation_rate,
            "hp_gang_censored_count": self.hp_gang_censored_count,
            "spot_eviction_events_score": self.spot_eviction_events_score,
            "unique_spot_jobs_evicted_score": self.unique_spot_jobs_evicted_score,
            "spot_lost_gpu_seconds_score": self.spot_lost_gpu_seconds_score,
            "solver_calls_score": self.solver_calls_score,
            "quota_windows_score": self.quota_windows_score,
            "calls_per_window": self.calls_per_window,
            "decision_count": self.decision_count,
            "useful_spot_gpu_seconds_all_phase": self.useful_spot_gpu_seconds_all_phase,
            "useful_spot_gpu_hours_all_phase_raw": self.useful_spot_gpu_hours_all_phase_raw,
            "spot_eviction_events_all_phase": self.spot_eviction_events_all_phase,
            "unique_spot_jobs_evicted_all_phase": self.unique_spot_jobs_evicted_all_phase,
            "spot_lost_gpu_seconds_all_phase": self.spot_lost_gpu_seconds_all_phase,
            "solver_calls_all_phase": self.solver_calls_all_phase,
            "completed_hp_jobs": self.completed_hp_jobs,
            "completed_spot_jobs": self.completed_spot_jobs,
            "remaining_queued_jobs": self.remaining_queued_jobs,
            "remaining_running_jobs": self.remaining_running_jobs,
            "useful_spot_gpu_hours_score_display": self.useful_spot_gpu_hours_score_display,
            "hp_gang_violation_rate_display": self.hp_gang_violation_rate_display,
            "calls_per_window_display": self.calls_per_window_display,
            "eviction_events_per_1000_spot_starts_display": self.eviction_events_per_1000_spot_starts_display,
            "rule_version": self.rule_version,
            "score_start": self.score_start,
            "score_end": self.score_end,
            "followup_end": self.followup_end,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }


class EventLedger:
    """Ordered event ledger recording all simulator state transitions."""
    def __init__(self, episode_id: str, spec: EpisodeSpec, run_id: str = ""):
        self.episode_id = episode_id
        self.run_id = run_id
        self.spec = spec
        self.events: List[SimulationEvent] = []
        self._seq = 0

        # Scored interval accumulators
        self.scored_useful_spot_gpu_seconds: float = 0.0
        self.scored_spot_lost_gpu_seconds: float = 0.0
        self.scored_spot_evictions: int = 0
        self.scored_solver_calls: int = 0
        self.unique_spot_jobs_evicted_score: Set[Any] = set()
        self.decision_count: int = 0
        self.scored_spot_starts: int = 0

        # Cumulative / legacy accumulators (preserved across snapshot restore)
        self.useful_spot_gpu_seconds: float = 0.0
        self.spot_lost_gpu_seconds: float = 0.0
        self.spot_evictions: int = 0
        self.total_solver_calls: int = 0

        # All phase accumulators
        self.useful_spot_gpu_seconds_all_phase: float = 0.0
        self.useful_spot_gpu_seconds_source: float = 0.0
        self.spot_lost_gpu_seconds_all_phase: float = 0.0
        self.spot_evictions_all_phase: int = 0
        self.unique_spot_jobs_evicted_all_phase: Set[Any] = set()
        self.solver_calls_all_phase: int = 0
        self.all_spot_starts: int = 0
        self.solver_calls_by_window: Dict[int, int] = {}

        # HP gang wait tracking: job_id -> dict
        self.hp_gang_jobs: Dict[Any, Dict[str, Any]] = {}
        # Min-heap ordered by deadline: (deadline, arrival_time, job_id)
        self.hp_deadline_queue: List[Tuple[int, int, Any]] = []
        self.hp_gang_violations: int = 0
        self.hp_gang_met: int = 0
        self.hp_gang_censored: int = 0
        self.uncontrollable_prefix_violations: int = 0

        # Job completion counts
        self.completed_jobs: Dict[str, int] = {"HP": 0, "Spot": 0}

        # Unique credit keys: (job_id, attempt_index, credit_sequence)
        self.seen_credit_keys: Set[Tuple[Any, int, int]] = set()

    def record_event(
        self,
        time: int,
        event_type: str,
        job_id: Optional[Any] = None,
        gpu_count: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
        vc_name: Optional[str] = None,
        attempt_index: Optional[int] = None,
        phase: Optional[str] = None,
    ) -> SimulationEvent:
        if phase is None:
            phase = get_phase(time, self.spec.score_start, self.spec.score_end)

        sanitized_details = sanitize_primitive(details or {})

        event = SimulationEvent(
            schema_version=SCHEMA_VERSION,
            run_id=self.run_id,
            episode_id=self.episode_id,
            event_seq=self._seq,
            time=time,
            phase=phase,
            event_type=event_type,
            job_id=job_id,
            vc_name=vc_name,
            attempt_index=attempt_index,
            gpu_count=gpu_count,
            details=sanitized_details
        )
        if event_type == "action" and phase == PHASE_SCORE:
            self.decision_count += 1
        self.events.append(event)
        self._seq += 1
        return event

    def record_spot_credit(
        self,
        time: int,
        job_id: Any,
        gpu_seconds: float,
        reason: str,
        attempt_index: int = 0,
        credit_sequence: int = 0,
        credited_run_seconds: float = 0.0,
        gpu_equivalents: float = 1.0,
        vc_name: Optional[str] = None,
        job_credit_cap_gpu_seconds: Optional[float] = None,
        cumulative_credited_gpu_seconds: Optional[float] = None,
    ) -> float:
        """Credits useful Spot GPU seconds. Enters primary reward only if in [score_start, score_end)."""
        in_score_window_research = (self.spec.score_start <= time < self.spec.score_end)
        in_score_window_source = (self.spec.score_start < time <= self.spec.score_end)

        c_key = (job_id, attempt_index, credit_sequence)
        if c_key in self.seen_credit_keys:
            raise ValueError(f"Duplicate credit key {c_key} emitted at time {time}")
        self.seen_credit_keys.add(c_key)

        if in_score_window_research:
            self.useful_spot_gpu_seconds += gpu_seconds
            self.scored_useful_spot_gpu_seconds += gpu_seconds
        if in_score_window_source:
            self.useful_spot_gpu_seconds_source += gpu_seconds
        self.useful_spot_gpu_seconds_all_phase += gpu_seconds

        self.record_event(
            time=time,
            event_type="credit",
            job_id=job_id,
            gpu_count=gpu_seconds / 3600.0,
            vc_name=vc_name,
            attempt_index=attempt_index,
            details={
                "reason": reason,
                "attempt_index": attempt_index,
                "credit_sequence": credit_sequence,
                "credited_run_seconds": credited_run_seconds,
                "gpu_equivalents": gpu_equivalents,
                "credited_gpu_seconds": gpu_seconds,
                "job_credit_cap_gpu_seconds": job_credit_cap_gpu_seconds,
                "cumulative_credited_gpu_seconds": cumulative_credited_gpu_seconds,
                "in_score_window_research": in_score_window_research,
                "in_score_window_source": in_score_window_source,
            }
        )
        return gpu_seconds if in_score_window_research else 0.0

    def record_preemption(
        self,
        time: int,
        job_id: Any,
        lost_seconds: float,
        attempt_index: int = 0,
        start_time: int = 0,
        worker_num: int = 1,
        gpu_request: float = 1.0,
        rollback_runtime_seconds: float = 0.0,
        remain_before: float = 0.0,
        remain_after: float = 0.0,
        overhead: float = 0.0,
        preempting_hp_job_id: Optional[Any] = None,
        vc_name: Optional[str] = None
    ):
        """Records a Spot job preemption and lost GPU work."""
        in_score_window = (self.spec.score_start <= time < self.spec.score_end)
        self.spot_evictions_all_phase += 1
        self.spot_lost_gpu_seconds_all_phase += lost_seconds
        self.unique_spot_jobs_evicted_all_phase.add(job_id)

        if in_score_window:
            self.spot_evictions += 1
            self.scored_spot_evictions += 1
            self.spot_lost_gpu_seconds += lost_seconds
            self.scored_spot_lost_gpu_seconds += lost_seconds
            self.unique_spot_jobs_evicted_score.add(job_id)

        self.record_event(
            time=time,
            event_type="preemption",
            job_id=job_id,
            gpu_count=lost_seconds / 3600.0,
            vc_name=vc_name,
            attempt_index=attempt_index,
            details={
                "attempt_index": attempt_index,
                "start_time": start_time,
                "worker_num": worker_num,
                "gpu_request": gpu_request,
                "rollback_runtime_seconds": rollback_runtime_seconds,
                "lost_gpu_seconds": lost_seconds,
                "remain_before": remain_before,
                "remain_after": remain_after,
                "overhead": overhead,
                "preempting_hp_job_id": preempting_hp_job_id,
                "in_score_window": in_score_window,
            }
        )

    def record_hp_arrival(
        self,
        time: int,
        job_id: Any,
        worker_num: int,
        gpu_request: float,
        submit_time: Optional[int] = None,
        vc_name: Optional[str] = None
    ):
        """Tracks HP gang jobs for deadline compliance and emits arrival event."""
        sub_time = submit_time if submit_time is not None else time
        is_whole_card_gang = (worker_num > 1) and (gpu_request >= 1.0) and (float(gpu_request).is_integer())
        in_score_window = (self.spec.score_start <= time < self.spec.score_end)
        has_followup = (time + self.spec.L_H <= self.spec.followup_end)

        eligible = is_whole_card_gang and in_score_window and has_followup
        if not is_whole_card_gang:
            exclusion_reason = "single_worker" if worker_num <= 1 else "fractional_gpu"
        elif not in_score_window:
            exclusion_reason = "arrival_before_score_start" if time < self.spec.score_start else "arrival_at_or_after_score_end"
        elif not has_followup:
            exclusion_reason = "deadline_exceeds_followup_end"
        else:
            exclusion_reason = None

        deadline = time + self.spec.L_H
        self.hp_gang_jobs[job_id] = {
            "job_id": job_id,
            "arrival_time": time,
            "start_time": None,
            "worker_num": worker_num,
            "gpu_request": gpu_request,
            "eligible": eligible,
            "deadline": deadline,
            "violated": False,
            "violation_emitted": False,
            "outcome_emitted": False,
            "vc_name": vc_name,
        }
        if eligible:
            heapq.heappush(self.hp_deadline_queue, (deadline, time, job_id))

        self.record_event(
            time=time,
            event_type="arrival",
            job_id=job_id,
            vc_name=vc_name,
            details={
                "job_type": "HP",
                "worker_num": worker_num,
                "gpu_request": gpu_request,
                "submit_time": sub_time,
                "executed_arrival_time": time,
                "vc_name": vc_name or "",
                "hp_cohort_eligible": eligible,
                "exclusion_reason": exclusion_reason,
            }
        )

    def record_job_start(
        self,
        time: int,
        job_id: Any,
        job_type: str,
        nodes: Any = None,
        requested_worker_count: Optional[int] = None,
        actual_worker_count: Optional[int] = None,
        complete_start: bool = True,
        allocations: Optional[List[Any]] = None,
        vc_name: Optional[str] = None,
        attempt_index: Optional[int] = None,
    ):
        """Records job start and updates wait time and outcome for HP jobs."""
        if job_type == "Spot":
            self.all_spot_starts += 1
            if self.spec.score_start <= time < self.spec.score_end:
                self.scored_spot_starts += 1

        req_w = requested_worker_count or 1
        act_w = actual_worker_count or (len(nodes) if isinstance(nodes, list) else 1)
        allocs = allocations if allocations is not None else (nodes if isinstance(nodes, list) else [])

        if job_id in self.hp_gang_jobs and complete_start:
            info = self.hp_gang_jobs[job_id]
            if info["start_time"] is None:
                info["start_time"] = time
                if info["eligible"]:
                    if time <= info["deadline"]:
                        # Met
                        if not info["outcome_emitted"]:
                            info["outcome_emitted"] = True
                            self.hp_gang_met += 1
                            self.record_event(
                                time=time,
                                event_type="deadline_outcome",
                                job_id=job_id,
                                vc_name=vc_name or info.get("vc_name"),
                                details={
                                    "arrival_time": info["arrival_time"],
                                    "deadline": info["deadline"],
                                    "start_time": time,
                                    "outcome": "met",
                                    "observed_time": time,
                                    "rule_version": self.spec.rule_version,
                                }
                            )
                    else:
                        # Missed (started late)
                        if not info["outcome_emitted"]:
                            info["outcome_emitted"] = True
                            self.hp_gang_violations += 1
                            info["violated"] = True
                            self.record_event(
                                time=time,
                                event_type="deadline_outcome",
                                job_id=job_id,
                                vc_name=vc_name or info.get("vc_name"),
                                details={
                                    "arrival_time": info["arrival_time"],
                                    "deadline": info["deadline"],
                                    "start_time": time,
                                    "outcome": "missed",
                                    "observed_time": time,
                                    "rule_version": self.spec.rule_version,
                                }
                            )
                        if not info["violation_emitted"]:
                            info["violation_emitted"] = True
                            self.record_event(
                                time=time,
                                event_type="hp_deadline",
                                job_id=job_id,
                                vc_name=vc_name or info.get("vc_name"),
                                details={
                                    "arrival_time": info["arrival_time"],
                                    "deadline": info["deadline"],
                                    "start_time": time,
                                    "status": "started_past_deadline",
                                    "wait_seconds": time - info["arrival_time"],
                                }
                            )

        self.record_event(
            time=time,
            event_type="start",
            job_id=job_id,
            vc_name=vc_name,
            attempt_index=attempt_index,
            details={
                "job_type": job_type,
                "requested_worker_count": req_w,
                "actual_worker_count": act_w,
                "complete_start": complete_start,
                "allocations": allocs,
                "nodes": allocs,
            }
        )

    def check_hp_deadlines(self, current_time: int) -> List[SimulationEvent]:
        """
        Emits hp_deadline and deadline_outcome events when a violation first becomes observable.
        After HP scheduling at simulator second t, any eligible job not started by
        t > arrival + L_H becomes violated.
        """
        emitted: List[SimulationEvent] = []
        while self.hp_deadline_queue and self.hp_deadline_queue[0][0] < current_time:
            deadline, arr_time, job_id = heapq.heappop(self.hp_deadline_queue)
            info = self.hp_gang_jobs.get(job_id)
            if info is None or not info["eligible"]:
                continue
            if info["start_time"] is None:
                if not info["outcome_emitted"]:
                    info["outcome_emitted"] = True
                    info["violated"] = True
                    self.hp_gang_violations += 1
                    self.record_event(
                        time=current_time,
                        event_type="deadline_outcome",
                        job_id=job_id,
                        vc_name=info.get("vc_name"),
                        details={
                            "arrival_time": info["arrival_time"],
                            "deadline": deadline,
                            "start_time": None,
                            "outcome": "missed",
                            "observed_time": current_time,
                            "rule_version": self.spec.rule_version,
                        }
                    )
                if not info["violation_emitted"]:
                    info["violation_emitted"] = True
                    ev = self.record_event(
                        time=current_time,
                        event_type="hp_deadline",
                        job_id=job_id,
                        vc_name=info.get("vc_name"),
                        details={
                            "arrival_time": info["arrival_time"],
                            "deadline": deadline,
                            "status": "unstarted_past_deadline",
                            "wait_seconds": current_time - info["arrival_time"]
                        }
                    )
                    emitted.append(ev)
        return emitted

    def finalize_hp_deadlines(self, current_time: int, externally_truncated: bool = False) -> List[SimulationEvent]:
        """
        Final followup accounting: emits deadline_outcome (and hp_deadline) for remaining unstarted eligible jobs.
        If externally_truncated before deadline, outcome is 'censored'.
        """
        emitted = self.check_hp_deadlines(current_time + 1)
        for job_id, info in self.hp_gang_jobs.items():
            if info["eligible"] and info["start_time"] is None and not info["outcome_emitted"]:
                info["outcome_emitted"] = True
                deadline = info["deadline"]
                if externally_truncated and current_time < deadline:
                    self.hp_gang_censored += 1
                    ev = self.record_event(
                        time=current_time,
                        event_type="deadline_outcome",
                        job_id=job_id,
                        vc_name=info.get("vc_name"),
                        details={
                            "arrival_time": info["arrival_time"],
                            "deadline": deadline,
                            "start_time": None,
                            "outcome": "censored",
                            "observed_time": current_time,
                            "rule_version": self.spec.rule_version,
                        }
                    )
                    emitted.append(ev)
                else:
                    self.hp_gang_violations += 1
                    info["violated"] = True
                    ev = self.record_event(
                        time=current_time,
                        event_type="deadline_outcome",
                        job_id=job_id,
                        vc_name=info.get("vc_name"),
                        details={
                            "arrival_time": info["arrival_time"],
                            "deadline": deadline,
                            "start_time": None,
                            "outcome": "missed",
                            "observed_time": current_time,
                            "rule_version": self.spec.rule_version,
                        }
                    )
                    emitted.append(ev)
                    if not info["violation_emitted"]:
                        info["violation_emitted"] = True
                        ev_hp = self.record_event(
                            time=current_time,
                            event_type="hp_deadline",
                            job_id=job_id,
                            vc_name=info.get("vc_name"),
                            details={
                                "arrival_time": info["arrival_time"],
                                "deadline": deadline,
                                "status": "never_started",
                                "wait_seconds": current_time - info["arrival_time"]
                            }
                        )
                        emitted.append(ev_hp)
        return emitted

    def record_solver_invocation(
        self,
        time: int,
        call_id: int,
        decision_id: int,
        quota_window: int,
        scenario_id: int,
        scenario_seed: int,
        side: str,
        T_max: float,
        worker_count: int,
        vc_name: Optional[str] = None
    ) -> SimulationEvent:
        """Records an actual CpSolver.solve invocation attempt."""
        self.solver_calls_all_phase += 1
        if self.spec.score_start <= time < self.spec.score_end:
            self.total_solver_calls += 1
            self.scored_solver_calls += 1
        self.solver_calls_by_window[quota_window] = self.solver_calls_by_window.get(quota_window, 0) + 1
        return self.record_event(
            time=time,
            event_type="solver_invocation",
            vc_name=vc_name,
            details={
                "call_id": call_id,
                "decision_id": decision_id,
                "quota_window": quota_window,
                "scenario_id": scenario_id,
                "scenario_seed": scenario_seed,
                "side": side,
                "T_max": T_max,
                "worker_count": worker_count,
            }
        )

    def record_solver_calls(self, time: int, calls: int, reason: str):
        """Compatibility diagnostic event recording aggregate solver calls."""
        self.record_event(
            time=time,
            event_type="solver_call",
            details={"calls": calls, "reason": reason}
        )

    def compute_summary_metrics(
        self,
        current_time: int,
        remaining_queued: int,
        remaining_running: int,
        terminated: bool,
        truncated: bool,
        terminal_reason: str,
        initial_reward_gpu_hours: float = 0.0,
        step_reward_gpu_hours_sum: float = 0.0,
    ) -> EpisodeMetrics:
        if terminated or truncated or current_time >= self.spec.followup_end:
            self.finalize_hp_deadlines(current_time, externally_truncated=truncated)

        # Evaluate HP gang cohort
        eligible_jobs = [j for j in self.hp_gang_jobs.values() if j["eligible"]]
        cohort_size = len(eligible_jobs)
        violations = self.hp_gang_violations
        censored = self.hp_gang_censored

        if censored > 0 or cohort_size == 0:
            violation_rate = None
        else:
            violation_rate = violations / cohort_size

        # Absolute 300s quota windows intersecting [score_start, score_end)
        num_windows = max(1, count_intersecting_quota_windows(self.spec.score_start, self.spec.score_end, 300))
        calls_per_window = (self.scored_solver_calls / num_windows) if num_windows > 0 else None

        # Display evictions per 1000 starts
        if self.scored_spot_starts > 0:
            eviction_rate_disp = round(1000.0 * self.scored_spot_evictions / self.scored_spot_starts, 2)
        else:
            eviction_rate_disp = None

        raw_useful_hours = self.scored_useful_spot_gpu_seconds / 3600.0
        raw_useful_all_phase_hours = self.useful_spot_gpu_seconds_all_phase / 3600.0

        status = "completed"
        if truncated:
            status = "externally_truncated"
        elif not terminated and current_time < self.spec.followup_end:
            status = "running"

        return EpisodeMetrics(
            schema_version=SCHEMA_VERSION,
            run_id=self.run_id,
            episode_id=self.episode_id,
            status=status,
            terminal_reason=terminal_reason,
            useful_spot_gpu_seconds_score=self.scored_useful_spot_gpu_seconds,
            useful_spot_gpu_hours_score_raw=raw_useful_hours,
            initial_reward_gpu_hours=initial_reward_gpu_hours,
            step_reward_gpu_hours_sum=step_reward_gpu_hours_sum,
            hp_gang_cohort_count=cohort_size,
            hp_gang_violation_count=violations,
            hp_gang_violation_rate=violation_rate,
            hp_gang_censored_count=censored,
            spot_eviction_events_score=self.scored_spot_evictions,
            unique_spot_jobs_evicted_score=len(self.unique_spot_jobs_evicted_score),
            spot_lost_gpu_seconds_score=self.scored_spot_lost_gpu_seconds,
            solver_calls_score=self.scored_solver_calls,
            quota_windows_score=num_windows,
            calls_per_window=calls_per_window,
            decision_count=self.decision_count,
            useful_spot_gpu_seconds_all_phase=self.useful_spot_gpu_seconds_all_phase,
            useful_spot_gpu_hours_all_phase_raw=raw_useful_all_phase_hours,
            spot_eviction_events_all_phase=self.spot_evictions_all_phase,
            unique_spot_jobs_evicted_all_phase=len(self.unique_spot_jobs_evicted_all_phase),
            spot_lost_gpu_seconds_all_phase=self.spot_lost_gpu_seconds_all_phase,
            solver_calls_all_phase=self.solver_calls_all_phase,
            completed_hp_jobs=self.completed_jobs["HP"],
            completed_spot_jobs=self.completed_jobs["Spot"],
            remaining_queued_jobs=remaining_queued,
            remaining_running_jobs=remaining_running,
            useful_spot_gpu_hours_score_display=round(raw_useful_hours, 4),
            hp_gang_violation_rate_display=round(violation_rate, 4) if violation_rate is not None else None,
            calls_per_window_display=round(calls_per_window, 2) if calls_per_window is not None else None,
            eviction_events_per_1000_spot_starts_display=eviction_rate_disp,
            rule_version=self.spec.rule_version,
            score_start=self.spec.score_start,
            score_end=self.spec.score_end,
            followup_end=self.spec.followup_end,
            terminated=terminated,
            truncated=truncated,
        )
