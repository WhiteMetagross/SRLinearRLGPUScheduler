"""
measurement.py: Authoritative replay measurement, event schema validation,
pure disk reduction, and strict reconciliation audit for research_sim.
No mutable simulator state, verifier predictions, or cached metrics are read by the reducer.
"""

import json
import math
from itertools import chain
from typing import Iterable, List, Dict, Any, Optional, Tuple, Set
import numpy as np

from .types import EpisodeSpec, SimulationEvent

SCHEMA_VERSION = "2.0"

# Phase constants
PHASE_WARMUP = "warmup"
PHASE_SCORE = "score"
PHASE_FOLLOWUP = "followup"

# Credit rules
CREDIT_RULE_RESEARCH = "event_in_score_half_open"      # [score_start, score_end)
CREDIT_RULE_SOURCE = "event_in_score_left_open"        # (score_start, score_end]

# Authoritative event types for metrics derivation
AUTHORITATIVE_EVENT_TYPES: Set[str] = {
    "arrival",
    "start",
    "credit",
    "completion",
    "preemption",
    "lease_expiration",
    "hp_deadline",
    "deadline_outcome",
    "solver_invocation",
    "action",
    "placement",
    "deferral",
    "quota_update",
    "terminal",
}

# Mapping of authoritative event types to required details fields
REQUIRED_EVENT_DETAILS: Dict[str, List[str]] = {
    "arrival": [
        "job_type",
        "worker_num",
        "gpu_request",
        "submit_time",
        "executed_arrival_time",
        "vc_name",
        "hp_cohort_eligible",
    ],
    "start": [
        "job_type",
        "requested_worker_count",
        "actual_worker_count",
        "complete_start",
        "allocations",
    ],
    "credit": [
        "reason",
        "attempt_index",
        "credit_sequence",
        "credited_run_seconds",
        "gpu_equivalents",
        "credited_gpu_seconds",
        "in_score_window_research",
        "in_score_window_source",
    ],
    "completion": [
        "job_type",
        "completion_time",
    ],
    "preemption": [
        "attempt_index",
        "start_time",
        "worker_num",
        "gpu_request",
        "rollback_runtime_seconds",
        "lost_gpu_seconds",
        "remain_before",
        "remain_after",
        "overhead",
        "in_score_window",
    ],
    "lease_expiration": [
        "job_type",
        "lease_seconds",
        "rollback_seconds",
        "attempt_index",
        "start_time",
    ],
    "hp_deadline": [
        "arrival_time",
        "deadline",
        "status",
        "wait_seconds",
    ],
    "deadline_outcome": [
        "arrival_time",
        "deadline",
        "outcome",
        "observed_time",
        "rule_version",
    ],
    "solver_invocation": [
        "call_id",
        "decision_id",
        "quota_window",
        "scenario_id",
        "scenario_seed",
        "side",
        "T_max",
        "worker_count",
    ],
    "action": [
        "decision_id",
        "time",
        "state_version",
        "candidate_keys",
        "action_mask",
        "proposed_slot",
        "verify_requested",
        "executed_slot",
        "executed_deferral",
        "quota_window",
        "calls_used",
        "budget_after",
    ],
    "placement": [
        "slot",
        "key",
    ],
    "deferral": [
        "slot",
    ],
    "quota_update": [
        "spot_quota",
    ],
    "terminal": [
        "time",
        "terminal_reason",
        "remaining_queued",
        "remaining_running",
        "last_event_seq",
        "status",
    ],
}


def get_phase(time: int, score_start: int, score_end: int) -> str:
    """Returns the authoritative phase label for replay time t."""
    if time < score_start:
        return PHASE_WARMUP
    elif time < score_end:
        return PHASE_SCORE
    else:
        return PHASE_FOLLOWUP


def count_intersecting_quota_windows(score_start: int, score_end: int, window_size: int = 300) -> int:
    """
    Counts the number of absolute quota windows of length window_size intersecting [score_start, score_end).
    Formula: ((score_end - 1) // window_size) - (score_start // window_size) + 1.
    If score_end <= score_start: returns 0.
    """
    if score_end <= score_start:
        return 0
    return ((score_end - 1) // window_size) - (score_start // window_size) + 1


def is_finite_number(val: Any) -> bool:
    """Returns True if val is a finite int or float."""
    if isinstance(val, (int, np.integer)):
        return True
    if isinstance(val, (float, np.floating)):
        return math.isfinite(val)
    return False


def validate_primitive(val: Any, path: str = ""):
    """Recursively asserts that val is composed only of JSON-primitive types with finite numbers."""
    if val is None or isinstance(val, (bool, str, int)):
        return
    if isinstance(val, float):
        if not math.isfinite(val):
            raise ValueError(f"Non-finite float {val} at {path}")
        return
    if isinstance(val, (list, tuple)):
        for i, item in enumerate(val):
            validate_primitive(item, f"{path}[{i}]")
        return
    if isinstance(val, dict):
        for k, v in val.items():
            if not isinstance(k, (str, int)):
                raise ValueError(f"Non-string dictionary key {type(k)} at {path}")
            validate_primitive(v, f"{path}.{k}")
        return
    raise ValueError(f"Disallowed non-primitive type {type(val)} at {path}")


def validate_event_dict(
    event: Dict[str, Any],
    prev_event: Optional[Dict[str, Any]] = None,
    spec: Optional[EpisodeSpec] = None,
    expected_run_id: Optional[str] = None,
    expected_episode_id: Optional[str] = None
) -> None:
    """
    Validates a single serialized event record against the authoritative schema and constraints.
    Raises ValueError with exact failure details on corruption or schema violation.
    """
    # Required top level fields
    required_top = [
        "schema_version", "run_id", "episode_id", "event_seq", "time",
        "phase", "event_type", "job_id", "vc_name", "attempt_index", "details"
    ]
    for rf in required_top:
        if rf not in event:
            raise ValueError(f"Event missing required top-level field '{rf}': {event}")

    # Schema version
    if event["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"Event schema_version '{event['schema_version']}' does not match expected '{SCHEMA_VERSION}'"
        )

    # Run ID and Episode ID consistency
    if expected_run_id is not None and event["run_id"] != expected_run_id:
        raise ValueError(f"Event run_id '{event['run_id']}' != expected '{expected_run_id}'")
    if expected_episode_id is not None and event["episode_id"] != expected_episode_id:
        raise ValueError(f"Event episode_id '{event['episode_id']}' != expected '{expected_episode_id}'")

    # Sequence integrity
    seq = event["event_seq"]
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        raise ValueError(f"Invalid event_seq: {seq}")
    if prev_event is not None:
        expected_seq = prev_event["event_seq"] + 1
        if seq != expected_seq:
            raise ValueError(f"event_seq gap or duplicate: expected {expected_seq}, got {seq}")
    elif seq != 0:
        raise ValueError(f"First event_seq must be 0, got {seq}")

    # Time monotonic progression
    time = event["time"]
    if not isinstance(time, int) or isinstance(time, bool) or time < 0:
        raise ValueError(f"Invalid integer replay time: {time}")
    if prev_event is not None and time < prev_event["time"]:
        raise ValueError(f"Decreasing replay time: {time} < {prev_event['time']} at seq {seq}")

    # Phase correctness
    phase = event["phase"]
    if phase not in (PHASE_WARMUP, PHASE_SCORE, PHASE_FOLLOWUP):
        raise ValueError(f"Malformed phase '{phase}' at seq {seq}")
    if spec is not None:
        expected_phase = get_phase(time, spec.score_start, spec.score_end)
        if phase != expected_phase:
            raise ValueError(
                f"Phase mismatch at time {time}: event has '{phase}', expected '{expected_phase}'"
            )

    # Details must be primitive and finite
    details = event["details"]
    if not isinstance(details, dict):
        raise ValueError(f"Event details must be a dict, got {type(details)} at seq {seq}")
    validate_primitive(details, f"event[{seq}].details")

    # Authoritative event required fields
    etype = event["event_type"]
    if etype in REQUIRED_EVENT_DETAILS:
        for req_field in REQUIRED_EVENT_DETAILS[etype]:
            if req_field not in details:
                raise ValueError(
                    f"Authoritative event '{etype}' at seq {seq} missing required detail '{req_field}'"
                )

    # Specific event validations
    if etype == "credit":
        credited_sec = details["credited_gpu_seconds"]
        if not is_finite_number(credited_sec) or credited_sec <= 0:
            raise ValueError(f"Non-positive or non-finite credit {credited_sec} at seq {seq}")
        if details.get("attempt_index") is None or details.get("credit_sequence") is None:
            raise ValueError(f"Credit event at seq {seq} missing attempt_index or credit_sequence")
        if details.get("attempt_index") != event.get("attempt_index"):
            raise ValueError(f"Credit attempt index differs between event and details at seq {seq}.")
        if details.get("gpu_equivalents", 0) <= 0 or details.get("credited_run_seconds", 0) <= 0:
            raise ValueError(f"Credit event has nonpositive GPU equivalents or run seconds at seq {seq}.")
        if details.get("credited_run_seconds", 0) * details.get("gpu_equivalents", 0) != credited_sec:
            expected_credit = details.get("credited_run_seconds", 0) * details.get("gpu_equivalents", 0)
            if not math.isclose(float(expected_credit), float(credited_sec), rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError(f"Credit amount disagrees with run seconds times GPU equivalents at seq {seq}.")
        if details["in_score_window_research"] != (spec is not None and spec.score_start <= time < spec.score_end):
            if spec is not None:
                raise ValueError(f"Research credit inclusion flag is incorrect at seq {seq}.")
        if spec is not None and details["in_score_window_source"] != (spec.score_start < time <= spec.score_end):
            raise ValueError(f"Source compatible credit inclusion flag is incorrect at seq {seq}.")

    if etype == "preemption":
        lost_sec = details["lost_gpu_seconds"]
        if not is_finite_number(lost_sec) or lost_sec < 0:
            raise ValueError(f"Negative or non-finite lost_gpu_seconds {lost_sec} at seq {seq}")
        expected_lost = (
            float(details["rollback_runtime_seconds"])
            * int(details["worker_num"])
            * float(details["gpu_request"])
        )
        if not math.isclose(float(lost_sec), expected_lost, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError(f"Lost work does not equal rollback runtime times requested GPU capacity at seq {seq}.")
        expected_remain = float(details["rollback_runtime_seconds"]) + float(details["overhead"])
        remain_delta = float(details["remain_after"]) - float(details["remain_before"])
        if not math.isclose(remain_delta, expected_remain, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"Preemption remain delta does not equal rollback plus overhead at seq {seq}.")
        if details["rollback_runtime_seconds"] < 0 or details["overhead"] < 0 or details["worker_num"] < 1 or details["gpu_request"] <= 0:
            raise ValueError(f"Invalid preemption capacity or runtime values at seq {seq}.")

    if etype == "deadline_outcome":
        outcome = details["outcome"]
        if outcome not in ("met", "missed", "censored"):
            raise ValueError(f"Invalid deadline outcome '{outcome}' at seq {seq}")
        if details.get("observed_time") != time:
            raise ValueError(f"deadline_outcome observed_time differs from event time at seq {seq}.")


def reduce_replay_events(
    events: Iterable[Dict[str, Any]],
    spec: EpisodeSpec,
    solver_invocations: Optional[List[Dict[str, Any]]] = None,
    expected_run_id: Optional[str] = None,
    expected_episode_id: Optional[str] = None,
    events_validated: bool = False,
) -> Dict[str, Any]:
    """
    Pure reducer that reconstructs raw metrics and accounting totals from serialized executed events alone.
    Does not read mutable simulator counters, verifier predictions, or cached metrics.json.
    ``events_validated`` may be true only when the caller completed the equivalent
    schema and identity validation, including any verified prefix cache checks.
    Raises ValueError immediately on any corruption, missing required detail, duplicate key, or inconsistency.
    """
    event_iterator = iter(events)
    try:
        first_event = next(event_iterator)
    except StopIteration as exc:
        raise ValueError("Cannot reduce empty events list.") from exc

    event_run_id = first_event.get("run_id")
    event_episode_id = first_event.get("episode_id")
    if not event_run_id or not event_episode_id:
        raise ValueError("Events require nonempty run_id and episode_id.")
    expected_run_id = expected_run_id or event_run_id
    expected_episode_id = expected_episode_id or event_episode_id

    # Validation pass & state accumulation
    prev_event: Optional[Dict[str, Any]] = None
    seen_credit_keys: Set[Tuple[Any, int, int]] = set()
    spot_attempt_indices: Dict[Any, int] = {}
    active_spot_attempts: Dict[Any, Tuple[int, int]] = {}
    next_credit_sequence: Dict[Tuple[Any, int], int] = {}

    # HP job tracking: job_id -> info
    hp_arrivals: Dict[Any, Dict[str, Any]] = {}
    all_arrival_ids: Set[Any] = set()
    hp_starts: Dict[Any, int] = {}
    hp_deadline_outcomes_emitted: Dict[Any, Dict[str, Any]] = {}

    # Scored and all-phase accumulators
    scored_credit_seconds_list: List[float] = []
    source_credit_seconds_list: List[float] = []
    all_credit_seconds_list: List[float] = []
    prefix_credit_seconds_list: List[float] = []

    spot_evictions_scored = 0
    spot_evictions_all_phase = 0
    spot_evicted_jobs_scored: Set[Any] = set()
    spot_evicted_jobs_all_phase: Set[Any] = set()
    spot_lost_seconds_scored_list: List[float] = []
    spot_lost_seconds_all_phase_list: List[float] = []

    scored_spot_starts = 0
    all_spot_starts = 0
    completed_spot_jobs = 0
    completed_hp_jobs = 0

    decision_count = 0
    first_action_seq: Optional[int] = None
    solver_calls_scored = 0
    solver_calls_all_phase = 0
    solver_calls_by_window: Dict[int, int] = {}
    solver_call_ids_events: Set[int] = set()

    terminal_event: Optional[Dict[str, Any]] = None
    terminal_events = 0
    max_time_observed = 0

    for ev in chain((first_event,), event_iterator):
        if terminal_event is not None:
            raise ValueError(
                f"Terminal event at seq {terminal_event['event_seq']} is not the final event."
            )
        if not events_validated:
            validate_event_dict(
                ev,
                prev_event=prev_event,
                spec=spec,
                expected_run_id=expected_run_id,
                expected_episode_id=expected_episode_id,
            )
        prev_event = ev
        t = ev["time"]
        max_time_observed = max(max_time_observed, t)
        etype = ev["event_type"]
        phase = ev["phase"]
        details = ev["details"]
        jid = ev["job_id"]

        # Track spot start attempts and counts
        if etype == "start":
            jtype = details.get("job_type", "HP" if jid in hp_arrivals else "Spot")
            requested = details["requested_worker_count"]
            actual = details["actual_worker_count"]
            allocations = details["allocations"]
            if requested < 1 or actual < 0 or not isinstance(allocations, list):
                raise ValueError(f"Invalid start worker count or allocation record for job {jid} at seq {ev['event_seq']}.")
            if details["complete_start"] and (actual != requested or (actual > 0 and not allocations)):
                raise ValueError(f"Complete start has inconsistent worker mapping for job {jid} at seq {ev['event_seq']}.")
            if jtype == "Spot":
                all_spot_starts += 1
                if phase == PHASE_SCORE:
                    scored_spot_starts += 1
                # Track attempt index
                curr_att = spot_attempt_indices.get(jid, -1) + 1
                spot_attempt_indices[jid] = curr_att
                if jid in active_spot_attempts:
                    raise ValueError(f"Spot job {jid} started a new attempt before its prior attempt ended at seq {ev['event_seq']}.")
                if ev.get("attempt_index") != curr_att:
                    raise ValueError(
                        f"Spot job {jid} start attempt_index {ev['attempt_index']} != expected {curr_att} at seq {ev['event_seq']}"
                    )
                active_spot_attempts[jid] = (curr_att, t)
                next_credit_sequence[(jid, curr_att)] = 0
            elif jtype == "HP":
                if details.get("complete_start", False):
                    if jid not in hp_starts:
                        hp_starts[jid] = t

        # Track arrival
        elif etype == "arrival":
            if jid in all_arrival_ids:
                raise ValueError(f"Duplicate arrival for job {jid} at seq {ev['event_seq']}.")
            all_arrival_ids.add(jid)
            if details["job_type"] == "HP":
                hp_arrivals[jid] = {
                    "arrival_time": details["executed_arrival_time"],
                    "worker_num": details["worker_num"],
                    "gpu_request": details["gpu_request"],
                    "vc_name": details["vc_name"],
                    "hp_cohort_eligible": details["hp_cohort_eligible"],
                }

        # Track credit
        elif etype == "credit":
            gpu_sec = float(details["credited_gpu_seconds"])
            cap = details.get("job_credit_cap_gpu_seconds")
            cumulative = details.get("cumulative_credited_gpu_seconds")
            if cap is not None and cumulative is not None:
                if float(cumulative) > float(cap) + 1e-6:
                    raise ValueError(
                        f"Spot credit exceeds physical job service for {jid} at seq {ev['event_seq']}."
                    )
            all_credit_seconds_list.append(gpu_sec)

            # Unique credit key enforcement: (job_id, attempt_index, credit_sequence)
            att_idx = details["attempt_index"]
            c_seq = details["credit_sequence"]
            c_key = (jid, att_idx, c_seq)
            if c_key in seen_credit_keys:
                raise ValueError(f"Duplicate positive credit key {c_key} at seq {ev['event_seq']}")
            active_attempt = active_spot_attempts.get(jid)
            if active_attempt is None or active_attempt[0] != att_idx:
                raise ValueError(f"Credit references no active Spot attempt {c_key} at seq {ev['event_seq']}.")
            expected_credit_seq = next_credit_sequence.get((jid, att_idx), 0)
            if c_seq != expected_credit_seq:
                raise ValueError(f"Credit sequence for {c_key} is not the expected {expected_credit_seq} at seq {ev['event_seq']}.")
            next_credit_sequence[(jid, att_idx)] = expected_credit_seq + 1
            seen_credit_keys.add(c_key)

            # Research scoring interval: [score_start, score_end)
            if spec.score_start <= t < spec.score_end:
                scored_credit_seconds_list.append(gpu_sec)
                if first_action_seq is None:
                    prefix_credit_seconds_list.append(gpu_sec)

            # Source compatible scoring interval: (score_start, score_end]
            if spec.score_start < t <= spec.score_end:
                source_credit_seconds_list.append(gpu_sec)

        # Track preemption
        elif etype == "preemption":
            active_attempt = active_spot_attempts.get(jid)
            if active_attempt is None:
                raise ValueError(f"Preemption references a Spot job that is not running: {jid} at seq {ev['event_seq']}.")
            if active_attempt != (details["attempt_index"], details["start_time"]):
                raise ValueError(f"Preemption attempt or start time differs from active job {jid} at seq {ev['event_seq']}.")
            del active_spot_attempts[jid]
            spot_evictions_all_phase += 1
            spot_evicted_jobs_all_phase.add(jid)
            lost_sec = float(details["lost_gpu_seconds"])
            spot_lost_seconds_all_phase_list.append(lost_sec)

            if phase == PHASE_SCORE:
                spot_evictions_scored += 1
                spot_evicted_jobs_scored.add(jid)
                spot_lost_seconds_scored_list.append(lost_sec)

        # Track lossless Chronus lease boundary. This closes one execution
        # attempt but is not an eviction and carries no lost work.
        elif etype == "lease_expiration":
            if details["rollback_seconds"] != 0:
                raise ValueError(
                    f"Lease expiration must preserve work at seq {ev['event_seq']}."
                )
            if details["job_type"] == "Spot":
                active_attempt = active_spot_attempts.get(jid)
                expected = (details["attempt_index"], details["start_time"])
                if active_attempt != expected:
                    raise ValueError(
                        f"Lease expiration differs from active Spot attempt {jid} at seq {ev['event_seq']}."
                    )
                del active_spot_attempts[jid]

        # Track completion
        elif etype == "completion":
            jtype = details.get("job_type")
            if jtype == "Spot":
                if jid not in active_spot_attempts:
                    raise ValueError(f"Spot completion references a job that is not running: {jid} at seq {ev['event_seq']}.")
                del active_spot_attempts[jid]
                completed_spot_jobs += 1
            elif jtype == "HP":
                completed_hp_jobs += 1

        # Track deadline outcome emitted
        elif etype == "deadline_outcome":
            if jid not in hp_arrivals or not hp_arrivals[jid]["hp_cohort_eligible"]:
                raise ValueError(f"deadline_outcome references noneligible or unknown HP job {jid} at seq {ev['event_seq']}.")
            if jid in hp_deadline_outcomes_emitted:
                raise ValueError(f"Duplicate deadline_outcome emitted for HP job {jid} at seq {ev['event_seq']}")
            hp_deadline_outcomes_emitted[jid] = details

        # Track solver invocation
        elif etype == "solver_invocation":
            cid = details["call_id"]
            if cid in solver_call_ids_events:
                raise ValueError(f"Duplicate solver call_id {cid} at seq {ev['event_seq']}")
            solver_call_ids_events.add(cid)
            solver_calls_all_phase += 1
            qw = details["quota_window"]
            solver_calls_by_window[qw] = solver_calls_by_window.get(qw, 0) + 1
            if phase == PHASE_SCORE:
                solver_calls_scored += 1

        # Track action
        elif etype == "action":
            if first_action_seq is None:
                first_action_seq = ev["event_seq"]
            if phase == PHASE_SCORE:
                decision_count += 1

        # Track terminal
        elif etype == "terminal":
            terminal_events += 1
            if terminal_events > 1:
                raise ValueError(f"Duplicate terminal event at seq {ev['event_seq']}.")
            if details["last_event_seq"] != ev["event_seq"]:
                raise ValueError(
                    f"terminal.last_event_seq {details['last_event_seq']} != event_seq {ev['event_seq']}"
                )
            terminal_event = ev

    if terminal_events != 1:
        raise ValueError(f"Completed event stream requires exactly one terminal event, found {terminal_events}.")
    terminal_status = terminal_event["details"].get("status")
    if terminal_status not in ("completed", "externally_truncated"):
        raise ValueError(f"Invalid terminal status '{terminal_status}'.")

    # Verify solver invocation records against disk solver_invocations if provided
    if solver_invocations is not None:
        disk_call_ids = {row["call_id"] for row in solver_invocations if "call_id" in row}
        if disk_call_ids != solver_call_ids_events:
            diff = disk_call_ids.symmetric_difference(solver_call_ids_events)
            raise ValueError(f"Solver invocation records mismatch between events and disk records: diff={diff}")
        for row in solver_invocations:
            if row.get("status") == "STARTED":
                raise ValueError(f"Found unfinalized STARTED solver invocation on completed run: {row}")

    # HP Cohort derivation and outcome evaluation
    cohort_jobs: List[Dict[str, Any]] = []
    waits_started: List[int] = []
    cohort_missed = 0
    cohort_censored = 0
    cohort_met = 0

    for jid, info in hp_arrivals.items():
        arr_time = info["arrival_time"]
        wn = info["worker_num"]
        req = info["gpu_request"]
        is_whole_card_gang = (wn > 1) and (req >= 1.0) and (req % 1.0 == 0.0)
        in_score_window = (spec.score_start <= arr_time < spec.score_end)
        has_followup = (arr_time + spec.L_H <= spec.followup_end)

        is_eligible = is_whole_card_gang and in_score_window and has_followup
        if is_eligible != info["hp_cohort_eligible"]:
            raise ValueError(
                f"Cohort eligibility mismatch for HP job {jid}: computed {is_eligible} != arrival event {info['hp_cohort_eligible']}"
            )

        if not is_eligible:
            continue

        deadline = arr_time + spec.L_H
        first_start = hp_starts.get(jid)

        if first_start is not None:
            wait = first_start - arr_time
            waits_started.append(wait)
            if first_start <= deadline:
                outcome = "met"
                cohort_met += 1
            else:
                outcome = "missed"
                cohort_missed += 1
        else:
            # Not started
            # If replay reached or passed deadline: missed. If externally truncated before deadline: censored.
            is_externally_truncated = (
                terminal_event is not None and
                terminal_event["details"].get("status") == "externally_truncated"
            )
            if max_time_observed >= deadline or not is_externally_truncated:
                outcome = "missed"
                cohort_missed += 1
            else:
                outcome = "censored"
                cohort_censored += 1

        cohort_jobs.append({
            "job_id": jid,
            "arrival_time": arr_time,
            "deadline": deadline,
            "first_start": first_start,
            "outcome": outcome,
        })

        # Cross-check emitted deadline_outcome event
        if jid not in hp_deadline_outcomes_emitted:
            raise ValueError(f"Eligible HP job {jid} has no deadline_outcome event.")
        emitted = hp_deadline_outcomes_emitted[jid]
        if emitted["outcome"] != outcome:
            raise ValueError(
                f"Derived outcome '{outcome}' != emitted deadline_outcome '{emitted['outcome']}' for HP job {jid}"
            )

    cohort_count = len(cohort_jobs)
    if cohort_censored > 0 or cohort_count == 0:
        violation_rate = None
    else:
        violation_rate = cohort_missed / cohort_count

    # Wait percentiles for observed starts
    if waits_started:
        wait_p95 = float(np.percentile(waits_started, 95))
        wait_p99 = float(np.percentile(waits_started, 99))
    else:
        wait_p95 = None
        wait_p99 = None

    # Quota windows
    num_quota_windows = count_intersecting_quota_windows(spec.score_start, spec.score_end, 300)
    calls_per_window = (solver_calls_scored / num_quota_windows) if num_quota_windows > 0 else None

    # Useful work sums using math.fsum
    useful_spot_gpu_seconds_score = math.fsum(scored_credit_seconds_list)
    useful_spot_gpu_hours_score_raw = useful_spot_gpu_seconds_score / 3600.0
    useful_spot_gpu_seconds_source = math.fsum(source_credit_seconds_list)
    useful_spot_gpu_hours_source_raw = useful_spot_gpu_seconds_source / 3600.0
    useful_spot_gpu_seconds_all_phase = math.fsum(all_credit_seconds_list)
    useful_spot_gpu_hours_all_phase_raw = useful_spot_gpu_seconds_all_phase / 3600.0
    prefix_credit_seconds = math.fsum(prefix_credit_seconds_list)
    if first_action_seq is None:
        step_credit_seconds = 0.0
    else:
        step_credit_seconds = useful_spot_gpu_seconds_score - prefix_credit_seconds

    # Lost work sums using math.fsum
    spot_lost_gpu_seconds_score = math.fsum(spot_lost_seconds_scored_list)
    spot_lost_gpu_seconds_all_phase = math.fsum(spot_lost_seconds_all_phase_list)

    # Eviction rate per 1000 starts
    if scored_spot_starts > 0:
        eviction_rate_display = round(1000.0 * spot_evictions_scored / scored_spot_starts, 2)
    else:
        eviction_rate_display = None

    # Determine status
    if terminal_event is not None:
        status = terminal_event["details"].get("status", "completed")
        terminal_reason = terminal_event["details"].get("terminal_reason", "")
        rem_queued = terminal_event["details"].get("remaining_queued", 0)
        rem_running = terminal_event["details"].get("remaining_running", 0)
    else:
        status = "unknown"
        terminal_reason = ""
        rem_queued = 0
        rem_running = 0

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": expected_run_id or event_run_id,
        "episode_id": expected_episode_id or event_episode_id,
        "status": status,
        "terminal_reason": terminal_reason,
        "useful_spot_gpu_seconds_score": useful_spot_gpu_seconds_score,
        "useful_spot_gpu_hours_score_raw": useful_spot_gpu_hours_score_raw,
        "useful_spot_gpu_seconds_source_compatible": useful_spot_gpu_seconds_source,
        "useful_spot_gpu_hours_source_compatible_raw": useful_spot_gpu_hours_source_raw,
        "useful_spot_gpu_seconds_all_phase": useful_spot_gpu_seconds_all_phase,
        "useful_spot_gpu_hours_all_phase_raw": useful_spot_gpu_hours_all_phase_raw,
        "hp_gang_cohort_count": cohort_count,
        "hp_gang_violation_count": cohort_missed,
        "hp_gang_violation_rate": violation_rate,
        "hp_gang_censored_count": cohort_censored,
        "hp_gang_waits_observed": len(waits_started),
        "hp_gang_wait_p95": wait_p95,
        "hp_gang_wait_p99": wait_p99,
        "spot_eviction_events_score": spot_evictions_scored,
        "unique_spot_jobs_evicted_score": len(spot_evicted_jobs_scored),
        "spot_lost_gpu_seconds_score": spot_lost_gpu_seconds_score,
        "spot_eviction_events_all_phase": spot_evictions_all_phase,
        "unique_spot_jobs_evicted_all_phase": len(spot_evicted_jobs_all_phase),
        "spot_lost_gpu_seconds_all_phase": spot_lost_gpu_seconds_all_phase,
        "solver_calls_score": solver_calls_scored,
        "solver_calls_all_phase": solver_calls_all_phase,
        "solver_calls_by_window": solver_calls_by_window,
        "quota_windows_score": num_quota_windows,
        "calls_per_window": calls_per_window,
        "decision_count": decision_count,
        "initial_reward_gpu_hours": prefix_credit_seconds / 3600.0,
        "step_reward_gpu_hours_sum": step_credit_seconds / 3600.0,
        "scored_spot_starts": scored_spot_starts,
        "all_spot_starts": all_spot_starts,
        "completed_hp_jobs": completed_hp_jobs,
        "completed_spot_jobs": completed_spot_jobs,
        "remaining_queued_jobs": rem_queued,
        "remaining_running_jobs": rem_running,
        # Rounded presentation fields
        "useful_spot_gpu_hours_score_display": round(useful_spot_gpu_hours_score_raw, 4),
        "hp_gang_violation_rate_display": round(violation_rate, 4) if violation_rate is not None else None,
        "calls_per_window_display": round(calls_per_window, 2) if calls_per_window is not None else None,
        "eviction_events_per_1000_spot_starts_display": eviction_rate_display,
    }


def reconcile_metrics(
    online: Dict[str, Any],
    reduced: Dict[str, Any],
    float_tol_abs: float = 1e-6,
    float_tol_rel: float = 1e-9,
    check_rewards: bool = False,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Compares online metrics against disk-reduced metrics.
    Exact comparison on integer counts, cohort size, status, IDs.
    math.fsum tolerance comparison on floating point GPU seconds.
    Returns (True, None) if matching, or (False, first_mismatch_dict).
    """
    exact_fields = [
        "schema_version",
        "run_id",
        "episode_id",
        "status",
        "terminal_reason",
        "hp_gang_cohort_count",
        "hp_gang_violation_count",
        "hp_gang_censored_count",
        "spot_eviction_events_score",
        "unique_spot_jobs_evicted_score",
        "spot_eviction_events_all_phase",
        "unique_spot_jobs_evicted_all_phase",
        "solver_calls_score",
        "solver_calls_all_phase",
        "quota_windows_score",
        "decision_count",
        "completed_hp_jobs",
        "completed_spot_jobs",
        "remaining_queued_jobs",
        "remaining_running_jobs",
    ]

    for field in exact_fields:
        if field not in online or field not in reduced:
            return False, {"field": field, "expected_reduced": reduced.get(field), "actual_online": online.get(field), "comparison": "missing_required_field"}
        if online[field] != reduced[field]:
            return False, {
                "field": field,
                "expected_reduced": reduced[field],
                "actual_online": online[field],
                "comparison": "exact",
            }

    # Floating point comparison fields
    float_fields = [
        "useful_spot_gpu_seconds_score",
        "useful_spot_gpu_seconds_all_phase",
        "spot_lost_gpu_seconds_score",
        "spot_lost_gpu_seconds_all_phase",
    ]
    for field in float_fields:
        if field in online or field in reduced:
            if field not in online or field not in reduced:
                return False, {"field": field, "expected_reduced": reduced.get(field), "actual_online": online.get(field), "comparison": "missing_required_field"}
            if online[field] is None or reduced[field] is None:
                if online[field] is not reduced[field]:
                    return False, {"field": field, "expected_reduced": reduced[field], "actual_online": online[field], "comparison": "null_check"}
                continue
            v_on = float(online[field])
            v_red = float(reduced[field])
            tol = float_tol_abs + float_tol_rel * max(abs(v_on), abs(v_red))
            if abs(v_on - v_red) > tol:
                return False, {
                    "field": field,
                    "expected_reduced": v_red,
                    "actual_online": v_on,
                    "diff": abs(v_on - v_red),
                    "tolerance": tol,
                    "comparison": "float_tol",
                }

    # Nullable float rates
    rate_fields = ["hp_gang_violation_rate", "calls_per_window"]
    for field in rate_fields:
        if field in online or field in reduced:
            if field not in online or field not in reduced:
                return False, {"field": field, "expected_reduced": reduced.get(field), "actual_online": online.get(field), "comparison": "missing_required_field"}
            v_on = online[field]
            v_red = reduced[field]
            if v_on is None and v_red is None:
                continue
            if (v_on is None) != (v_red is None):
                return False, {
                    "field": field,
                    "expected_reduced": v_red,
                    "actual_online": v_on,
                    "comparison": "null_check",
                }

    # RL episode reward is split at the first executed action. The independent
    # reducer supplies these two values from the credit event stream.
    reward_fields = ("initial_reward_gpu_hours", "step_reward_gpu_hours_sum") if check_rewards else ()
    for field in reward_fields:
        if field in online or field in reduced:
            if field not in online or field not in reduced:
                return False, {"field": field, "expected_reduced": reduced.get(field), "actual_online": online.get(field), "comparison": "missing_required_field"}
            v_on = float(online[field])
            v_red = float(reduced[field])
            tol = float_tol_abs / 3600.0 + float_tol_rel * max(abs(v_on), abs(v_red))
            if not math.isfinite(v_on) or abs(v_on - v_red) > tol:
                return False, {"field": field, "expected_reduced": v_red, "actual_online": v_on, "diff": abs(v_on-v_red), "tolerance": tol, "comparison": "float_tol"}
            tol = float_tol_abs + float_tol_rel * max(abs(v_on), abs(v_red))
            if abs(v_on - v_red) > tol:
                return False, {
                    "field": field,
                    "expected_reduced": v_red,
                    "actual_online": v_on,
                    "diff": abs(v_on - v_red),
                    "tolerance": tol,
                    "comparison": "float_tol",
                }

    return True, None


def serialize_event_record(event: Any) -> str:
    """Serializes an event to a single compact UTF-8 JSON line with sort_keys=True and no NaN/inf."""
    if hasattr(event, "to_dict"):
        d = event.to_dict()
    elif hasattr(event, "__dict__"):
        d = dict(event.__dict__)
    elif isinstance(event, dict):
        d = dict(event)
    else:
        raise ValueError(f"Unserializable event type: {type(event)}")

    # Ensure no NaN or inf in serialization
    return json.dumps(d, sort_keys=True, separators=(",", ":"), allow_nan=False)
