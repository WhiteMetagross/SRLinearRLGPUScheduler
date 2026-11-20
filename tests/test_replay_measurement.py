"""
test_replay_measurement.py: Unit tests for event schema validation, pure disk reduction,
credit boundaries, HP service cohort outcomes, Spot evictions, corruption rejection,
and strict reconciliation audit.
"""

import pytest
import math
from typing import Dict, Any, List

from research_sim.types import EpisodeSpec
from research_sim.measurement import (
    SCHEMA_VERSION,
    PHASE_WARMUP,
    PHASE_SCORE,
    PHASE_FOLLOWUP,
    get_phase,
    count_intersecting_quota_windows,
    validate_event_dict,
    reduce_replay_events,
    reconcile_metrics,
    serialize_event_record,
)


@pytest.fixture
def base_spec():
    return EpisodeSpec(
        warmup_start=0,
        score_start=3000,
        score_end=6000,
        followup_end=9600,
        L_H=3600,
        B_call=2000,
        K=4,
        M=8,
    )


def make_event(
    seq: int,
    time: int,
    etype: str,
    spec: EpisodeSpec,
    job_id: Any = None,
    vc_name: str = "vc1",
    attempt_index: Any = None,
    details: Dict[str, Any] = None,
    run_id: str = "test_run_1",
    episode_id: str = "test_ep_1",
) -> Dict[str, Any]:
    if etype == "credit" and attempt_index is None and details is not None:
        attempt_index = details.get("attempt_index")
    phase = get_phase(time, spec.score_start, spec.score_end)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "episode_id": episode_id,
        "event_seq": seq,
        "time": time,
        "phase": phase,
        "event_type": etype,
        "job_id": job_id,
        "vc_name": vc_name,
        "attempt_index": attempt_index,
        "details": details or {},
    }


def test_count_intersecting_quota_windows():
    # 3000 to 6000 with 300s window
    # floor((6000-1)//300) = 19, floor(3000//300) = 10 -> 19 - 10 + 1 = 10 windows
    assert count_intersecting_quota_windows(3000, 6000, 300) == 10
    # Aligned 0 to 300: 1 window
    assert count_intersecting_quota_windows(0, 300, 300) == 1
    # Nonaligned: 100 to 400: (399//300) - (100//300) + 1 = 1 - 0 + 1 = 2 windows
    assert count_intersecting_quota_windows(100, 400, 300) == 2
    # Nonpositive interval
    assert count_intersecting_quota_windows(500, 500, 300) == 0
    assert count_intersecting_quota_windows(600, 500, 300) == 0


def test_credit_boundaries(base_spec):
    spec = base_spec
    events = []
    seq = 0

    # Start spot job 101 at t=0
    events.append(make_event(
        seq, 0, "start", spec, job_id=101, attempt_index=0,
        details={
            "job_type": "Spot", "requested_worker_count": 1, "actual_worker_count": 1,
            "complete_start": True, "allocations": [{"node_name": "node_0", "card_ids": [0]}]
        }
    ))
    seq += 1

    # Credit at score_start - 1 (t=2999) -> warmup, excluded from research score
    events.append(make_event(
        seq, 2999, "credit", spec, job_id=101, attempt_index=0,
        details={
            "reason": "checkpoint", "attempt_index": 0, "credit_sequence": 0,
            "credited_run_seconds": 2999, "gpu_equivalents": 1.0, "credited_gpu_seconds": 2999.0,
            "in_score_window_research": False, "in_score_window_source": False
        }
    ))
    seq += 1

    # Credit at score_start (t=3000) -> score, included in research score, excluded from source
    events.append(make_event(
        seq, 3000, "credit", spec, job_id=101, attempt_index=0,
        details={
            "reason": "checkpoint", "attempt_index": 0, "credit_sequence": 1,
            "credited_run_seconds": 1, "gpu_equivalents": 1.0, "credited_gpu_seconds": 1.0,
            "in_score_window_research": True, "in_score_window_source": False
        }
    ))
    seq += 1

    # Two credits in one second: at t=4000
    events.append(make_event(
        seq, 4000, "credit", spec, job_id=101, attempt_index=0,
        details={
            "reason": "checkpoint", "attempt_index": 0, "credit_sequence": 2,
            "credited_run_seconds": 1000, "gpu_equivalents": 1.0, "credited_gpu_seconds": 1000.0,
            "in_score_window_research": True, "in_score_window_source": True
        }
    ))
    seq += 1
    events.append(make_event(
        seq, 4000, "credit", spec, job_id=101, attempt_index=0,
        details={
            "reason": "checkpoint", "attempt_index": 0, "credit_sequence": 3,
            "credited_run_seconds": 500, "gpu_equivalents": 1.0, "credited_gpu_seconds": 500.0,
            "in_score_window_research": True, "in_score_window_source": True
        }
    ))
    seq += 1

    # Credit at score_end - 1 (t=5999) -> score, included in both research and source
    events.append(make_event(
        seq, 5999, "credit", spec, job_id=101, attempt_index=0,
        details={
            "reason": "checkpoint", "attempt_index": 0, "credit_sequence": 4,
            "credited_run_seconds": 1999, "gpu_equivalents": 1.0, "credited_gpu_seconds": 1999.0,
            "in_score_window_research": True, "in_score_window_source": True
        }
    ))
    seq += 1

    # Credit at score_end (t=6000) -> followup, excluded from research, included in source
    events.append(make_event(
        seq, 6000, "credit", spec, job_id=101, attempt_index=0,
        details={
            "reason": "checkpoint", "attempt_index": 0, "credit_sequence": 5,
            "credited_run_seconds": 1, "gpu_equivalents": 1.0, "credited_gpu_seconds": 1.0,
            "in_score_window_research": False, "in_score_window_source": True
        }
    ))
    seq += 1

    # Terminal event
    events.append(make_event(
        seq, 7000, "terminal", spec,
        details={
            "time": 7000, "terminal_reason": "planned_horizon_reached",
            "remaining_queued": 0, "remaining_running": 0,
            "last_event_seq": seq, "status": "completed"
        }
    ))

    reduced = reduce_replay_events(events, spec)
    # Research score interval [3000, 6000): credits at 3000 (1.0) + 4000 (1000.0) + 4000 (500.0) + 5999 (1999.0) = 3500.0
    assert reduced["useful_spot_gpu_seconds_score"] == 3500.0
    assert reduced["useful_spot_gpu_hours_score_raw"] == 3500.0 / 3600.0

    # Source compatible interval (3000, 6000]: credits at 4000 (1000) + 4000 (500) + 5999 (1999) + 6000 (1.0) = 3500.0
    assert reduced["useful_spot_gpu_seconds_source_compatible"] == 3500.0

    # All phase: 2999 + 1 + 1000 + 500 + 1999 + 1 = 6500.0
    assert reduced["useful_spot_gpu_seconds_all_phase"] == 6500.0


def test_hp_service_cohort_and_deadlines(base_spec):
    spec = base_spec
    events = []
    seq = 0

    # Arrivals in chronological order
    events.append(make_event(
        seq, 3100, "arrival", spec, job_id=1,
        details={
            "job_type": "HP", "worker_num": 1, "gpu_request": 1.0,
            "submit_time": 3100, "executed_arrival_time": 3100,
            "vc_name": "vc1", "hp_cohort_eligible": False
        }
    ))
    seq += 1

    events.append(make_event(
        seq, 3200, "arrival", spec, job_id=2,
        details={
            "job_type": "HP", "worker_num": 2, "gpu_request": 0.5,
            "submit_time": 3200, "executed_arrival_time": 3200,
            "vc_name": "vc1", "hp_cohort_eligible": False
        }
    ))
    seq += 1

    events.append(make_event(
        seq, 3300, "arrival", spec, job_id=3,
        details={
            "job_type": "HP", "worker_num": 2, "gpu_request": 1.0,
            "submit_time": 3300, "executed_arrival_time": 3300,
            "vc_name": "vc1", "hp_cohort_eligible": True
        }
    ))
    seq += 1

    events.append(make_event(
        seq, 3400, "arrival", spec, job_id=4,
        details={
            "job_type": "HP", "worker_num": 2, "gpu_request": 1.0,
            "submit_time": 3400, "executed_arrival_time": 3400,
            "vc_name": "vc1", "hp_cohort_eligible": True
        }
    ))
    seq += 1

    events.append(make_event(
        seq, 3500, "arrival", spec, job_id=5,
        details={
            "job_type": "HP", "worker_num": 2, "gpu_request": 1.0,
            "submit_time": 3500, "executed_arrival_time": 3500,
            "vc_name": "vc1", "hp_cohort_eligible": True
        }
    ))
    seq += 1

    events.append(make_event(
        seq, 3600, "arrival", spec, job_id=6,
        details={
            "job_type": "HP", "worker_num": 2, "gpu_request": 1.0,
            "submit_time": 3600, "executed_arrival_time": 3600,
            "vc_name": "vc1", "hp_cohort_eligible": True
        }
    ))
    seq += 1

    # Followup / Deadline events in chronological order:
    # HP 3 starts at 6899 (before deadline 6900)
    events.append(make_event(
        seq, 6899, "start", spec, job_id=3,
        details={
            "job_type": "HP", "requested_worker_count": 2, "actual_worker_count": 2,
            "complete_start": True, "allocations": [{"node_name": "n1", "card_ids": [0, 1]}]
        }
    ))
    seq += 1
    events.append(make_event(
        seq, 6899, "deadline_outcome", spec, job_id=3,
        details={
            "arrival_time": 3300, "deadline": 6900, "outcome": "met",
            "observed_time": 6899, "rule_version": "1.0"
        }
    ))
    seq += 1

    # HP 4 starts at 7000 (exactly at deadline 7000)
    events.append(make_event(
        seq, 7000, "start", spec, job_id=4,
        details={
            "job_type": "HP", "requested_worker_count": 2, "actual_worker_count": 2,
            "complete_start": True, "allocations": [{"node_name": "n1", "card_ids": [2, 3]}]
        }
    ))
    seq += 1
    events.append(make_event(
        seq, 7000, "deadline_outcome", spec, job_id=4,
        details={
            "arrival_time": 3400, "deadline": 7000, "outcome": "met",
            "observed_time": 7000, "rule_version": "1.0"
        }
    ))
    seq += 1

    # HP 5 deadline miss at 7100, then starts at 7101
    events.append(make_event(
        seq, 7100, "hp_deadline", spec, job_id=5,
        details={
            "arrival_time": 3500, "deadline": 7100, "status": "unstarted_past_deadline",
            "wait_seconds": 3600
        }
    ))
    seq += 1
    events.append(make_event(
        seq, 7100, "deadline_outcome", spec, job_id=5,
        details={
            "arrival_time": 3500, "deadline": 7100, "outcome": "missed",
            "observed_time": 7100, "rule_version": "1.0"
        }
    ))
    seq += 1
    events.append(make_event(
        seq, 7101, "start", spec, job_id=5,
        details={
            "job_type": "HP", "requested_worker_count": 2, "actual_worker_count": 2,
            "complete_start": True, "allocations": [{"node_name": "n1", "card_ids": [0, 1]}]
        }
    ))
    seq += 1

    # HP 6 deadline miss at 7200, never starts
    events.append(make_event(
        seq, 7200, "deadline_outcome", spec, job_id=6,
        details={
            "arrival_time": 3600, "deadline": 7200, "outcome": "missed",
            "observed_time": 7200, "rule_version": "1.0"
        }
    ))
    seq += 1

    # Terminal event past all deadlines
    events.append(make_event(
        seq, 9600, "terminal", spec,
        details={
            "time": 9600, "terminal_reason": "planned_horizon_reached",
            "remaining_queued": 1, "remaining_running": 0,
            "last_event_seq": seq, "status": "completed"
        }
    ))

    reduced = reduce_replay_events(events, spec)
    assert reduced["hp_gang_cohort_count"] == 4  # jobs 3, 4, 5, 6
    assert reduced["hp_gang_violation_count"] == 2  # jobs 5 and 6
    assert reduced["hp_gang_censored_count"] == 0
    assert reduced["hp_gang_violation_rate"] == 0.5
    assert reduced["hp_gang_waits_observed"] == 3  # jobs 3, 4, 5 started


def test_hp_censored_external_truncation(base_spec):
    spec = base_spec
    events = []
    seq = 0

    # Eligible gang arrives at t=4000 (deadline 7600)
    events.append(make_event(
        seq, 4000, "arrival", spec, job_id=10,
        details={
            "job_type": "HP", "worker_num": 2, "gpu_request": 1.0,
            "submit_time": 4000, "executed_arrival_time": 4000,
            "vc_name": "vc1", "hp_cohort_eligible": True
        }
    ))
    seq += 1

    # External truncation occurs at t=5000 (< 7600) while job 10 is still pending
    events.append(make_event(
        seq, 5000, "deadline_outcome", spec, job_id=10,
        details={
            "arrival_time": 4000, "deadline": 7600, "outcome": "censored",
            "observed_time": 5000, "rule_version": "1.0"
        }
    ))
    seq += 1

    events.append(make_event(
        seq, 5000, "terminal", spec,
        details={
            "time": 5000, "terminal_reason": "externally_truncated",
            "remaining_queued": 1, "remaining_running": 0,
            "last_event_seq": seq, "status": "externally_truncated"
        }
    ))

    reduced = reduce_replay_events(events, spec)
    assert reduced["hp_gang_cohort_count"] == 1
    assert reduced["hp_gang_censored_count"] == 1
    assert reduced["hp_gang_violation_rate"] is None  # Must be null if censored > 0


def test_eligible_hp_job_requires_emitted_outcome(base_spec):
    spec = base_spec
    events = [make_event(0, 3300, "arrival", spec, job_id=31, details={
        "job_type": "HP", "worker_num": 2, "gpu_request": 1.0,
        "submit_time": 3300, "executed_arrival_time": 3300,
        "vc_name": "vc1", "hp_cohort_eligible": True,
    })]
    events.append(make_event(1, 9600, "terminal", spec, details={
        "time": 9600, "terminal_reason": "planned_horizon_reached",
        "remaining_queued": 1, "remaining_running": 0,
        "last_event_seq": 1, "status": "completed",
    }))
    with pytest.raises(ValueError, match="no deadline_outcome"):
        reduce_replay_events(events, spec)


def test_eviction_and_lost_work(base_spec):
    spec = base_spec
    events = []
    seq = 0

    # Start spot job 201 (attempt 0)
    events.append(make_event(
        seq, 3100, "start", spec, job_id=201, attempt_index=0,
        details={
            "job_type": "Spot", "requested_worker_count": 2, "actual_worker_count": 2,
            "complete_start": True, "allocations": [{"node_name": "n1", "card_ids": [0, 1]}]
        }
    ))
    seq += 1

    # Preempt job 201 at t=3500: rollback=400s, 2 workers, 1 gpu -> 800 lost gpu seconds
    events.append(make_event(
        seq, 3500, "preemption", spec, job_id=201, attempt_index=0,
        details={
            "attempt_index": 0, "start_time": 3100, "worker_num": 2, "gpu_request": 1.0,
            "rollback_runtime_seconds": 400, "lost_gpu_seconds": 800.0,
            "remain_before": 2000, "remain_after": 2400, "overhead": 0, "in_score_window": True
        }
    ))
    seq += 1

    # Restart job 201 (attempt 1)
    events.append(make_event(
        seq, 4000, "start", spec, job_id=201, attempt_index=1,
        details={
            "job_type": "Spot", "requested_worker_count": 2, "actual_worker_count": 2,
            "complete_start": True, "allocations": [{"node_name": "n1", "card_ids": [0, 1]}]
        }
    ))
    seq += 1

    # Preempt job 201 again at t=4200: rollback=200s, 2 workers, 1 gpu -> 400 lost gpu seconds
    events.append(make_event(
        seq, 4200, "preemption", spec, job_id=201, attempt_index=1,
        details={
            "attempt_index": 1, "start_time": 4000, "worker_num": 2, "gpu_request": 1.0,
            "rollback_runtime_seconds": 200, "lost_gpu_seconds": 400.0,
            "remain_before": 2400, "remain_after": 2600, "overhead": 0, "in_score_window": True
        }
    ))
    seq += 1

    # Separate Spot job starts and is preempted in followup.
    events.append(make_event(
        seq, 6000, "start", spec, job_id=202, attempt_index=0,
        details={
            "job_type": "Spot", "requested_worker_count": 1, "actual_worker_count": 1,
            "complete_start": True, "allocations": [{"node_name": "n1", "card_ids": [2]}]
        }
    ))
    seq += 1

    # Preemption in followup (t=6500)
    events.append(make_event(
        seq, 6500, "preemption", spec, job_id=202, attempt_index=0,
        details={
            "attempt_index": 0, "start_time": 6000, "worker_num": 1, "gpu_request": 1.0,
            "rollback_runtime_seconds": 500, "lost_gpu_seconds": 500.0,
            "remain_before": 1000, "remain_after": 1500, "overhead": 0, "in_score_window": False
        }
    ))
    seq += 1

    events.append(make_event(
        seq, 7000, "terminal", spec,
        details={
            "time": 7000, "terminal_reason": "planned_horizon_reached",
            "remaining_queued": 1, "remaining_running": 0,
            "last_event_seq": seq, "status": "completed"
        }
    ))

    reduced = reduce_replay_events(events, spec)
    assert reduced["spot_eviction_events_score"] == 2
    assert reduced["unique_spot_jobs_evicted_score"] == 1  # Only job 201
    assert reduced["spot_lost_gpu_seconds_score"] == 1200.0

    assert reduced["spot_eviction_events_all_phase"] == 3
    assert reduced["unique_spot_jobs_evicted_all_phase"] == 2  # jobs 201 and 202
    assert reduced["spot_lost_gpu_seconds_all_phase"] == 1700.0


def test_malformed_event_streams_fail_immediately(base_spec):
    spec = base_spec

    # 1. Missing required top-level field
    ev_bad = make_event(0, 0, "terminal", spec, details={
        "time": 0, "terminal_reason": "completed", "remaining_queued": 0,
        "remaining_running": 0, "last_event_seq": 0, "status": "completed"
    })
    del ev_bad["schema_version"]
    with pytest.raises(ValueError, match="missing required top-level field"):
        reduce_replay_events([ev_bad], spec)

    # 2. Sequence gap
    ev0 = make_event(0, 0, "arrival", spec, details={
        "job_type": "Spot", "worker_num": 1, "gpu_request": 1.0,
        "submit_time": 0, "executed_arrival_time": 0, "vc_name": "vc1", "hp_cohort_eligible": False
    })
    ev2 = make_event(2, 10, "arrival", spec, details={
        "job_type": "Spot", "worker_num": 1, "gpu_request": 1.0,
        "submit_time": 10, "executed_arrival_time": 10, "vc_name": "vc1", "hp_cohort_eligible": False
    })
    with pytest.raises(ValueError, match="gap or duplicate"):
        reduce_replay_events([ev0, ev2], spec)

    # 3. Decreasing replay time
    ev1_time_back = make_event(1, 0, "arrival", spec, details={
        "job_type": "Spot", "worker_num": 1, "gpu_request": 1.0,
        "submit_time": 0, "executed_arrival_time": 0, "vc_name": "vc1", "hp_cohort_eligible": False
    })
    ev0_time5 = make_event(0, 5, "arrival", spec, details={
        "job_type": "Spot", "worker_num": 1, "gpu_request": 1.0,
        "submit_time": 5, "executed_arrival_time": 5, "vc_name": "vc1", "hp_cohort_eligible": False
    })
    with pytest.raises(ValueError, match="Decreasing replay time"):
        reduce_replay_events([ev0_time5, ev1_time_back], spec)

    # 4. Duplicate positive credit key
    c_start = make_event(0, 3000, "start", spec, job_id=99, attempt_index=0, details={
        "job_type": "Spot", "requested_worker_count": 1, "actual_worker_count": 1,
        "complete_start": True, "allocations": [{"node_name": "n1", "card_ids": [0]}]
    })
    c1 = make_event(1, 3100, "credit", spec, job_id=99, details={
        "reason": "checkpoint", "attempt_index": 0, "credit_sequence": 0,
        "credited_run_seconds": 3600, "gpu_equivalents": 1.0, "credited_gpu_seconds": 3600.0,
        "in_score_window_research": True, "in_score_window_source": True
    })
    c2 = make_event(2, 3200, "credit", spec, job_id=99, details={
        "reason": "checkpoint", "attempt_index": 0, "credit_sequence": 0,  # Duplicate (99, 0, 0)
        "credited_run_seconds": 100, "gpu_equivalents": 1.0, "credited_gpu_seconds": 100.0,
        "in_score_window_research": True, "in_score_window_source": True
    })
    with pytest.raises(ValueError, match="Duplicate positive credit key"):
        reduce_replay_events([c_start, c1, c2], spec)

    # 5. Non-finite float (NaN)
    c_nan = make_event(0, 3100, "credit", spec, job_id=99, details={
        "reason": "checkpoint", "attempt_index": 0, "credit_sequence": 0,
        "credited_run_seconds": 3600, "gpu_equivalents": 1.0, "credited_gpu_seconds": float("nan"),
        "in_score_window_research": True, "in_score_window_source": True
    })
    with pytest.raises(ValueError, match="Non-finite float"):
        reduce_replay_events([c_nan], spec)


def test_reconciliation_audit(base_spec):
    online = {
        "schema_version": "2.0",
        "run_id": "run1",
        "episode_id": "episode1",
        "status": "completed",
        "terminal_reason": "planned_horizon_reached",
        "hp_gang_cohort_count": 5,
        "hp_gang_violation_count": 1,
        "hp_gang_censored_count": 0,
        "hp_gang_violation_rate": 0.2,
        "spot_eviction_events_score": 3,
        "unique_spot_jobs_evicted_score": 2,
        "spot_eviction_events_all_phase": 4,
        "unique_spot_jobs_evicted_all_phase": 3,
        "solver_calls_score": 10,
        "solver_calls_all_phase": 12,
        "quota_windows_score": 10,
        "calls_per_window": 1.0,
        "decision_count": 4,
        "completed_hp_jobs": 5,
        "completed_spot_jobs": 8,
        "remaining_queued_jobs": 1,
        "remaining_running_jobs": 2,
        "useful_spot_gpu_seconds_score": 36000.0,
        "useful_spot_gpu_hours_score_raw": 10.0,
        "useful_spot_gpu_seconds_all_phase": 40000.0,
        "useful_spot_gpu_hours_all_phase_raw": 40000.0 / 3600.0,
        "spot_lost_gpu_seconds_score": 1200.0,
        "spot_lost_gpu_seconds_all_phase": 1400.0,
    }

    reduced_matching = dict(online)
    passed, mismatch = reconcile_metrics(online, reduced_matching)
    assert passed is True
    assert mismatch is None

    # Mismatch in integer count
    reduced_diff_count = dict(online)
    reduced_diff_count["spot_eviction_events_score"] = 4
    passed, mismatch = reconcile_metrics(online, reduced_diff_count)
    assert passed is False
    assert mismatch["field"] == "spot_eviction_events_score"

    # Mismatch in float outside tolerance
    reduced_diff_float = dict(online)
    reduced_diff_float["useful_spot_gpu_seconds_score"] = 36000.5
    passed, mismatch = reconcile_metrics(online, reduced_diff_float)
    assert passed is False
    assert mismatch["field"] == "useful_spot_gpu_seconds_score"

    reduced_bad_identity = dict(online)
    reduced_bad_identity["run_id"] = "other_run"
    passed, mismatch = reconcile_metrics(online, reduced_bad_identity)
    assert passed is False
    assert mismatch["field"] == "run_id"
