from types import SimpleNamespace

import pytest

from benchmark.publication_metrics import derive_publication_metrics


def _event(seq, time, event_type, job_id=None, **details):
    return {
        "event_seq": seq,
        "time": time,
        "event_type": event_type,
        "job_id": job_id,
        "details": details,
    }


def test_publication_metrics_integrate_queue_occupancy_and_censoring():
    spec = SimpleNamespace(
        score_start=100,
        score_end=3700,
        followup_end=7300,
        calendar_anchor="2024-03-01 00:00:00",
    )
    events = [
        _event(0, 100, "arrival", 1, job_type="Spot", worker_num=2,
               gpu_request=1.0, executed_arrival_time=100, hp_cohort_eligible=False),
        _event(1, 200, "start", 1, complete_start=True),
        _event(2, 1200, "credit", 1, credited_gpu_seconds=2000.0),
        _event(3, 1200, "completion", 1),
        _event(4, 2000, "arrival", 2, job_type="HP", worker_num=2,
               gpu_request=1.0, executed_arrival_time=2000, hp_cohort_eligible=True),
        _event(5, 3000, "start", 2, complete_start=True),
        _event(6, 3000, "deadline_outcome", 2, outcome="met"),
        _event(7, 7300, "terminal"),
    ]
    metrics = derive_publication_metrics(events, spec, total_capacity_gpu=10)
    assert metrics["spot_queued_gpu_hours"] == pytest.approx(200.0 / 3600.0)
    assert metrics["spot_allocated_gpu_hours"] == pytest.approx(2000.0 / 3600.0)
    assert metrics["useful_spot_gpu_hours"] == pytest.approx(2000.0 / 3600.0)
    assert metrics["lost_spot_gpu_hours"] == 0.0
    assert metrics["spot_evictions"] == 0
    assert metrics["hp_queued_gpu_hours"] == pytest.approx(2000.0 / 3600.0)
    assert metrics["hp_allocated_gpu_hours"] == pytest.approx(1400.0 / 3600.0)
    assert metrics["cohorts"]["spot"]["observed_completion_rate"] == 1.0
    assert metrics["cohorts"]["spot"]["completed_only_jct_seconds"]["mean"] == 1100.0
    assert metrics["cohorts"]["hp"]["completion_within_1h_rate"] == 0.0
    assert metrics["cohorts"]["hp"]["restricted_1h_time_to_completion_seconds"]["mean"] == 3600.0
    assert metrics["cohorts"]["hp"]["restricted_1h_eligible_arrivals"] == 1
    assert metrics["observation_complete"] is True
    assert metrics["daily"][0]["hp_gang_cohort"] == 1
    assert metrics["daily"][0]["hp_gang_violations"] == 0
    assert metrics["hp_gang_cohort_count"] == 1
    assert metrics["hp_gang_violation_count"] == 0
    assert metrics["hp_gang_violation_rate"] == 0.0


def test_publication_metrics_include_requeue_after_preemption():
    spec = SimpleNamespace(
        score_start=0,
        score_end=3600,
        followup_end=7200,
        calendar_anchor="2024-03-01 00:00:00",
    )
    events = [
        _event(0, 0, "arrival", 1, job_type="Spot", worker_num=1,
               gpu_request=1.0, executed_arrival_time=0, hp_cohort_eligible=False),
        _event(1, 100, "start", 1, complete_start=True),
        _event(2, 200, "preemption", 1, lost_gpu_seconds=50.0),
        _event(3, 400, "start", 1, complete_start=True),
        _event(4, 500, "completion", 1),
        _event(5, 3600, "terminal"),
    ]
    metrics = derive_publication_metrics(events, spec, total_capacity_gpu=1)
    assert metrics["spot_queued_gpu_hours"] == pytest.approx(300.0 / 3600.0)
    assert metrics["spot_allocated_gpu_hours"] == pytest.approx(200.0 / 3600.0)
    total_queue = metrics["cohorts"]["spot"]["completed_only_total_queue_seconds"]
    assert total_queue["mean"] == 300.0
    assert metrics["daily"][0]["spot_evictions"] == 1
    assert metrics["daily"][0]["lost_spot_gpu_hours"] == pytest.approx(50.0 / 3600.0)
    assert metrics["spot_evictions"] == 1
    assert metrics["lost_spot_gpu_hours"] == pytest.approx(50.0 / 3600.0)


def test_publication_metrics_report_lease_turnover_separately_from_eviction():
    spec = SimpleNamespace(
        score_start=0,
        score_end=3600,
        followup_end=7200,
        calendar_anchor="2024-03-01 00:00:00",
    )
    events = [
        _event(0, 0, "arrival", 1, job_type="Spot", worker_num=2,
               gpu_request=1.0, executed_arrival_time=0, hp_cohort_eligible=False),
        _event(1, 0, "start", 1, complete_start=True),
        _event(2, 300, "lease_expiration", 1, overhead_seconds=120),
        _event(3, 300, "start", 1, complete_start=True),
        _event(4, 600, "completion", 1),
        _event(5, 3600, "terminal"),
    ]
    metrics = derive_publication_metrics(events, spec, total_capacity_gpu=2)
    assert metrics["spot_evictions"] == 0
    assert metrics["lost_spot_gpu_hours"] == 0.0
    assert metrics["lease_expirations"] == 1
    assert metrics["spot_lease_expirations"] == 1
    assert metrics["hp_lease_expirations"] == 0
    assert metrics["lease_turnover_overhead_gpu_hours"] == pytest.approx(240.0 / 3600.0)


def test_publication_metrics_do_not_extrapolate_externally_truncated_run():
    spec = SimpleNamespace(
        score_start=0,
        score_end=3600,
        followup_end=7200,
        calendar_anchor="2024-03-01 00:00:00",
    )
    events = [
        _event(0, 0, "arrival", 1, job_type="Spot", worker_num=1,
               gpu_request=1.0, executed_arrival_time=0, hp_cohort_eligible=False),
        _event(1, 100, "start", 1, complete_start=True),
        _event(2, 200, "terminal", status="EXTERNALLY_TRUNCATED"),
    ]
    metrics = derive_publication_metrics(events, spec, total_capacity_gpu=2)
    assert metrics["observation_complete"] is False
    assert metrics["score_hours"] == pytest.approx(200.0 / 3600.0)
    assert metrics["spot_allocated_gpu_hours"] == pytest.approx(100.0 / 3600.0)
    assert metrics["idle_gpu_hours"] == pytest.approx(300.0 / 3600.0)
    assert metrics["cohorts"]["spot"]["restricted_1h_eligible_arrivals"] == 0
    assert metrics["cohorts"]["spot"]["restricted_1h_censored_arrivals"] == 1
    assert metrics["cohorts"]["spot"]["completion_within_1h_rate"] is None
