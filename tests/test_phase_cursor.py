import pytest
import pandas as pd
from types import SimpleNamespace
from research_sim.types import (
    EpisodeSpec,
    Phase,
    InvalidActionError,
    ResearchSimError,
    Placement,
    WorkerAllocation
)
from research_sim.trace import EpisodeTraceView
from research_sim.core import ResearchSimulator
from research_sim.verifier import NoCheckVerifier


def create_mock_trace_view(num_nodes=2, gpus_per_node=8):
    """Creates small deterministic synthetic trace view for unit tests."""
    node_data = []
    for i in range(num_nodes):
        node_data.append({
            "node_name": i,
            "gpu_model": "A100-SXM4-80GB",
            "vc_name": "A100-SXM4-80GB",
            "gpu_capacity_num": gpus_per_node,
            "cpu_num": 128
        })
    node_df = pd.DataFrame(node_data)

    # 3 jobs: 1 HP arrival at t=0, 2 Spot arrivals at t=0
    job_data = [
        {
            "job_name": 101,
            "organization": 0,
            "gpu_model": "A100-SXM4-80GB",
            "worker_num": 1,
            "gpu_request": 2.0,
            "cpu_request": 8.0,
            "submit_time": 0.0,
            "duration": 100.0,
            "job_type": "HP"
        },
        {
            "job_name": 102,
            "organization": 0,
            "gpu_model": "A100-SXM4-80GB",
            "worker_num": 1,
            "gpu_request": 2.0,
            "cpu_request": 8.0,
            "submit_time": 0.0,
            "duration": 50.0,
            "job_type": "Spot"
        },
        {
            "job_name": 103,
            "organization": 0,
            "gpu_model": "A100-SXM4-80GB",
            "worker_num": 1,
            "gpu_request": 2.0,
            "cpu_request": 8.0,
            "submit_time": 0.0,
            "duration": 50.0,
            "job_type": "Spot"
        }
    ]
    job_df = pd.DataFrame(job_data)

    spec = EpisodeSpec(
        node_filter="A100-SXM4-80GB",
        warmup_start=0,
        score_start=0,
        score_end=300,
        followup_end=3900,
        trace_mode="research_full_history",
        forecast_mode="causal_capacity"
    )

    hashes = {"node_sha256": "dummy_node", "job_sha256": "dummy_job"}
    return EpisodeTraceView(node_df, job_df, spec, hashes), spec


def test_pausable_decision_and_atomic_resume():
    view, spec = create_mock_trace_view()
    sim = ResearchSimulator(
        spec=spec,
        view=view,
        policy_type="decision_boundary",
        verifier=NoCheckVerifier(),
        log_dir="/tmp/test_sim_logs"
    )

    # Advance until first decision
    context1, delta1 = sim.advance_until_decision()
    assert context1 is not None
    assert context1.time == 0
    assert context1.cursor.phase == Phase.SPOT_SCAN
    assert context1.job_id == 102  # First Spot job
    assert len(context1.candidates.candidates) >= 1

    # Apply unverified first placement candidate (action = 2 * 1 + 0 = 2)
    evs1 = sim.apply(2)
    assert len(evs1) >= 1
    assert any(ev.event_type == "placement" for ev in evs1)

    # Next call to advance_until_decision must NOT re-run arrivals or HP scan!
    # It must advance directly to the next Spot job (103) in the same second t=0
    context2, delta2 = sim.advance_until_decision()
    assert context2 is not None
    assert context2.time == 0
    assert context2.job_id == 103
    assert context2.cursor.phase == Phase.SPOT_SCAN

    # Second decision sees updated cluster (2 cards occupied by job 102)
    obs2 = sim.observe()
    assert obs2["action_mask"][0] == True  # deferral is always legal

    # Defer job 103 (action = 0)
    evs2 = sim.apply(0)
    assert any(ev.event_type == "deferral" for ev in evs2)

    # Advance until horizon
    context3, delta3 = sim.advance_until_decision()
    # The next arrival, completion, or quota change may reoffer the task.
    assert context3 is not None
    assert context3.time > 0
    assert context3.job_id == 103


def test_diagnostic_time_limit_keeps_simulator_resumable():
    view, spec = create_mock_trace_view()
    sim = ResearchSimulator(
        spec=spec, view=view, policy_type="decision_boundary",
        verifier=NoCheckVerifier(), log_dir="/tmp/test_time_limit",
    )
    for _ in range(2):
        context, _ = sim.advance_until_decision()
        assert context is not None
        sim.apply(0)
    context, delta = sim.advance_until_decision(stop_at_time=5)
    assert context is None
    assert delta.terminal_reason == "diagnostic_time_limit"
    assert sim.time == 6
    assert not sim._is_terminated
    context, delta = sim.advance_until_decision(stop_at_time=10)
    assert context is None
    assert delta.terminal_reason == "diagnostic_time_limit"
    assert sim.time == 11


def test_stale_or_invalid_action_rejection():
    view, spec = create_mock_trace_view()
    sim = ResearchSimulator(
        spec=spec,
        view=view,
        policy_type="decision_boundary",
        verifier=NoCheckVerifier(),
        log_dir="/tmp/test_sim_logs"
    )

    context, delta = sim.advance_until_decision()
    assert context is not None

    # Invalid action out of candidate slot range
    with pytest.raises(InvalidActionError):
        sim.apply(999)

    before_events = len(sim.ledger.events)
    before_remaining = sim.budget.remaining_calls
    with pytest.raises(InvalidActionError):
        sim.apply(3)  # Verification is illegal without a ready verifier.
    assert len(sim.ledger.events) == before_events
    assert sim.budget.remaining_calls == before_remaining

    # Apply valid action
    sim.apply(0)  # defer

    # Attempting to apply again without advancing raises ResearchSimError
    with pytest.raises(ResearchSimError):
        sim.apply(0)


def test_native_forecast_failure_is_not_silently_replaced():
    view, base = create_mock_trace_view()
    spec = EpisodeSpec(
        score_start=0,
        score_end=300,
        followup_end=3900,
        forecast_mode="native_online",
    )
    sim = ResearchSimulator(
        spec=spec, view=view, policy_type="native_gfs", log_dir="/tmp/test_native_failure"
    )
    with pytest.raises(ResearchSimError, match="no recorder history"):
        sim.advance_until_decision()
    assert not sim.quota_ready
