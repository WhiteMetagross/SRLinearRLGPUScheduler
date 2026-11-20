import pytest
import pandas as pd
from research_sim.types import EpisodeSpec, InvalidActionError, StaleStateError, ResearchSimError, Placement, WorkerAllocation
from research_sim.trace import EpisodeTraceView
from research_sim.core import ResearchSimulator
from research_sim.verifier import NoCheckVerifier


def create_sim_fixture():
    node_df = pd.DataFrame([
        {"node_name": 0, "gpu_model": "A100-SXM4-80GB", "vc_name": "A100-SXM4-80GB", "gpu_capacity_num": 8, "cpu_num": 128}
    ])
    job_df = pd.DataFrame([
        {"job_name": 1, "organization": 0, "gpu_model": "A100-SXM4-80GB", "worker_num": 1, "gpu_request": 2.0, "cpu_request": 8.0, "submit_time": 0.0, "duration": 50.0, "job_type": "Spot"}
    ])
    spec = EpisodeSpec(
        node_filter="A100-SXM4-80GB",
        warmup_start=0,
        score_start=0,
        score_end=100,
        followup_end=3700,
        trace_mode="research_full_history",
        forecast_mode="causal_capacity",
        M=4
    )
    hashes = {"node_sha256": "dummy", "job_sha256": "dummy"}
    view = EpisodeTraceView(node_df, job_df, spec, hashes)
    sim = ResearchSimulator(spec=spec, view=view, policy_type="decision_boundary", verifier=NoCheckVerifier(), log_dir="/tmp/test_atomic")
    ctx, _ = sim.advance_until_decision()
    return sim, ctx


def test_complete_placement_commit():
    sim, ctx = create_sim_fixture()
    assert ctx is not None
    init_version = sim.state_version

    # Commit slot 1 (unverified placement, action flat = 2)
    events = sim.apply(2)
    assert any(e.event_type == "placement" for e in events)
    assert sim.state_version == init_version + 1
    # Check that node cards are allocated
    node = sim.cluster.vc_dict["A100-SXM4-80GB"].get_node(0)
    assert len(node.node_gpu_dict[0]) == 1
    assert len(node.node_gpu_dict[1]) == 1


def test_stale_version_rejected_without_state_change():
    sim, ctx = create_sim_fixture()
    init_version = sim.state_version

    # Pass stale state version
    stale_action = {"slot": 1, "verify": 0, "state_version": init_version + 99}
    with pytest.raises(StaleStateError):
        sim.apply(stale_action)

    # State version, node allocations, and ledger must remain unchanged
    assert sim.state_version == init_version
    node = sim.cluster.vc_dict["A100-SXM4-80GB"].get_node(0)
    assert len(node.node_gpu_dict[0]) == 0


def test_invalid_action_out_of_range():
    sim, ctx = create_sim_fixture()
    # M=4 -> max slot is 4. Slot 99 is out of range
    with pytest.raises(InvalidActionError):
        sim.apply(99 * 2)


def test_forced_mid_commit_failure_rolls_back_completely(monkeypatch):
    sim, ctx = create_sim_fixture()
    init_version = sim.state_version
    init_event_count = len(sim.ledger.events)

    # Monkeypatch run_spot to simulate mid-commit failure
    def faulty_run_spot(job):
        raise RuntimeError("Simulated crash in run_spot during commit")

    monkeypatch.setattr(sim, "run_spot", faulty_run_spot)

    with pytest.raises(ResearchSimError):
        sim.apply(2)

    # Verify rollback:
    # 1. State version unchanged
    assert sim.state_version == init_version
    # 2. Node allocations completely rolled back
    node = sim.cluster.vc_dict["A100-SXM4-80GB"].get_node(0)
    for c in range(node.num_gpus):
        assert len(node.node_gpu_dict[c]) == 0
        assert node.node_galloc[c] == 0.0
        assert node.node_type_galloc[c]["Spot"] == 0.0
    # 3. Job status returned to pending
    assert sim.active_scan_job["status"] == "pend"
    assert sim.active_scan_job["nodes"] == []
    # 4. Ledger rolled back (only invalid_action event recorded)
    assert any(e.event_type == "invalid_action" for e in sim.ledger.events)
