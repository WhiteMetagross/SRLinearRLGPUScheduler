import pytest
import pandas as pd
from research_sim.types import EpisodeSpec, Phase
from research_sim.trace import EpisodeTraceView
from research_sim.core import ResearchSimulator


def create_preemption_fixture():
    # 1 node with 4 GPUs
    node_df = pd.DataFrame([{
        "node_name": 0,
        "gpu_model": "A100-SXM4-80GB",
        "vc_name": "A100-SXM4-80GB",
        "gpu_capacity_num": 4,
        "cpu_num": 128
    }])

    # t=0: Spot job takes all 4 GPUs
    # t=10: HP job arrives requiring 4 GPUs -> must preempt Spot job!
    job_df = pd.DataFrame([
        {
            "job_name": 1,
            "organization": 0,
            "gpu_model": "A100-SXM4-80GB",
            "worker_num": 1,
            "gpu_request": 4.0,
            "cpu_request": 8.0,
            "submit_time": 0.0,
            "duration": 100.0,
            "job_type": "Spot"
        },
        {
            "job_name": 2,
            "organization": 0,
            "gpu_model": "A100-SXM4-80GB",
            "worker_num": 1,
            "gpu_request": 4.0,
            "cpu_request": 8.0,
            "submit_time": 10.0,
            "duration": 50.0,
            "job_type": "HP"
        }
    ])

    spec = EpisodeSpec(
        node_filter="A100-SXM4-80GB",
        warmup_start=0,
        score_start=0,
        score_end=300,
        followup_end=3900,
        trace_mode="research_full_history",
        forecast_mode="causal_capacity"
    )

    hashes = {"node_sha256": "dummy", "job_sha256": "dummy"}
    return EpisodeTraceView(node_df, job_df, spec, hashes), spec


def test_hp_preempts_spot_when_capacity_exhausted():
    view, spec = create_preemption_fixture()
    sim = ResearchSimulator(
        spec=spec,
        view=view,
        policy_type="native_gfs",
        log_dir="/tmp/test_preempt_logs"
    )

    while True:
        context, delta = sim.advance_until_decision()
        if delta.terminated or delta.truncated:
            break

    # Check metrics and events
    metrics = sim.get_metrics()
    assert metrics.spot_evictions_count >= 1
    preempt_evs = [ev for ev in sim.ledger.events if ev.event_type == "preemption"]
    assert len(preempt_evs) >= 1
    assert preempt_evs[0].job_id == 1  # Spot job 1 was preempted
    assert preempt_evs[0].time == 10   # At t=10 when HP job 2 arrived


def test_fractional_spot_placed_on_native_path():
    # 1 node with 4 GPUs
    node_df = pd.DataFrame([{
        "node_name": 0,
        "gpu_model": "A100-SXM4-80GB",
        "vc_name": "A100-SXM4-80GB",
        "gpu_capacity_num": 4,
        "cpu_num": 128
    }])

    # Fractional spot job (0.5 GPU)
    job_df = pd.DataFrame([
        {
            "job_name": 50,
            "organization": 0,
            "gpu_model": "A100-SXM4-80GB",
            "worker_num": 1,
            "gpu_request": 0.5,
            "cpu_request": 4.0,
            "submit_time": 0.0,
            "duration": 30.0,
            "job_type": "Spot"
        }
    ])

    spec = EpisodeSpec(
        node_filter="A100-SXM4-80GB",
        warmup_start=0,
        score_start=0,
        score_end=300,
        followup_end=3900,
        trace_mode="research_full_history",
        forecast_mode="causal_capacity"
    )

    hashes = {"node_sha256": "dummy", "job_sha256": "dummy"}
    view = EpisodeTraceView(node_df, job_df, spec, hashes)

    # In decision_boundary mode, fractional tasks bypass the pausable RL surface
    # and get placed directly on native path without pausing
    sim = ResearchSimulator(
        spec=spec,
        view=view,
        policy_type="decision_boundary",
        log_dir="/tmp/test_frac_logs"
    )

    context, delta = sim.advance_until_decision()
    # Should run to completion without pausing for fractional tasks
    assert context is None
    assert delta.terminated or delta.truncated
    assert sim.ledger.completed_jobs["Spot"] == 1


def test_research_gang_preemption_places_every_worker():
    node_df = pd.DataFrame([{
        "node_name": 0, "gpu_model": "A100-SXM4-80GB",
        "vc_name": "A100-SXM4-80GB", "gpu_capacity_num": 4, "cpu_num": 128,
    }])
    job_df = pd.DataFrame([
        {"job_name": 1, "organization": 0, "gpu_model": "A100-SXM4-80GB",
         "worker_num": 1, "gpu_request": 4.0, "cpu_request": 8.0,
         "submit_time": 0.0, "duration": 100.0, "job_type": "Spot"},
        {"job_name": 2, "organization": 0, "gpu_model": "A100-SXM4-80GB",
         "worker_num": 2, "gpu_request": 2.0, "cpu_request": 8.0,
         "submit_time": 10.0, "duration": 20.0, "job_type": "HP"},
    ])
    spec = EpisodeSpec(score_start=1000, score_end=1100, followup_end=4700,
                       trace_mode="research_full_history", forecast_mode="causal_capacity")
    view = EpisodeTraceView(node_df, job_df, spec, {"node_sha256": "fixture", "job_sha256": "fixture"})
    sim = ResearchSimulator(spec, view, "native_gfs", log_dir="/tmp/test_hp_gang_preempt")
    while True:
        _, delta = sim.advance_until_decision()
        if delta.terminated or delta.truncated:
            break
    hp_start = next(e for e in sim.ledger.events if e.event_type == "start" and e.job_id == 2)
    assert hp_start.time == 10
    assert len(hp_start.details["nodes"]) == 2
    cards = [card for entry in hp_start.details["nodes"] for gpu_list in entry.values() for card in gpu_list]
    assert sorted(cards) == [0, 1, 2, 3]
    assert any(e.event_type == "preemption" and e.job_id == 1 for e in sim.ledger.events)


def test_research_gang_infeasible_attempt_does_not_allocate_partial_workers():
    node_df = pd.DataFrame([{
        "node_name": 0, "gpu_model": "A100-SXM4-80GB",
        "vc_name": "A100-SXM4-80GB", "gpu_capacity_num": 4, "cpu_num": 128,
    }])
    job_df = pd.DataFrame([{
        "job_name": 2, "organization": 0, "gpu_model": "A100-SXM4-80GB",
        "worker_num": 3, "gpu_request": 2.0, "cpu_request": 8.0,
        "submit_time": 0.0, "duration": 20.0, "job_type": "HP",
    }])
    spec = EpisodeSpec(score_start=1, score_end=100, followup_end=3700,
                       trace_mode="research_full_history", forecast_mode="causal_capacity")
    view = EpisodeTraceView(node_df, job_df, spec, {"node_sha256": "fixture", "job_sha256": "fixture"})
    sim = ResearchSimulator(spec, view, "native_gfs", log_dir="/tmp/test_hp_gang_infeasible")
    _, delta = sim.advance_until_decision()
    assert delta.terminated
    assert not any(e.event_type == "start" for e in sim.ledger.events)
    node = sim.cluster.vc_dict["A100-SXM4-80GB"].node_list[0]
    assert node.free_gpus == 4
