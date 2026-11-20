import pandas as pd

from research_sim.core import ResearchSimulator
from research_sim.trace import EpisodeTraceView
from research_sim.types import EpisodeSpec


def _sim(mode: str) -> ResearchSimulator:
    node_df = pd.DataFrame([{
        "node_name": 0,
        "gpu_model": "A100-SXM4-80GB",
        "vc_name": "A100-SXM4-80GB",
        "gpu_capacity_num": 8,
        "cpu_num": 128,
    }])
    job_df = pd.DataFrame([
        {
            "job_name": 1,
            "organization": 0,
            "gpu_model": "A100-SXM4-80GB",
            "worker_num": 3,
            "gpu_request": 2.0,
            "cpu_request": 8.0,
            "submit_time": 0.0,
            "duration": 100.0,
            "job_type": "Spot",
        },
        {
            "job_name": 2,
            "organization": 1,
            "gpu_model": "A100-SXM4-80GB",
            "worker_num": 2,
            "gpu_request": 1.0,
            "cpu_request": 8.0,
            "submit_time": 0.0,
            "duration": 100.0,
            "job_type": "HP",
        },
    ])
    spec = EpisodeSpec(
        score_start=1,
        score_end=10,
        followup_end=3610,
        trace_mode="research_full_history",
        forecast_mode="causal_capacity",
        gpu_accounting_mode=mode,
    )
    view = EpisodeTraceView(
        node_df,
        job_df,
        spec,
        {"node_sha256": "fixture", "job_sha256": "fixture"},
    )
    return ResearchSimulator(spec, view, log_dir="/tmp/test_gpu_demand_accounting")


def test_total_demand_mode_counts_all_workers_at_start_and_preemption():
    sim = _sim("total_demand")
    spot, hp = sim.trace.job_list
    sim.pend_job(spot)
    sim.run_spot(spot)
    assert sim.gpu_request_dict[spot["vc_name"]]["Spot"] == 6.0
    sim.preempt(hp, [spot])
    assert sim.gpu_request_dict[spot["vc_name"]]["Spot"] == 0.0

    sim.pend_job(hp)
    sim.run_guar(hp)
    assert sim.gpu_request_dict[hp["vc_name"]]["HP"] == 2.0
    assert sim.guar_request_dict[hp["org_key"]] == 2.0


def test_source_per_worker_mode_preserves_released_gfs_counter():
    sim = _sim("source_per_worker")
    spot, hp = sim.trace.job_list
    sim.pend_job(spot)
    sim.run_spot(spot)
    assert sim.gpu_request_dict[spot["vc_name"]]["Spot"] == 2.0
    sim.pend_job(hp)
    sim.run_guar(hp)
    assert sim.gpu_request_dict[hp["vc_name"]]["HP"] == 1.0
    assert sim.guar_request_dict[hp["org_key"]] == 1.0
