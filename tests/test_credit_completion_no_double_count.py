"""Completion at a checkpoint must not credit the same work twice."""

import pandas as pd

from research_sim.core import ResearchSimulator
from research_sim.trace import EpisodeTraceView
from research_sim.types import EpisodeSpec


def test_spot_completion_near_checkpoint_is_capped_by_service_demand(tmp_path):
    node_df = pd.DataFrame([{
        "node_name": 0, "gpu_model": "A100-SXM4-80GB", "vc_name": "A100-SXM4-80GB",
        "gpu_capacity_num": 8, "cpu_num": 128,
    }])
    job_df = pd.DataFrame([{
        "job_name": 1, "organization": 0, "gpu_model": "A100-SXM4-80GB",
        "worker_num": 1, "gpu_request": 4.0, "cpu_request": 16.0,
        "submit_time": 0.0, "duration": 3599.0, "job_type": "Spot",
    }])
    spec = EpisodeSpec(
        node_filter="A100-SXM4-80GB", warmup_start=0, score_start=0,
        score_end=3700, followup_end=3701,
        L_H=1, trace_mode="source_compatible", forecast_mode="causal_capacity",
    )
    view = EpisodeTraceView(node_df, job_df, spec, {"node_sha256": "test", "job_sha256": "test"})
    sim = ResearchSimulator(spec=spec, view=view, policy_type="native_gfs", log_dir=str(tmp_path))
    while True:
        _, delta = sim.advance_until_decision()
        if delta.terminated or delta.truncated:
            break
    credits = [event for event in sim.ledger.events if event.event_type == "credit"]
    assert len(credits) == 1
    assert credits[0].details["credited_run_seconds"] == 3599
    assert credits[0].details["cumulative_credited_gpu_seconds"] == 3599 * 4
    assert credits[0].details["job_credit_cap_gpu_seconds"] == 3599 * 4
    assert credits[0].details["credited_gpu_seconds"] == 3599 * 4
