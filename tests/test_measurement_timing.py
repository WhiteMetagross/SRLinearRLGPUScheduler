import pytest
import pandas as pd
from unittest.mock import MagicMock

from research_sim.types import EpisodeSpec
from research_sim.trace import EpisodeTraceView
from research_sim.core import ResearchSimulator


class FakeClock:
    def __init__(self, start_ns: int = 1_000_000_000, step_ns: int = 5_000_000):
        self.current_ns = start_ns
        self.step_ns = step_ns

    def __call__(self) -> int:
        val = self.current_ns
        self.current_ns += self.step_ns
        return val


def test_injected_timer_spans_and_order():
    node_data = [
        {"node_name": 0, "gpu_model": "A100-SXM4-80GB", "vc_name": "A100-SXM4-80GB", "gpu_capacity_num": 8, "cpu_num": 128}
    ]
    node_df = pd.DataFrame(node_data)
    job_data = [
        {"job_name": 1, "organization": 0, "gpu_model": "A100-SXM4-80GB", "worker_num": 1, "gpu_request": 4.0, "cpu_request": 16.0, "submit_time": 10.0, "duration": 50.0, "job_type": "Spot"}
    ]
    job_df = pd.DataFrame(job_data)

    spec = EpisodeSpec(
        score_start=0,
        score_end=100,
        followup_end=3700,
        L_H=100,
        trace_mode="research_full_history",
        forecast_mode="causal_capacity"
    )
    view = EpisodeTraceView(node_df, job_df, spec, {"node_sha256": "d", "job_sha256": "d"})

    sim = ResearchSimulator(spec=spec, view=view, policy_type="decision_boundary")
    clock = FakeClock(start_ns=1_000_000_000, step_ns=10_000_000)  # 10ms per tick
    sim.timer_func = clock

    ctx, delta = sim.advance_until_decision()
    assert ctx is not None

    # Candidate generation occurred before pause
    assert sim._last_candidate_start_ns is not None
    assert sim._last_candidate_end_ns is not None
    assert sim._decision_paused_time_ns is not None
    assert sim._last_candidate_start_ns < sim._last_candidate_end_ns
    assert sim._last_candidate_end_ns <= sim._decision_paused_time_ns

    # Apply decision
    sim.apply((0, 0))

    assert len(sim.decision_timing_records) == 1
    rec = sim.decision_timing_records[0]
    assert rec["decision_id"] == 0
    assert rec["selected_slot"] == 0
    assert rec["verify_flag"] == 0
    assert rec["candidate_start_ns"] < rec["candidate_end_ns"]
    assert rec["response_start_ns"] < rec["response_end_ns"]
    assert rec["candidate_generation_ms"] > 0
    assert rec["policy_response_ms"] > 0
    assert rec["decision_pipeline_ms"] >= rec["policy_response_ms"]


def test_native_run_zero_decisions_null_timing():
    node_data = [
        {"node_name": 0, "gpu_model": "A100-SXM4-80GB", "vc_name": "A100-SXM4-80GB", "gpu_capacity_num": 8, "cpu_num": 128}
    ]
    node_df = pd.DataFrame(node_data)
    job_data = [
        {"job_name": 1, "organization": 0, "gpu_model": "A100-SXM4-80GB", "worker_num": 1, "gpu_request": 4.0, "cpu_request": 16.0, "submit_time": 10.0, "duration": 50.0, "job_type": "HP"}
    ]
    job_df = pd.DataFrame(job_data)

    spec = EpisodeSpec(
        score_start=0,
        score_end=100,
        followup_end=3700,
        L_H=100,
        trace_mode="research_full_history",
        forecast_mode="causal_capacity"
    )
    view = EpisodeTraceView(node_df, job_df, spec, {"node_sha256": "d", "job_sha256": "d"})

    sim = ResearchSimulator(spec=spec, view=view, policy_type="native_gfs")
    while True:
        ctx, delta = sim.advance_until_decision()
        if delta.terminated or delta.truncated:
            break

    # Native GFS runs have 0 paused RL decisions
    assert len(sim.decision_timing_records) == 0
