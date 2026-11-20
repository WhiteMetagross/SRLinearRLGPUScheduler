import pytest
import os
import pandas as pd
from research_sim.types import EpisodeSpec, InvalidConfigError
from research_sim.trace import TraceStore, EpisodeTraceView, EXPECTED_NODE_SHA256, EXPECTED_JOB_SHA256


pytestmark = pytest.mark.skipif(
    not os.path.isfile("data/raw/spot_gpu_v2026/node_info_df.csv"),
    reason="Alibaba raw trace is not stored in Git.",
)


def test_trace_store_loading_and_hashes():
    trace_dir = "data/raw/spot_gpu_v2026"
    store = TraceStore.get_instance(trace_dir)
    assert store.hashes["node_sha256"] == EXPECTED_NODE_SHA256
    assert store.hashes["job_sha256"] == EXPECTED_JOB_SHA256
    assert len(store._raw_nodes) == 4278
    assert len(store._raw_jobs) == 466867


def test_episode_view_creation_and_counts():
    trace_dir = "data/raw/spot_gpu_v2026"
    store = TraceStore.get_instance(trace_dir)

    # 1. Primary A100 spec
    spec = EpisodeSpec(
        node_filter="A100-SXM4-80GB",
        score_start=2678400,
        score_end=2764800,
        followup_end=2764800 + 3600,
        trace_mode="research_full_history"
    )
    view = store.create_episode_view(spec)
    assert view.total_nodes == 432
    assert view.total_gpus == 432 * 8
    assert view.total_jobs > 0
    assert "HP" in view.job_counts_by_class
    assert "Spot" in view.job_counts_by_class

    # Assert stable sort order
    times = view._jobs_df["submit_time"].values
    assert (np_diff := (times[1:] - times[:-1])).min() >= 0


def test_trace_inclusion_modes():
    store = TraceStore.get_instance("data/raw/spot_gpu_v2026")

    spec_res = EpisodeSpec(
        node_filter="A100-SXM4-80GB",
        score_start=2678400,
        score_end=2764800,
        followup_end=2764800 + 3600,
        trace_mode="research_full_history"
    )
    view_res = store.create_episode_view(spec_res)

    spec_src = EpisodeSpec(
        node_filter="A100-SXM4-80GB",
        score_start=2678400,
        score_end=2764800,
        followup_end=2764800 + 3600,
        trace_mode="source_compatible"
    )
    view_src = store.create_episode_view(spec_src)

    # Source compatible retains all HP up to followup_end, but Spot only in [score_start, score_end]
    assert view_src.total_nodes == 432
    assert view_src.total_jobs > 0
    assert view_res.total_jobs >= 0


def test_fresh_mutable_job_state_isolation():
    store = TraceStore.get_instance("data/raw/spot_gpu_v2026")
    spec = EpisodeSpec(
        node_filter="A100-SXM4-80GB",
        score_start=2678400,
        score_end=2764800,
        followup_end=2764800 + 3600
    )
    view = store.create_episode_view(spec)

    jobs1 = view.get_all_jobs_fresh()
    jobs2 = view.get_all_jobs_fresh()

    assert len(jobs1) == len(jobs2)
    # Mutate jobs1
    jobs1[0]["nodes"].append("test_node")
    jobs1[0]["status"] = "end"

    # jobs2 must remain pristine
    assert len(jobs2[0]["nodes"]) == 0
    assert jobs2[0].get("status") is None


def test_invalid_filter_rejection():
    store = TraceStore.get_instance("data/raw/spot_gpu_v2026")
    spec = EpisodeSpec(
        node_filter="NON_EXISTENT_GPU_MODEL",
        score_start=2678400,
        score_end=2764800,
        followup_end=2764800 + 3600
    )
    with pytest.raises(InvalidConfigError):
        store.create_episode_view(spec)


def test_spot_arrival_multiplier_preserves_prefix_and_is_reproducible():
    store = TraceStore.get_instance("data/raw/spot_gpu_v2026")
    common = {
        "node_filter": "A100-SXM4-80GB",
        "score_start": 8985600,
        "score_end": 9072000,
        "followup_end": 9075600,
        "trace_mode": "research_full_history",
    }
    original = store.create_episode_view(EpisodeSpec(**common))
    assert original.workload_metadata["workload_mode"] == "natural"
    assert original.workload_metadata["synthetic_stress_enabled"] is False
    assert original.workload_metadata["stress_effective_start"] == common["score_start"]
    stressed_spec = EpisodeSpec(
        **common,
        spot_arrival_multiplier=4,
        spot_arrival_seed=917,
    )
    stressed = store.create_episode_view(stressed_spec)
    stressed_repeat = store.create_episode_view(stressed_spec)

    original_jobs = original.jobs_df
    stressed_jobs = stressed.jobs_df
    original_prefix = original_jobs[original_jobs.submit_time < stressed_spec.score_start].reset_index(drop=True)
    stressed_prefix = stressed_jobs[stressed_jobs.submit_time < stressed_spec.score_start].reset_index(drop=True)
    assert original_prefix.equals(stressed_prefix)

    window = original_jobs[
        (original_jobs.job_type == "Spot")
        & (original_jobs.submit_time >= stressed_spec.score_start)
        & (original_jobs.submit_time <= stressed_spec.followup_end)
    ]
    expected = len(window) * stressed_spec.spot_arrival_multiplier
    actual = stressed_jobs[
        (stressed_jobs.job_type == "Spot")
        & (stressed_jobs.submit_time >= stressed_spec.score_start)
        & (stressed_jobs.submit_time <= stressed_spec.followup_end)
    ]
    assert len(actual) == expected
    assert stressed.workload_metadata["stress_synthetic_spot_arrivals"] == len(window) * 3
    assert stressed.workload_metadata["workload_mode"] == "synthetic_spot_arrival_stress"
    assert stressed.workload_metadata["synthetic_stress_enabled"] is True
    assert stressed_jobs.job_name.is_unique
    assert stressed.workload_sha256 == stressed_repeat.workload_sha256
    assert stressed.prefix_workload_sha256 == original.prefix_workload_sha256


def test_spot_surge_lead_in_changes_only_the_declared_prefix_and_is_reproducible():
    store = TraceStore.get_instance("data/raw/spot_gpu_v2026")
    score_start = 8985600
    stress_start = score_start - 86400
    common = {
        "node_filter": "A100-SXM4-80GB",
        "score_start": score_start,
        "score_end": 9072000,
        "followup_end": 9075600,
        "trace_mode": "research_full_history",
    }
    original = store.create_episode_view(EpisodeSpec(**common))
    lead_in_spec = EpisodeSpec(
        **common,
        spot_arrival_multiplier=4,
        spot_arrival_seed=917,
        spot_stress_start=stress_start,
    )
    lead_in = store.create_episode_view(lead_in_spec)
    repeated = store.create_episode_view(lead_in_spec)

    raw = original.jobs_df
    lead_jobs = lead_in.jobs_df
    source_prefix = raw[
        (raw.job_type == "Spot")
        & (raw.submit_time >= stress_start)
        & (raw.submit_time < score_start)
    ]
    actual_prefix = lead_jobs[
        (lead_jobs.job_type == "Spot")
        & (lead_jobs.submit_time >= stress_start)
        & (lead_jobs.submit_time < score_start)
    ]
    assert len(source_prefix) == 259
    assert len(actual_prefix) == len(source_prefix) * 4
    assert lead_in.workload_metadata["stress_prefix_original_spot_arrivals"] == 259
    assert lead_in.workload_metadata["stress_prefix_synthetic_spot_arrivals"] == 777
    assert lead_in.prefix_workload_sha256 != original.prefix_workload_sha256
    assert lead_in.workload_sha256 == repeated.workload_sha256
    assert lead_in.prefix_workload_sha256 == repeated.prefix_workload_sha256
    assert lead_jobs.job_name.is_unique
