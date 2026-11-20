from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from research_sim.control import (
    ADMISSION_FEATURE_NAMES,
    build_admission_job_view,
)
from research_sim.control_env import UnifiedControlEnv
from research_sim.core import ResearchSimulator
from research_sim.measurement import reduce_replay_events
from research_sim.trace import EpisodeTraceView
from research_sim.types import EpisodeSpec, ResearchSimError


def _fixture(jobs, *, capacity=4, score_end=600):
    node_df = pd.DataFrame([{
        "node_name": 0,
        "gpu_model": "A100-SXM4-80GB",
        "vc_name": "A100-SXM4-80GB",
        "gpu_capacity_num": capacity,
        "cpu_num": 128,
    }])
    job_df = pd.DataFrame(jobs)
    spec = EpisodeSpec(
        score_start=0,
        score_end=score_end,
        followup_end=score_end + 3600,
        trace_mode="research_full_history",
        forecast_mode="causal_capacity",
        B_call=0,
        K=0,
        control_queue_size=8,
    )
    view = EpisodeTraceView(
        node_df,
        job_df,
        spec,
        {"node_sha256": "fixture", "job_sha256": "fixture"},
    )
    return view, spec


def _job(job_id, gpu_request, *, worker_num=1, duration=900):
    return {
        "job_name": job_id,
        "organization": 0,
        "gpu_model": "A100-SXM4-80GB",
        "worker_num": worker_num,
        "gpu_request": float(gpu_request),
        "cpu_request": 8.0,
        "submit_time": 0.0,
        "duration": float(duration),
        "job_type": "Spot",
    }


def _run_control(env, first_action=None):
    obs, info = env.reset(seed=7)
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    action = zero if first_action is None else np.asarray(first_action, dtype=np.float32)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        obs, _, terminated, truncated, info = env.step(action if steps == 0 else zero)
        steps += 1
        assert env.observation_space.contains(obs)
        assert steps < 50
    return env.sim


def test_v4_spaces_and_gymnasium_contract():
    view, spec = _fixture([_job(1, 2)])
    env = UnifiedControlEnv(spec, view, log_dir="/tmp/test_control_env_v4")
    obs, info = env.reset(seed=3)
    assert env.action_space.shape == (1 + len(ADMISSION_FEATURE_NAMES),)
    assert env.observation_space.contains(obs)
    assert info["control_schema_version"] == "4.0"
    check_env(
        UnifiedControlEnv(spec, view, log_dir="/tmp/test_control_env_v4_check"),
        skip_render_check=True,
    )


def test_zero_control_action_matches_native_resource_outcomes():
    jobs = [_job(1, 4, duration=200), _job(2, 2, duration=300)]
    view, spec = _fixture(jobs)
    native = ResearchSimulator(
        spec,
        view,
        policy_type="native_gfs",
        log_dir="/tmp/test_control_native",
    )
    while True:
        _, delta = native.advance_until_decision()
        if delta.terminated or delta.truncated:
            break

    controlled = _run_control(
        UnifiedControlEnv(spec, view, log_dir="/tmp/test_control_zero")
    )
    native_metrics = native.get_metrics()
    controlled_metrics = controlled.get_metrics()
    assert controlled_metrics.useful_spot_gpu_hours_score_raw == pytest.approx(
        native_metrics.useful_spot_gpu_hours_score_raw
    )
    assert controlled_metrics.spot_evictions_count == native_metrics.spot_evictions_count
    assert controlled_metrics.hp_gang_violation_count == native_metrics.hp_gang_violation_count
    native_starts = [e.job_id for e in native.ledger.events if e.event_type == "start"]
    controlled_starts = [e.job_id for e in controlled.ledger.events if e.event_type == "start"]
    assert controlled_starts == native_starts


def test_zero_admission_weights_skip_feature_rebuild(monkeypatch):
    view, spec = _fixture([_job(1, 4), _job(2, 2)])
    env = UnifiedControlEnv(spec, view, log_dir="/tmp/test_control_zero_fast_path")
    env.reset(seed=3)
    env.sim.spot_que_list["A100-SXM4-80GB"] = list(env.sim.trace.job_list)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("zero weights must use exact native ordering")

    monkeypatch.setattr(env.sim, "_admission_views", fail_if_called)
    ordered = env.sim._order_spot_queue("A100-SXM4-80GB")
    assert [job["job_name"] for job in ordered] == [1, 2]


def test_quota_action_changes_admission_without_changing_trace():
    view, spec = _fixture([_job(1, 4, duration=900)])
    low_env = UnifiedControlEnv(spec, view, log_dir="/tmp/test_control_low_quota")
    low_env.reset(seed=5)
    low_action = np.zeros(low_env.action_space.shape, dtype=np.float32)
    low_action[0] = -1.0
    _, low_reward, _, _, low_info = low_env.step(low_action)
    low_starts = [
        event for event in low_env.sim.ledger.events
        if event.event_type == "start" and event.time < 300
    ]

    native_env = UnifiedControlEnv(spec, view, log_dir="/tmp/test_control_base_quota")
    native_env.reset(seed=5)
    _, native_reward, _, _, native_info = native_env.step(
        np.zeros(native_env.action_space.shape, dtype=np.float32)
    )
    native_starts = [
        event for event in native_env.sim.ledger.events
        if event.event_type == "start" and event.time < 300
    ]
    assert low_starts == []
    assert [event.job_id for event in native_starts] == [1]
    assert low_info["reward_components"]["spot_queue_gpu_hours"] > 0.0
    assert native_info["reward_components"]["spot_queue_gpu_hours"] == 0.0
    assert low_reward < native_reward


def test_priority_action_changes_which_feasible_job_starts():
    view, spec = _fixture([_job(1, 4), _job(2, 2)])
    env = UnifiedControlEnv(spec, view, log_dir="/tmp/test_control_priority")
    env.reset(seed=9)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    demand_index = ADMISSION_FEATURE_NAMES.index("total_gpu_demand_fraction")
    action[1 + demand_index] = -1.0
    env.step(action)
    starts = [
        event.job_id for event in env.sim.ledger.events
        if event.event_type == "start" and event.time < 300
    ]
    assert starts == [2]


def test_admission_view_excludes_trace_known_duration():
    common = {
        "job_name": 3,
        "vc_name": "A100-SXM4-80GB",
        "worker_num": 2,
        "gpu_request": 1.0,
        "submit_time": 10,
        "guarantee_request": 1,
    }
    short = build_admission_job_view(
        {**common, "duration": 10},
        time=100,
        total_nodes=4,
        total_capacity=32,
        max_gpu_per_worker=8,
        max_guarantee_hours=4,
        estimated_duration_seconds=600,
    )
    long = build_admission_job_view(
        {**common, "duration": 100000},
        time=100,
        total_nodes=4,
        total_capacity=32,
        max_gpu_per_worker=8,
        max_guarantee_hours=4,
        estimated_duration_seconds=600,
    )
    assert short == long


def test_control_env_rejects_invalid_action():
    view, spec = _fixture([_job(1, 2)])
    env = UnifiedControlEnv(spec, view, log_dir="/tmp/test_control_invalid")
    env.reset(seed=1)
    with pytest.raises(ResearchSimError, match="action space"):
        env.step(np.full(env.action_space.shape, 2.0, dtype=np.float32))


@pytest.mark.parametrize(
    "profile",
    (
        "yarn_cs_source_compatible",
        "chronus_source_compatible",
        "lyra_source_compatible",
        "fgd_source_compatible",
        "capacity_open_fcfs_best_fit_probe",
        "staggered_lease_best_fit_probe",
        "whole_card_cost_best_fit_probe",
        "whole_card_fgd_probe",
    ),
)
def test_literature_scheduler_profiles_complete_tiny_replay(profile):
    view, spec = _fixture([_job(1, 2, duration=350)], score_end=600)
    sim = _run_control(UnifiedControlEnv(
        spec,
        view,
        scheduler_profile=profile,
        log_dir=f"/tmp/test_{profile}",
    ))
    assert sim.scheduler_profile == profile
    assert sim.ledger.completed_jobs["Spot"] == 1


def test_yarn_profile_retains_gfs_quota():
    view, spec = _fixture([_job(1, 4, duration=100)], score_end=300)
    env = UnifiedControlEnv(
        spec,
        view,
        scheduler_profile="yarn_cs_source_compatible",
        log_dir="/tmp/test_yarn_quota",
    )
    env.reset(seed=5)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[0] = -1.0
    env.step(action)
    starts = [
        event.job_id for event in env.sim.ledger.events
        if event.event_type == "start" and event.time < 300
    ]
    assert starts == []


def test_capacity_open_probe_overrides_gfs_quota():
    view, spec = _fixture([_job(1, 4, duration=100)], score_end=300)
    env = UnifiedControlEnv(
        spec,
        view,
        scheduler_profile="capacity_open_fcfs_best_fit_probe",
        log_dir="/tmp/test_capacity_open_probe",
    )
    env.reset(seed=5)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[0] = -1.0
    env.step(action)
    starts = [
        event.job_id for event in env.sim.ledger.events
        if event.event_type == "start" and event.time < 300
    ]
    assert starts == [1]


def test_yarn_profile_uses_released_direct_preemption_path():
    jobs = [
        _job(1, 1, duration=2000),
        {
            **_job(2, 1, duration=100),
            "submit_time": 10.0,
            "job_type": "HP",
        },
    ]
    view, spec = _fixture(jobs, capacity=2, score_end=300)
    sim = _run_control(UnifiedControlEnv(
        spec,
        view,
        scheduler_profile="yarn_cs_source_compatible",
        log_dir="/tmp/test_yarn_source_preemption",
    ))
    preemptions = [
        event for event in sim.ledger.events
        if event.event_type == "preemption" and event.job_id == 1
    ]
    assert preemptions
    hp_start = next(
        event for event in sim.ledger.events
        if event.event_type == "start" and event.job_id == 2
    )
    assert hp_start.time == 10


def test_chronus_profile_waits_for_lease_and_never_preempts():
    jobs = [
        _job(1, 4, duration=2000),
        {
            **_job(2, 4, duration=100),
            "submit_time": 10.0,
            "job_type": "HP",
        },
    ]
    view, spec = _fixture(jobs, score_end=1500)
    sim = _run_control(UnifiedControlEnv(
        spec,
        view,
        scheduler_profile="chronus_source_compatible",
        log_dir="/tmp/test_chronus_leases",
    ))
    hp_start = next(
        event for event in sim.ledger.events
        if event.event_type == "start" and event.job_id == 2
    )
    assert hp_start.time == 1200
    assert not any(event.event_type == "preemption" for event in sim.ledger.events)
    expiration = next(
        event for event in sim.ledger.events
        if event.event_type == "lease_expiration" and event.job_id == 1
    )
    assert expiration.details["rollback_seconds"] == 0
    assert expiration.details["overhead_seconds"] == 120
    assert expiration.details["remain_after"] == expiration.details["remain_before"]
    assert expiration.details["lease_seconds"] == 1200
    assert expiration.details["checkpoint_progress_seconds"] == 1199
    reduced = reduce_replay_events(
        [event.to_dict() for event in sim.ledger.events],
        spec,
        expected_run_id=sim.run_id,
        expected_episode_id=sim.episode_id,
    )
    assert reduced["spot_eviction_events_score"] == 0


def test_chronus_lease_does_not_replace_application_checkpoint_interval():
    view, spec = _fixture([_job(1, 4, duration=5000)], score_end=6000)
    sim = _run_control(UnifiedControlEnv(
        spec,
        view,
        scheduler_profile="chronus_source_compatible",
        log_dir="/tmp/test_chronus_checkpoint_progress",
    ))
    credits = [
        event for event in sim.ledger.events
        if (
            event.event_type == "credit"
            and event.job_id == 1
            and event.details["reason"] == "checkpoint"
        )
    ]
    expirations = [
        event for event in sim.ledger.events
        if event.event_type == "lease_expiration" and event.job_id == 1
    ]
    assert expirations
    assert all(event.details["lease_seconds"] == 1200 for event in expirations)
    assert all(event.details["overhead_seconds"] == 120 for event in expirations)
    assert len(credits) == 1
    assert credits[0].details["credited_run_seconds"] == 3600


def test_chronus_overhead_never_earns_useful_service():
    view, spec = _fixture([_job(1, 4, duration=900)], score_end=2000)
    sim = _run_control(UnifiedControlEnv(
        spec,
        view,
        scheduler_profile="chronus_source_compatible",
        log_dir="/tmp/test_chronus_service_cap",
    ))
    credits = [
        event for event in sim.ledger.events
        if event.event_type == "credit" and event.job_id == 1
    ]
    assert sum(event.details["credited_run_seconds"] for event in credits) == 900
    assert credits[-1].details["cumulative_credited_gpu_seconds"] == 900 * 4
    assert credits[-1].details["job_credit_cap_gpu_seconds"] == 900 * 4


def test_staggered_lease_probe_expires_spot_after_five_minutes():
    view, spec = _fixture([_job(1, 4, duration=900)], score_end=700)
    sim = _run_control(UnifiedControlEnv(
        spec,
        view,
        scheduler_profile="staggered_lease_best_fit_probe",
        log_dir="/tmp/test_staggered_lease_probe",
    ))
    expiration = next(
        event for event in sim.ledger.events
        if event.event_type == "lease_expiration" and event.job_id == 1
    )
    assert expiration.time == 300
    assert expiration.details["lease_seconds"] == 300
    assert expiration.details["overhead_seconds"] == 120


def test_fgd_profile_selects_minimum_fragmentation_delta():
    class FakeNode:
        def __init__(self, name, free_gpus):
            self.node_name = name
            self.free_gpus = free_gpus

        def allocate_gpu(self, cards, job):
            self.free_gpus -= cards
            return list(range(cards))

    nodes = [FakeNode("free-four", 4), FakeNode("free-six", 6)]
    sim = ResearchSimulator.__new__(ResearchSimulator)
    sim.cluster = SimpleNamespace(
        vc_dict={"A100-SXM4-80GB": SimpleNamespace(node_list=nodes)}
    )
    job = {
        "vc_name": "A100-SXM4-80GB",
        "gpu_request": 2.0,
        "worker_num": 1,
        "nodes": [],
    }
    assert sim._fgd_place(job) is True
    assert list(job["nodes"][0]) == ["free-six"]


def test_fgd_profile_uses_fractional_source_path():
    class FakeNode:
        def __init__(self, name, allocated):
            self.node_name = name
            self.num_gpus = 1
            self.node_galloc = {0: allocated}
            self.free_gpus = 0

        def allocate_share_gpu(self, card, job):
            self.node_galloc[card] += job["gpu_request"]
            return True

    loose = FakeNode("loose", 0.2)
    tight = FakeNode("tight", 0.6)
    sim = ResearchSimulator.__new__(ResearchSimulator)
    sim.cluster = SimpleNamespace(
        vc_dict={"A100-SXM4-80GB": SimpleNamespace(node_list=[loose, tight])}
    )
    job = {
        "vc_name": "A100-SXM4-80GB",
        "gpu_request": 0.2,
        "worker_num": 1,
        "nodes": [],
    }
    assert sim._fgd_place(job) is True
    assert list(job["nodes"][0]) == ["tight"]


def test_lyra_queue_key_preserves_source_stable_ties():
    sim = ResearchSimulator.__new__(ResearchSimulator)
    sim.scheduler_profile = "lyra_source_compatible"
    first = {"gpu_request": 2.0, "worker_num": 8, "submit_time": 100, "job_name": 1}
    second = {"gpu_request": 2.0, "worker_num": 1, "submit_time": 0, "job_name": 2}
    assert sim._hp_queue_key(first) == sim._hp_queue_key(second)


def test_lyra_profile_selects_lower_preemption_cost_node():
    expensive = {"job_name": 1, "type": "Spot", "worker_num": 1, "gpu_request": 1.0}
    cheap = {"job_name": 2, "type": "Spot", "worker_num": 4, "gpu_request": 1.0}

    def fake_node(name, victim):
        return SimpleNamespace(
            node_name=name,
            num_gpus=1,
            node_type_galloc=[{"HP": 0.0}],
            node_gpu_dict=[[victim]],
        )

    sim = ResearchSimulator.__new__(ResearchSimulator)
    sim.scheduler_profile = "lyra_source_compatible"
    sim.cluster = SimpleNamespace(vc_dict={
        "A100-SXM4-80GB": SimpleNamespace(node_list=[
            fake_node("expensive", expensive),
            fake_node("cheap", cheap),
        ])
    })
    job = {"vc_name": "A100-SXM4-80GB", "gpu_request": 1.0}
    ok, node, cards, victims = sim._profile_preemption_plan(job)
    assert ok is True
    assert node.node_name == "cheap"
    assert cards == [0]
    assert [victim["job_name"] for victim in victims] == [2]
