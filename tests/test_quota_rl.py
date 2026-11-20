import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from benchmark.forecast_interventions import SRLinearOnlineQuotaProvider
from benchmark.rl_run import load_rl_suite
from benchmark.rl_experiments import (
    _execute_job as _execute_rl_job,
    _jobs,
    _render_status as _render_rl_status,
)
from benchmark.seeds import stochastic_seeds_from_suite
from methods.quota_rl.env import QuotaResidualEnv, RiskBudgets, SafetyConfig
from methods.quota_rl.model import RecurrentQuotaActorCritic
from methods.quota_rl.ppo import ConstrainedRecurrentPPO, collect_rollout


class _CountingForecast:
    def __init__(self):
        self.calls = 0

    def fit_predict(self, history, origin, run_dir, calendar_timestamp=None):
        del history, origin, run_dir, calendar_timestamp
        self.calls += 1
        return SimpleNamespace(point=np.asarray([[1.0, 2.0], [3.0, 4.0]]))


class _RandomConsumingForecast(_CountingForecast):
    def fit_predict(self, history, origin, run_dir, calendar_timestamp=None):
        random.random()
        np.random.random()
        torch.rand(1)
        return super().fit_predict(history, origin, run_dir, calendar_timestamp)


def _context(*, time=0, hp_queue=0.0):
    return SimpleNamespace(
        state_version=time,
        time=time,
        base_quota_by_pool={"A100": {1: 50}},
        capacity_by_pool={"A100": 100},
        hp_used_by_pool={"A100": 40.0},
        spot_used_by_pool={"A100": 10.0},
        hp_queue_count=int(hp_queue > 0),
        hp_queue_gpu_demand=float(hp_queue),
        admission_jobs=(),
        eviction_failure_rate_by_pool={"A100": 0.0},
        score_progress=min(1.0, time / 1200.0),
    )


class _FakeSim:
    def __init__(self):
        self.terminated = False

    def terminate(self, **_kwargs):
        self.terminated = True


class _FakeInner:
    def __init__(self, *, hp_queue=0.0, horizon=4):
        self.episode_spec = SimpleNamespace(control_queue_size=8)
        self.hp_queue = float(hp_queue)
        self.horizon = int(horizon)
        self.steps = 0
        self.current_context = None
        self.sim = _FakeSim()
        self.last_action = None

    def reset(self, *, seed=None, options=None):
        del seed, options
        self.steps = 0
        self.current_context = _context(hp_queue=self.hp_queue)
        return {}, {"time": 0, "terminated": False, "truncated": False}

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32)
        self.steps += 1
        terminated = self.steps >= self.horizon
        self.current_context = None if terminated else _context(
            time=self.steps * 300,
            hp_queue=self.hp_queue,
        )
        info = {
            "time": self.steps * 300,
            "delta_t": 300,
            "hp_constraint_cost": 0.0,
            "reward_components": {
                "useful_spot_gpu_hours": 4.0,
                "spot_queue_gpu_hours": 1.0,
                "lost_gpu_hours": 0.0,
            },
        }
        return {}, 0.0, terminated, False, info


class _GradientCheckingInner(_FakeInner):
    def step(self, action):
        parameter = torch.tensor(2.0, requires_grad=True)
        loss = parameter.square()
        loss.backward()
        assert parameter.grad is not None
        return super().step(action)


class _Policy:
    def __init__(self, residual):
        self.residual = float(residual)

    def action(self, _context_value):
        return np.asarray((self.residual, 0, 0, 0, 0, 0, 0), dtype=np.float32)

    def residual_for_base(self, _context_value, _base_quota):
        return self.residual

    def manifest(self):
        return {"policy_type": "fixture"}


class _Prepared:
    def __init__(self, residual, provider=None):
        self.policy = _Policy(residual)
        self.forecast_provider = provider
        self.manifest = {"policy_type": "fixture", "residual": residual}


class _Factory:
    def __init__(self, residual=0.05, provider=None):
        self.residual = residual
        self.provider = provider

    def prepare(self, _env):
        return _Prepared(self.residual, self.provider)


def _wrapped(
    *,
    hp_queue=0.0,
    residual=0.05,
    provider=None,
    include_forecast=False,
    horizon=4,
    action_mode="legacy_residual",
    reward_weights=None,
):
    return QuotaResidualEnv(
        _FakeInner(hp_queue=hp_queue, horizon=horizon),
        base_factory=_Factory(residual, provider),
        risk_budgets=RiskBudgets(
            hp_violation_rate_limit=0.02,
            hp_cohort_count=100,
            lost_gpu_hours=10.0,
            hp_queue_gpu_hours=10.0,
            expected_steps=horizon,
        ),
        safety=SafetyConfig(),
        method_id="fixture",
        include_forecast_features=include_forecast,
        action_mode=action_mode,
        reward_weights=reward_weights,
    )


def test_zero_rl_action_reproduces_base_action():
    env = _wrapped(residual=0.05)
    observation, _ = env.reset(seed=1)
    assert env.observation_space.contains(observation)
    env.step(np.asarray([0.0], dtype=np.float32))
    assert env.inner.last_action[0] == np.float32(0.05)
    np.testing.assert_array_equal(env.inner.last_action[1:], np.zeros(6, dtype=np.float32))


def test_zero_rl_action_preserves_positive_srlinear_base_before_shielding():
    provider = SimpleNamespace(
        srlinear_quota_by_pool={"A100": {1: 50}},
        safe_quota_by_pool={"A100": {1: 40}},
    )
    env = _wrapped(
        residual=0.03,
        provider=provider,
        include_forecast=True,
    )
    env.reset(seed=1)
    _, _, _, _, info = env.step(np.asarray([0.0], dtype=np.float32))
    assert env.inner.last_action[0] == np.float32(0.03)
    assert info["rl_action"]["shielded"] is False


def test_srac_zero_action_reproduces_contextual_endpoint():
    provider = SimpleNamespace(
        srlinear_quota_by_pool={"A100": {1: 70}},
        native_quota_by_pool={"A100": {1: 50}},
        uncertainty_by_pool={"A100": 10.0},
    )
    env = _wrapped(
        residual=0.05,
        provider=provider,
        include_forecast=True,
        action_mode="srac_mix",
    )
    observation, _ = env.reset(seed=1)
    assert observation.shape == (38,)
    assert env.action_space.shape == (2,)
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    np.testing.assert_allclose(env.inner.last_action[0], -0.15, atol=1e-7)
    assert info["rl_action"]["target_quota"] == 55.0
    assert info["rl_action"]["executed_alpha"] == 0.0
    assert info["rl_action"]["executed_beta"] == 0.0


def test_srac_action_mixes_quota_and_applies_uncertainty_with_ramps():
    provider = SimpleNamespace(
        srlinear_quota_by_pool={"A100": {1: 70}},
        native_quota_by_pool={"A100": {1: 50}},
        uncertainty_by_pool={"A100": 10.0},
    )
    env = _wrapped(
        residual=0.05,
        provider=provider,
        include_forecast=True,
        action_mode="srac_mix",
    )
    env.reset(seed=1)
    _, _, _, _, info = env.step(np.ones(2, dtype=np.float32))
    diagnostics = info["rl_action"]
    assert diagnostics["executed_alpha"] == 0.25
    assert diagnostics["executed_beta"] == 0.25
    np.testing.assert_allclose(diagnostics["target_quota"], 56.25, atol=1e-7)
    np.testing.assert_allclose(env.inner.last_action[0], -0.1375, atol=1e-7)


def test_safety_blocks_positive_delta_when_hp_queue_exists():
    env = _wrapped(hp_queue=4.0, residual=0.02)
    env.reset(seed=1)
    _, _, _, _, info = env.step(np.asarray([1.0], dtype=np.float32))
    assert env.inner.last_action[0] == np.float32(0.02)
    assert "hp_queue" in info["rl_action"]["shield_reasons"]
    assert info["constraint_costs_raw"][2] > 0.0


def test_continuous_protective_action_only_reduces_quota():
    env = _wrapped(residual=0.02, action_mode="continuous_protective")
    env.reset(seed=1)
    _, _, _, _, positive = env.step(np.asarray([1.0], dtype=np.float32))
    assert env.inner.last_action[0] == np.float32(0.02)
    assert positive["rl_action"]["protection_intensity"] == 0.0
    env.reset(seed=1)
    _, _, _, _, negative = env.step(np.asarray([-1.0], dtype=np.float32))
    assert env.inner.last_action[0] < np.float32(0.02)
    assert negative["rl_action"]["protection_intensity"] == 1.0


def test_discrete_protective_action_uses_declared_level():
    env = QuotaResidualEnv(
        _FakeInner(),
        base_factory=_Factory(0.02),
        risk_budgets=RiskBudgets(0.02, 100, 10.0, 10.0, 4),
        safety=SafetyConfig(max_delta_change_fraction=0.05),
        method_id="guard_fixture",
        include_forecast_features=False,
        action_mode="discrete_protective",
        protection_levels=(0.0, 0.25, 0.5, 1.0),
    )
    env.reset(seed=1)
    _, _, _, _, info = env.step(np.asarray([-0.63], dtype=np.float32))
    assert info["rl_action"]["protection_intensity"] == 0.5
    np.testing.assert_allclose(env.inner.last_action[0], -0.005, atol=1e-7)


def test_operational_reward_matches_frozen_reward_components():
    env = _wrapped(
        residual=0.0,
        reward_weights={
            "useful_spot_gpu_hours": 1.0,
            "spot_queue_gpu_hours": 1.0,
            "lost_gpu_hours": 1.0,
            "hp_queue_gpu_hours_proxy": 0.0,
        },
    )
    env.reset(seed=1)
    _, reward, _, _, info = env.step(np.asarray([0.0], dtype=np.float32))
    expected = (4.0 - 1.0) / (100.0 * 300.0 / 3600.0)
    np.testing.assert_allclose(reward, expected)
    assert info["training_reward_terms"]["weighted_utility"] == 3.0


def test_srlinear_point_cache_reuses_exact_causal_history(tmp_path):
    history = pd.DataFrame(
        {"a": [1.0, 2.0], "b": [3.0, 4.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="h"),
    )
    context = SimpleNamespace(org_request_history=history, guarantee_rate=0.98)
    origin = pd.Timestamp("2024-01-02")
    first_model = _CountingForecast()
    first = SRLinearOnlineQuotaProvider(
        model=first_model,
        mode="mean",
        run_dir=tmp_path / "first",
        seed=17,
        calibration={},
        cache_dir=tmp_path / "cache",
        cache_namespace="fixture",
    )
    first._refresh(context, origin)
    second_model = _CountingForecast()
    second = SRLinearOnlineQuotaProvider(
        model=second_model,
        mode="mean",
        run_dir=tmp_path / "second",
        seed=17,
        calibration={},
        cache_dir=tmp_path / "cache",
        cache_namespace="fixture",
    )
    second._refresh(context, origin)
    assert first_model.calls == 1
    assert second_model.calls == 0
    np.testing.assert_array_equal(first._point, second._point)
    assert second.manifest()["forecast_cache_hits"] == 1
    assert second.manifest()["fit_count"] == 0
    assert second.manifest()["refresh_count"] == 1


def test_srlinear_point_cache_isolates_different_causal_histories(tmp_path):
    history = pd.DataFrame(
        {"a": [1.0, 2.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="h"),
    )
    origin = pd.Timestamp("2024-01-02")
    provider = SRLinearOnlineQuotaProvider(
        model=_CountingForecast(),
        mode="mean",
        run_dir=tmp_path / "first",
        seed=17,
        calibration={},
        cache_dir=tmp_path / "cache",
        cache_namespace="fixture",
    )
    provider._refresh(SimpleNamespace(org_request_history=history), origin)
    changed = history.copy()
    changed.iloc[0, 0] = 99.0
    second = SRLinearOnlineQuotaProvider(
        model=_CountingForecast(),
        mode="mean",
        run_dir=tmp_path / "second",
        seed=17,
        calibration={},
        cache_dir=tmp_path / "cache",
        cache_namespace="fixture",
    )
    second._refresh(SimpleNamespace(org_request_history=changed), origin)
    assert second.model.calls == 1
    assert second.manifest()["forecast_cache_hits"] == 0
    assert second.manifest()["forecast_cache_misses"] == 1
    cache_files = list((tmp_path / "cache" / "fixture" / "seed17").rglob("*.npz"))
    assert len(cache_files) == 2


def test_srlinear_fit_preserves_global_framework_rng_state(tmp_path):
    history = pd.DataFrame(
        {"a": [1.0, 2.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="h"),
    )
    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    expected = (random.random(), np.random.random(), float(torch.rand(1)))
    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    provider = SRLinearOnlineQuotaProvider(
        model=_RandomConsumingForecast(),
        mode="mean",
        run_dir=tmp_path / "fit",
        seed=17,
        calibration={},
    )
    provider._refresh(
        SimpleNamespace(org_request_history=history), pd.Timestamp("2024-01-02")
    )
    actual = (random.random(), np.random.random(), float(torch.rand(1)))
    np.testing.assert_allclose(actual, expected)


def test_srlinear_policy_cannot_exceed_forecast_base():
    provider = SimpleNamespace(
        srlinear_quota_by_pool={"A100": {1: 50}},
        safe_quota_by_pool={"A100": {1: 40}},
    )
    env = QuotaResidualEnv(
        _FakeInner(),
        base_factory=_Factory(-0.01, provider),
        risk_budgets=RiskBudgets(0.02, 100, 10.0, 10.0, 4),
        safety=SafetyConfig(max_delta_change_fraction=0.05),
        method_id="srqrl",
        include_forecast_features=True,
    )
    observation, _ = env.reset(seed=2)
    assert observation[-3] == np.float32(1.0)
    _, _, _, _, info = env.step(np.asarray([1.0], dtype=np.float32))
    assert env.inner.last_action[0] == np.float32(0.0)
    assert "srlinear_forecast_cap" in info["rl_action"]["shield_reasons"]


def test_recurrent_ppo_collects_and_updates_one_rollout():
    torch.manual_seed(3)
    env = _wrapped(residual=0.0, horizon=4)
    observation, _ = env.reset(seed=3)
    model = RecurrentQuotaActorCritic(len(observation), hidden_size=16)
    trainer = ConstrainedRecurrentPPO(model, ppo_epochs=2)
    hidden = model.initial_hidden(1, "cpu")
    batch, _, _ = collect_rollout(
        env,
        model,
        observation,
        hidden,
        rollout_steps=4,
        device=torch.device("cpu"),
    )
    result = trainer.update(batch)
    assert batch.steps == 4
    assert batch.terminal is True
    assert np.isfinite(result["total_loss"])
    assert len(result["lagrange"]) == 3


def test_pid_dual_state_updates_and_round_trips():
    torch.manual_seed(3)
    env = _wrapped(residual=0.0, horizon=4)
    observation, _ = env.reset(seed=3)
    model = RecurrentQuotaActorCritic(len(observation), hidden_size=16)
    trainer = ConstrainedRecurrentPPO(model, ppo_epochs=1, dual_mode="pid")
    batch, _, _ = collect_rollout(
        env,
        model,
        observation,
        model.initial_hidden(1, "cpu"),
        rollout_steps=4,
        device=torch.device("cpu"),
    )
    result = trainer.update(batch)
    state = trainer.state_dict()
    restored = ConstrainedRecurrentPPO(
        RecurrentQuotaActorCritic(len(observation), hidden_size=16),
        ppo_epochs=1,
        dual_mode="pid",
    )
    restored.load_state_dict(state)
    assert result["dual_mode"] == "pid"
    np.testing.assert_allclose(restored.lagrange, trainer.lagrange)
    np.testing.assert_allclose(restored.pid_integral, trainer.pid_integral)


def test_zero_initialized_actor_reproduces_base_deterministically():
    model = RecurrentQuotaActorCritic(37, hidden_size=16)
    observation = torch.ones((1, 37), dtype=torch.float32)
    output = model.step(
        observation,
        model.initial_hidden(1, "cpu"),
        deterministic=True,
    )
    torch.testing.assert_close(output["action"], torch.zeros((1, 1)))


def test_two_dimensional_actor_is_zero_initialized_for_srac():
    model = RecurrentQuotaActorCritic(38, hidden_size=16, action_dim=2)
    output = model.step(
        torch.ones((1, 38), dtype=torch.float32),
        model.initial_hidden(1, "cpu"),
        deterministic=True,
    )
    torch.testing.assert_close(output["action"], torch.zeros((1, 2)))
    assert model.manifest()["action_dim"] == 2


def test_risk_cost_scaling_sets_full_budget_mean_to_one():
    budgets = RiskBudgets(0.02, 100, 10.0, 20.0, 4)
    one_step_budget = budgets.totals / budgets.expected_steps
    np.testing.assert_allclose(
        budgets.scale_costs(one_step_budget),
        np.ones(3, dtype=np.float32),
    )


def test_rollout_does_not_disable_gradients_inside_environment_step():
    inner = _GradientCheckingInner(horizon=2)
    env = QuotaResidualEnv(
        inner,
        base_factory=_Factory(0.0),
        risk_budgets=RiskBudgets(0.02, 100, 10.0, 10.0, 2),
        safety=SafetyConfig(),
        method_id="gradient_fixture",
        include_forecast_features=False,
    )
    observation, _ = env.reset(seed=4)
    model = RecurrentQuotaActorCritic(len(observation), hidden_size=8)
    collect_rollout(
        env,
        model,
        observation,
        model.initial_hidden(1, "cpu"),
        rollout_steps=2,
        device=torch.device("cpu"),
    )


def test_rl_suite_declares_three_methods_and_generated_seeds():
    path = Path(__file__).parents[1] / "benchmark" / "rl_suite.yaml"
    suite = load_rl_suite(path)
    assert set(suite["methods"]) == {"cqrl", "srqrl", "srlinear_rl"}
    assert suite["methods"]["srlinear_rl"]["base_baseline"] == "gfs_srlinear"
    assert suite["algorithm"]["checkpoint_every_updates"] == 4
    assert stochastic_seeds_from_suite(suite) == (
        3302413169,
        2035845825,
        1193624495,
    )


def test_rl_batch_declares_nine_full_jobs_in_requested_family_order():
    path = Path(__file__).parents[1] / "benchmark" / "rl_suite.yaml"
    suite = load_rl_suite(path)
    jobs = _jobs(suite)
    assert len(jobs) == 9
    assert [job["method_id"] for job in jobs[:3]] == ["srlinear_rl"] * 3
    assert [job["method_id"] for job in jobs[3:6]] == ["srqrl"] * 3
    assert [job["method_id"] for job in jobs[6:]] == ["cqrl"] * 3
    assert {job["expected_train_steps"] for job in jobs} == {17568}
    assert {job["expected_eval_steps"] for job in jobs} == {8640}


def test_srlinear_risk_suite_declares_four_methods_and_seed_limited_pilot():
    path = Path(__file__).parents[1] / "benchmark" / "srlinear_rl_suite.yaml"
    suite = load_rl_suite(path)
    assert set(suite["methods"]) == {
        "sr_utility_ppo",
        "sr_pid_ppo",
        "sr_priority_pid",
        "srguard_pid",
    }
    assert all(
        method["base_baseline"] == "gfs_srlinear"
        for method in suite["methods"].values()
    )
    jobs = _jobs(suite, seed_limit=1)
    assert len(jobs) == 4
    assert {job["seed"] for job in jobs} == {3302413169}
    pid_jobs = _jobs(suite, seed_limit=1, method_ids=["sr_pid_ppo"])
    assert [job["method_id"] for job in pid_jobs] == ["sr_pid_ppo"]


def test_srac_suite_declares_one_three_seed_method():
    path = Path(__file__).parents[1] / "benchmark" / "srac_rl_suite.yaml"
    suite = load_rl_suite(path)
    assert set(suite["methods"]) == {"srac"}
    assert suite["methods"]["srac"]["base_baseline"] == "contextual_linear"
    assert suite["methods"]["srac"]["forecast_baseline"] == "srgfs_contextual_mix"
    assert suite["methods"]["srac"]["action_mode"] == "srac_mix"
    jobs = _jobs(suite)
    assert len(jobs) == 3
    assert {job["seed"] for job in jobs} == {
        3302413169,
        2035845825,
        1193624495,
    }


def test_rl_job_resumes_when_atomic_recovery_exists(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "recovery.pt").write_bytes(b"checkpoint")
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    progress_file = job_dir / "progress.json"
    progress_file.write_text(json.dumps({
        "state": "TRAINING",
        "method_id": "srlinear_rl",
        "seed": 3302413169,
        "run_dir": str(run_dir),
        "recovery_checkpoint": str(run_dir / "recovery.pt"),
        "completed_train_steps": 256,
        "completed_eval_steps": 0,
    }), encoding="utf-8")
    captured = {}

    def fake_run(command, **_kwargs):
        captured["command"] = command
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr("benchmark.rl_experiments.subprocess.run", fake_run)
    result = _execute_rl_job({
        "method_id": "srlinear_rl",
        "seed": 3302413169,
        "progress_file": str(progress_file),
    }, "suite.yaml", 4)
    assert result["resumed"] is True
    assert "resume" in captured["command"]
    assert str(run_dir) in captured["command"]


def test_rl_status_labels_dead_supervisor_as_stale(tmp_path):
    job_dir = tmp_path / "jobs" / "srlinear_rl" / "seed1"
    job_dir.mkdir(parents=True)
    progress_file = job_dir / "progress.json"
    progress_file.write_text(json.dumps({
        "state": "TRAINING",
        "completed_train_steps": 256,
        "completed_eval_steps": 0,
    }), encoding="utf-8")
    (tmp_path / "batch.json").write_text(json.dumps({
        "batch_id": "fixture",
        "jobs": [{
            "method_id": "srlinear_rl",
            "seed": 1,
            "expected_train_steps": 512,
            "expected_eval_steps": 128,
            "progress_file": str(progress_file),
        }],
    }), encoding="utf-8")
    (tmp_path / "status.json").write_text(
        json.dumps({"state": "RUNNING"}), encoding="utf-8"
    )
    rendered = _render_rl_status(tmp_path)
    assert "state=STALE_INTERRUPTED" in rendered
    assert "train   256/512" in rendered
