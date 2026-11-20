"""Build quota RL environments from frozen benchmark declarations."""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch

from benchmark.forecast_interventions import (
    activate_forecast_at_score_boundary,
    build_forecast_intervention,
)
from benchmark.run import (
    REPOSITORY_ROOT,
    _baseline_policy,
    _environment_config,
    _resolve,
    _validate_trace_hashes,
    build_spec_from_dict,
    load_suite,
    load_yaml,
)
from research_sim.control_env import UnifiedControlEnv
from research_sim.trace import TraceStore

from .env import QuotaResidualEnv, RiskBudgets, SafetyConfig


def _framework_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_framework_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


@dataclass
class PreparedController:
    policy: Any
    forecast_provider: Any | None
    manifest: Mapping[str, Any]


@dataclass
class BenchmarkBaseFactory:
    base_suite_path: Path
    baseline_id: str
    seed: int
    run_dir: Path
    forecast_device: str
    smoke_forecast: bool
    forecast_cache_dir: Path | None = None
    forecast_baseline_id: str | None = None

    def prepare(self, env: UnifiedControlEnv) -> PreparedController:
        suite = load_suite(self.base_suite_path)
        policy, baseline, policy_manifest = _baseline_policy(
            suite,
            self.baseline_id,
            allow_smoke_artifact=self.smoke_forecast,
        )
        provider = None
        forecast_baseline_id = self.forecast_baseline_id or self.baseline_id
        try:
            forecast_baseline = dict(suite["baselines"][forecast_baseline_id])
        except KeyError as exc:
            raise ValueError(
                f"Unknown forecast endpoint baseline {forecast_baseline_id!r}."
            ) from exc
        declaration = forecast_baseline.get("forecast_intervention")
        if declaration is not None:
            declaration = copy.deepcopy(declaration)
            declaration["device"] = self.forecast_device
            if self.forecast_cache_dir is not None:
                declaration["cache_dir"] = str(self.forecast_cache_dir)
            rng_state = _framework_rng_state()
            try:
                provider = build_forecast_intervention(
                    declaration,
                    suite_base=self.base_suite_path.parent,
                    simulator=env.sim,
                    run_dir=self.run_dir,
                    seed=int(self.seed),
                    smoke=self.smoke_forecast,
                )
            finally:
                _restore_framework_rng_state(rng_state)
            bind = getattr(policy, "bind_forecast_provider", None)
            if callable(bind):
                bind(provider)
            spec = dataclasses.replace(
                env.episode_spec,
                forecast_mode="external_causal",
                forecast_provider_id=provider.provider_id,
                forecast_provider_version=provider.provider_version,
            )
            env.sim.spec = spec
            env.sim.quota_provider = provider
            env.quota_provider = provider
            env.episode_spec = spec
            env.current_context = activate_forecast_at_score_boundary(env.sim, provider)
            policy_manifest = {
                **dict(policy_manifest),
                "forecast_intervention": provider.manifest(),
                "forecast_baseline_id": forecast_baseline_id,
            }
        return PreparedController(
            policy=policy,
            forecast_provider=provider,
            manifest={
                **dict(policy_manifest),
                "baseline_id": self.baseline_id,
                "scheduler_profile": baseline.get("scheduler_profile", "native_gfs"),
            },
        )


def _eligible_hp_gang_count(view: Any, spec: Any) -> int:
    jobs = view.job_df
    gpu_request = jobs["gpu_request"].astype(float)
    mask = (
        jobs["job_type"].astype(str).eq("HP")
        & jobs["worker_num"].astype(int).gt(1)
        & gpu_request.ge(1.0)
        & np.isclose(gpu_request, np.rint(gpu_request))
        & jobs["submit_time"].astype(float).ge(float(spec.score_start))
        & jobs["submit_time"].astype(float).lt(float(spec.score_end))
        & jobs["submit_time"].astype(float).add(float(spec.L_H)).le(float(spec.followup_end))
    )
    count = int(mask.sum())
    if count <= 0:
        raise ValueError("Selected split has no eligible HP gang cohort.")
    return count


def build_quota_rl_env(
    rl_suite: Mapping[str, Any],
    *,
    method_id: str,
    split: str,
    seed: int,
    run_dir: Path,
    smoke_forecast: bool,
) -> tuple[QuotaResidualEnv, dict[str, Any]]:
    """Create one fresh continuous simulator environment for one method seed."""

    try:
        method = dict(rl_suite["methods"][method_id])
    except KeyError as exc:
        raise ValueError(f"Unknown RL method {method_id!r}.") from exc
    if split == "test":
        raise ValueError("Final test split is locked during RL development.")
    base_suite_path = Path(str(rl_suite["environment"]["base_suite"]))
    if not base_suite_path.is_absolute():
        base_suite_path = (Path(str(rl_suite["_suite_path"])).parent / base_suite_path).resolve()
    base_suite = load_suite(base_suite_path)
    config_path = _environment_config(base_suite, split)
    cfg = load_yaml(str(config_path))
    spec = dataclasses.replace(build_spec_from_dict(cfg), policy_seed=int(seed))
    trace_dir = _resolve(REPOSITORY_ROOT, str(cfg["trace_dir"]))
    view = TraceStore.get_instance(str(trace_dir)).create_episode_view(spec)
    from benchmark.protocol_helpers import _paper_manifest_for

    paper_manifest = _paper_manifest_for(str(config_path), cfg, spec)
    if paper_manifest is None:
        raise ValueError("Quota RL requires a frozen paper configuration.")
    _validate_trace_hashes(paper_manifest, view)
    baseline_id = str(method["base_baseline"])
    baseline = dict(base_suite["baselines"][baseline_id])
    snapshot_dir = _resolve(REPOSITORY_ROOT, str(cfg["snapshot_dir"]))
    inner = UnifiedControlEnv(
        spec,
        view,
        scheduler_profile=str(baseline.get("scheduler_profile", "native_gfs")),
        use_warmup_snapshot=True,
        snapshot_dir=str(snapshot_dir),
        snapshot_missing_mode="raise",
        log_dir=str(run_dir / "logs"),
        run_id=run_dir.name,
    )
    expected_steps = int(math.ceil(
        (spec.score_end - spec.score_start) / spec.control_interval_seconds
    ))
    score_hours = (spec.score_end - spec.score_start) / 3600.0
    budgets_cfg = dict(rl_suite["constraints"])
    budgets = RiskBudgets(
        hp_violation_rate_limit=float(budgets_cfg["hp_violation_rate_limit"]),
        hp_cohort_count=_eligible_hp_gang_count(view, spec),
        lost_gpu_hours=float(budgets_cfg["lost_gpu_hours_per_score_hour"]) * score_hours,
        hp_queue_gpu_hours=float(budgets_cfg["hp_queue_gpu_hours_per_score_hour"]) * score_hours,
        expected_steps=expected_steps,
    )
    safety = SafetyConfig(**dict(rl_suite["safety"]))
    base_factory = BenchmarkBaseFactory(
        base_suite_path=base_suite_path,
        baseline_id=baseline_id,
        seed=int(seed),
        run_dir=run_dir,
        forecast_device=str(rl_suite["resources"]["forecast_device"]),
        smoke_forecast=bool(smoke_forecast),
        forecast_cache_dir=(
            None
            if rl_suite["resources"].get("forecast_cache_dir") is None
            else (
                Path(str(rl_suite["resources"]["forecast_cache_dir"]))
                if Path(str(rl_suite["resources"]["forecast_cache_dir"])).is_absolute()
                else (
                    Path(str(rl_suite["_suite_path"])).parent
                    / str(rl_suite["resources"]["forecast_cache_dir"])
                ).resolve()
            )
        ),
        forecast_baseline_id=(
            None
            if method.get("forecast_baseline") is None
            else str(method["forecast_baseline"])
        ),
    )
    env = QuotaResidualEnv(
        inner,
        base_factory=base_factory,
        risk_budgets=budgets,
        safety=safety,
        method_id=method_id,
        include_forecast_features=bool(method["include_forecast_features"]),
        action_mode=str(method.get("action_mode", "legacy_residual")),
        protection_levels=tuple(method.get("protection_levels", (0.0, 0.25, 0.5, 1.0))),
        reward_weights=method.get("reward_weights"),
    )
    metadata = {
        "method_id": method_id,
        "method": method,
        "base_suite": str(base_suite_path.relative_to(REPOSITORY_ROOT)),
        "environment_config": str(config_path.relative_to(REPOSITORY_ROOT)),
        "split": split,
        "seed": int(seed),
        "expected_steps": expected_steps,
        "score_hours": score_hours,
        "hp_cohort_count": budgets.hp_cohort_count,
        "risk_budgets": budgets.manifest(),
        "safety": safety.manifest(),
        "test_outcomes_accessed": False,
    }
    return env, metadata
