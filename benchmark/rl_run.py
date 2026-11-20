"""Prepare, smoke test, train, and evaluate safe residual quota RL methods."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any, Mapping

# Required before the first deterministic CUDA matrix operation.  SRLinear
# enables deterministic PyTorch algorithms during each causal forecast fit.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml


BENCHMARK_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = BENCHMARK_ROOT.parent
ENVIRONMENT_ROOT = REPOSITORY_ROOT / "environment"
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(ENVIRONMENT_ROOT / "src"))

from benchmark.publication_metrics import derive_publication_metrics  # noqa: E402
from benchmark.run import (  # noqa: E402
    _EventDictView,
    _environment_config,
    _resolve,
    load_suite,
    run_baseline,
    verify_workspace,
)
from benchmark.seeds import stochastic_seeds_from_suite  # noqa: E402
from methods.quota_rl.factory import build_quota_rl_env  # noqa: E402
from methods.quota_rl.model import RecurrentQuotaActorCritic  # noqa: E402
from methods.quota_rl.ppo import ConstrainedRecurrentPPO, collect_rollout  # noqa: E402
from research_sim.control_vector import resolve_policy_device  # noqa: E402
from research_sim.measurement import reduce_replay_events  # noqa: E402


RECOVERY_SCHEMA = "safe-residual-quota-rl-recovery-1.0"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist one complete recovery generation before exposing it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _load_recovery(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"RL recovery checkpoint is missing: {path}")
    payload = torch.load(path, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != RECOVERY_SCHEMA:
        raise ValueError(f"Unsupported RL recovery checkpoint: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_threads(count: int) -> dict[str, Any]:
    if count < 1:
        raise ValueError("threads_per_worker must be positive.")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(count)
    torch.set_num_threads(count)
    return {
        "declared_threads_per_worker": count,
        "torch_threads": torch.get_num_threads(),
    }


def load_rl_suite(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise ValueError("RL suite requires schema_version 1.0.")
    methods = payload.get("methods")
    if not isinstance(methods, Mapping) or not methods:
        raise ValueError("RL suite must declare at least one method.")
    for method_id, method in methods.items():
        if not isinstance(method_id, str) or not isinstance(method, Mapping):
            raise ValueError("RL method declarations must be named mappings.")
        if "base_baseline" not in method or "include_forecast_features" not in method:
            raise ValueError(f"RL method {method_id!r} lacks its base declaration.")
    if payload.get("environment", {}).get("test_policy") != "locked":
        raise ValueError("RL suite must keep the final test split locked.")
    payload["_suite_path"] = str(resolved)
    stochastic_seeds_from_suite(payload)
    return payload


def _base_suite_path(suite: Mapping[str, Any]) -> Path:
    path = Path(str(suite["environment"]["base_suite"]))
    if not path.is_absolute():
        path = (Path(str(suite["_suite_path"])).parent / path).resolve()
    return path


def preflight(suite: Mapping[str, Any]) -> dict[str, Any]:
    if (REPOSITORY_ROOT / "test_access_grant.json").exists():
        raise ValueError("Final test access grant exists. Development commands fail closed.")
    base_suite = load_suite(_base_suite_path(suite))
    workspace = verify_workspace(base_suite, splits=("train", "val"))
    for split in ("train", "val"):
        config = _environment_config(base_suite, split)
        cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
        snapshot_dir = _resolve(REPOSITORY_ROOT, str(cfg["snapshot_dir"]))
        if not snapshot_dir.is_dir() or not any(snapshot_dir.glob("*.pkl")):
            raise FileNotFoundError(f"Warmup snapshot is missing for {split}: {snapshot_dir}")
    seeds = stochastic_seeds_from_suite(suite)
    return {
        "status": "PASS",
        "suite_id": suite["suite_id"],
        "methods": list(suite["methods"]),
        "seeds": list(seeds),
        "job_count": len(seeds) * len(suite["methods"]),
        "workspace": workspace,
        "test_locked": True,
    }


def _git_state() -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    return {"commit": commit, "dirty": bool(status)}


def _hardware() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _evaluate(
    env: Any,
    model: RecurrentQuotaActorCritic,
    *,
    seed: int,
    device: torch.device,
    max_steps: int | None,
    episode_id: str,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    observation, _ = env.reset(seed=seed, options={"episode_id": episode_id})
    total_capacity = float(sum(env.current_context.capacity_by_pool.values()))
    hidden = model.initial_hidden(1, device)
    steps = 0
    action_rows: list[dict[str, Any]] = []
    terminated = truncated = False
    model.eval()
    while not (terminated or truncated):
        if max_steps is not None and steps >= max_steps:
            env.terminate_external()
            truncated = True
            break
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            output = model.step(obs_tensor, hidden, deterministic=True)
        action = output["action"].squeeze(0).cpu().numpy().astype(np.float32)
        observation, reward, terminated, truncated, info = env.step(action)
        hidden = output["hidden"].detach()
        action_rows.append({
            "step": steps,
            "time": int(info["time"]),
            "policy_action": action.astype(float).tolist(),
            "executed_action": info["rl_action"],
            "reward": float(reward),
            "constraint_costs_raw": info["constraint_costs_raw"],
        })
        steps += 1
        if progress_callback is not None:
            progress_callback(steps)
    events = env.sim.ledger.events
    event_dicts = _EventDictView(events)
    reduced = reduce_replay_events(
        event_dicts,
        env.inner.episode_spec,
        expected_run_id=env.inner.run_id,
        expected_episode_id=episode_id,
    )
    publication = derive_publication_metrics(
        event_dicts,
        env.inner.episode_spec,
        total_capacity_gpu=total_capacity,
    )
    return {
        "steps": steps,
        "status": "EXTERNALLY_TRUNCATED" if truncated and not terminated else "COMPLETED",
        "reduced_metrics": reduced,
        "publication_metrics": publication,
        "runtime_diagnostics": env.runtime_diagnostics(),
        "actions": action_rows,
        "base_policy": env.base_manifest,
    }


def _assert_numeric_tree_close(
    left: Any,
    right: Any,
    *,
    path: str = "root",
    atol: float = 1e-9,
) -> None:
    """Compare nested metric payloads while preserving exact keys and lengths."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise AssertionError(
                f"Parity key mismatch at {path}: {sorted(left)} != {sorted(right)}"
            )
        for key in left:
            _assert_numeric_tree_close(
                left[key], right[key], path=f"{path}.{key}", atol=atol
            )
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise AssertionError(
                f"Parity length mismatch at {path}: {len(left)} != {len(right)}"
            )
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            _assert_numeric_tree_close(
                left_value, right_value, path=f"{path}[{index}]", atol=atol
            )
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not np.isclose(float(left), float(right), rtol=0.0, atol=atol):
            raise AssertionError(f"Parity value mismatch at {path}: {left} != {right}")
        return
    if left != right:
        raise AssertionError(f"Parity value mismatch at {path}: {left!r} != {right!r}")


def run_zero_action_parity(
    suite: Mapping[str, Any],
    *,
    method_ids: list[str],
    seed: int,
    steps: int,
    verify: bool,
) -> Path:
    """Compare each zero action RL wrapper with its direct deterministic base."""
    if steps < 1:
        raise ValueError("Parity requires at least one control step.")
    if verify:
        preflight(suite)
    base_suite_path = _base_suite_path(suite)
    base_suite = load_suite(base_suite_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    parity_root = REPOSITORY_ROOT / "runs" / "quota_rl_parity" / timestamp
    parity_root.mkdir(parents=True, exist_ok=False)
    rows = []
    direct_runs: dict[str, Path] = {}
    for method_id in method_ids:
        method = dict(suite["methods"][method_id])
        baseline_id = str(method["base_baseline"])
        if baseline_id not in direct_runs:
            direct_runs[baseline_id] = run_baseline(
                base_suite,
                baseline_id,
                "val",
                steps,
                seed,
                verify=False,
                progress_every_steps=steps,
                forecast_smoke_hyperparameters=True,
            )
        direct_dir = direct_runs[baseline_id]
        wrapper_dir = parity_root / method_id
        wrapper_dir.mkdir(parents=True, exist_ok=False)
        env, metadata = build_quota_rl_env(
            suite,
            method_id=method_id,
            split="val",
            seed=seed,
            run_dir=wrapper_dir,
            smoke_forecast=True,
        )
        model = RecurrentQuotaActorCritic(
            observation_dim=int(env.observation_space.shape[0]),
            hidden_size=int(suite["algorithm"]["hidden_size"]),
            constraint_count=3,
            action_dim=int(env.action_space.shape[0]),
            initial_log_std=float(suite["algorithm"]["initial_log_std"]),
        ).to(torch.device("cpu"))
        evaluation = _evaluate(
            env,
            model,
            seed=seed,
            device=torch.device("cpu"),
            max_steps=steps,
            episode_id=f"parity_{method_id}",
        )
        _atomic_json(wrapper_dir / "evaluation.json", evaluation)
        direct_actions = json.loads(
            (direct_dir / "control_actions.json").read_text(encoding="utf-8")
        )
        if len(direct_actions) != len(evaluation["actions"]):
            raise AssertionError(
                f"Action count mismatch for {method_id}: "
                f"{len(direct_actions)} != {len(evaluation['actions'])}"
            )
        maximum_action_error = 0.0
        maximum_effective_quota_error = 0.0
        for index, (direct_row, wrapped_row) in enumerate(
            zip(direct_actions, evaluation["actions"])
        ):
            policy_action = np.asarray(wrapped_row["policy_action"], dtype=np.float64)
            if not np.array_equal(policy_action, np.zeros_like(policy_action)):
                raise AssertionError(
                    f"Untrained actor was not zero at {method_id} step {index}."
                )
            direct_action = np.asarray(direct_row["action"], dtype=np.float64)
            wrapped_action = np.asarray(
                wrapped_row["executed_action"]["executed_simulator_action"],
                dtype=np.float64,
            )
            if str(method.get("action_mode", "legacy_residual")) == "srac_mix":
                direct_target = float(direct_row["effective_target_quota"])
                wrapped_target = float(wrapped_row["executed_action"]["target_quota"])
                error = abs(direct_target - wrapped_target)
                tolerance = 1e-4
                maximum_effective_quota_error = max(
                    maximum_effective_quota_error, error
                )
            else:
                error = float(np.max(np.abs(direct_action - wrapped_action)))
                tolerance = 1e-9
            maximum_action_error = max(maximum_action_error, error)
            if error > tolerance:
                raise AssertionError(
                    f"Action parity failed for {method_id} step {index}: {error}"
                )
        direct_metrics = json.loads(
            (direct_dir / "publication_metrics.json").read_text(encoding="utf-8")
        )
        _assert_numeric_tree_close(
            direct_metrics,
            evaluation["publication_metrics"],
            path=f"{method_id}.publication_metrics",
        )
        row = {
            "method_id": method_id,
            "base_baseline": baseline_id,
            "status": "PASS",
            "steps": steps,
            "seed": seed,
            "maximum_action_absolute_error": maximum_action_error,
            "maximum_effective_quota_absolute_error": maximum_effective_quota_error,
            "effective_quota_tolerance": 1e-4,
            "publication_metrics_equal": True,
            "direct_run_dir": str(direct_dir.relative_to(REPOSITORY_ROOT)),
            "wrapper_run_dir": str(wrapper_dir.relative_to(REPOSITORY_ROOT)),
            "metadata": metadata,
        }
        rows.append(row)
        _atomic_json(wrapper_dir / "parity.json", row)
    report = {
        "schema_version": "1.0",
        "suite_id": suite["suite_id"],
        "status": "PASS",
        "kind": "bounded_zero_action_parity",
        "split": "val",
        "seed": seed,
        "steps": steps,
        "methods": rows,
        "test_outcomes_accessed": False,
        "git": _git_state(),
    }
    _atomic_json(parity_root / "report.json", report)
    configured_result = suite.get("parity_result", "results/rl_zero_action_parity.json")
    result_path = (REPOSITORY_ROOT / str(configured_result)).resolve()
    if REPOSITORY_ROOT.resolve() not in result_path.parents:
        raise ValueError("Parity result must remain inside the repository.")
    _atomic_json(result_path, report)
    return result_path


def run_method(
    suite: Mapping[str, Any],
    *,
    method_id: str,
    seed: int,
    max_train_steps: int | None,
    max_eval_steps: int | None,
    smoke_forecast: bool,
    verify: bool,
    external_progress_file: Path | None = None,
    resume_run_dir: Path | None = None,
    checkpoint_every_updates: int | None = None,
    interrupt_after_updates: int | None = None,
    train_only: bool = False,
) -> Path:
    if verify:
        preflight(suite)
    _set_seed(seed)
    thread_manifest = _configure_threads(int(suite["resources"]["threads_per_worker"]))
    device = resolve_policy_device(str(suite["resources"]["policy_device"]))
    suite_sha256 = _sha256(Path(str(suite["_suite_path"])))
    algorithm = dict(suite["algorithm"])
    method_config = dict(suite["methods"][method_id])
    recovery_interval = int(
        algorithm.get("checkpoint_every_updates", 4)
        if checkpoint_every_updates is None
        else checkpoint_every_updates
    )
    if recovery_interval < 1:
        raise ValueError("checkpoint_every_updates must be positive.")
    recovery: dict[str, Any] | None = None
    if resume_run_dir is None:
        output_root = _resolve(
            Path(str(suite["_suite_path"])).parent,
            str(suite["environment"]["output_root"]),
        )
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = f"{method_id}_seed{seed}_{timestamp}"
        run_dir = output_root / method_id / f"seed{seed}" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = resume_run_dir.resolve()
        recovery = _load_recovery(run_dir / "recovery.pt")
        run_id = str(recovery["run_id"])
        if str(recovery["method_id"]) != method_id or int(recovery["seed"]) != seed:
            raise ValueError("Recovery checkpoint method or seed does not match request.")
        if str(recovery["suite_id"]) != str(suite["suite_id"]):
            raise ValueError("Recovery checkpoint suite ID does not match current suite.")
        if str(recovery["suite_sha256"]) != suite_sha256:
            raise ValueError("RL suite changed after checkpoint creation. Resume refused.")
        if dict(recovery["algorithm"]) != algorithm:
            raise ValueError("RL algorithm changed after checkpoint creation. Resume refused.")
        smoke_forecast = bool(recovery["smoke_forecast"])
    status_path = run_dir / "status.json"
    progress_path = run_dir / "progress.json"
    recovery_path = run_dir / "recovery.pt"

    def write_progress(payload: Mapping[str, Any]) -> None:
        enriched = {
            **dict(payload),
            "run_dir": str(run_dir),
            "recovery_checkpoint": str(recovery_path),
            "smoke_forecast": bool(smoke_forecast),
        }
        _atomic_json(progress_path, enriched)
        if external_progress_file is not None:
            _atomic_json(external_progress_file.resolve(), enriched)

    resumed_count = 0 if recovery is None else int(recovery.get("resumed_count", 0)) + 1
    eval_limit_requested = (
        -1 if max_eval_steps is None else int(max_eval_steps)
    ) if recovery is None else int(recovery["eval_limit"])
    bounded_smoke = (
        max_train_steps is not None or max_eval_steps is not None
    ) if recovery is None else bool(recovery["bounded_smoke"])
    _atomic_json(status_path, {
        "state": "RUNNING",
        "run_id": run_id,
        "resumed_count": resumed_count,
    })
    if recovery is None:
        write_progress({
            "state": "INITIALIZING",
            "run_id": run_id,
            "method_id": method_id,
            "seed": seed,
            "completed_train_steps": 0,
            "planned_train_steps": None,
            "completed_eval_steps": 0,
            "planned_eval_steps": None,
            "resumed_count": resumed_count,
        })
    started = time.perf_counter()
    elapsed_before = 0.0 if recovery is None else float(recovery["elapsed_wall_seconds"])

    def build_model_and_trainer(observation_dim: int, action_dim: int) -> tuple[
        RecurrentQuotaActorCritic, ConstrainedRecurrentPPO
    ]:
        model_value = RecurrentQuotaActorCritic(
            observation_dim=observation_dim,
            hidden_size=int(algorithm["hidden_size"]),
            constraint_count=3,
            action_dim=int(action_dim),
            initial_log_std=float(algorithm["initial_log_std"]),
        ).to(device)
        trainer_value = ConstrainedRecurrentPPO(
            model_value,
            learning_rate=float(algorithm["learning_rate"]),
            clip_epsilon=float(algorithm["clip_epsilon"]),
            value_coefficient=float(algorithm["value_coefficient"]),
            entropy_coefficient=float(algorithm["entropy_coefficient"]),
            max_grad_norm=float(algorithm["max_grad_norm"]),
            ppo_epochs=int(algorithm["ppo_epochs"]),
            dual_learning_rate=float(algorithm["dual_learning_rate"]),
            dual_mode=str(method_config.get("dual_mode", "integral")),
            pid_kp=float(algorithm.get("pid_kp", 0.05)),
            pid_ki=float(algorithm.get("pid_ki", 0.005)),
            pid_kd=float(algorithm.get("pid_kd", 0.02)),
            pid_ema_alpha=float(algorithm.get("pid_ema_alpha", 0.9)),
        )
        return model_value, trainer_value

    try:
        checkpoint_path = run_dir / "checkpoint.pt"
        if recovery is not None and str(recovery["phase"]) == "TRAINED":
            train_metadata = dict(recovery["train_metadata"])
            train_summary = dict(recovery["train_summary"])
            train_limit = int(recovery["train_limit"])
            completed_steps = int(recovery["completed_steps"])
            eval_limit_saved = int(recovery["eval_limit"])
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            model, trainer = build_model_and_trainer(
                int(checkpoint["model"]["observation_dim"]),
                int(checkpoint["model"].get("action_dim", 1)),
            )
            trainer.load_state_dict(checkpoint["trainer"])
            train_env = None
        else:
            if recovery is None:
                train_env, train_metadata = build_quota_rl_env(
                    suite,
                    method_id=method_id,
                    split="train",
                    seed=seed,
                    run_dir=run_dir / "train",
                    smoke_forecast=smoke_forecast,
                )
                observation, _ = train_env.reset(
                    seed=seed, options={"episode_id": f"{run_id}_train"}
                )
                # Forecast initialization may touch global framework RNG state.
                # Restore the paired run seed immediately before policy creation.
                _set_seed(seed)
                model, trainer = build_model_and_trainer(
                    int(observation.shape[0]), int(train_env.action_space.shape[0])
                )
                hidden = model.initial_hidden(1, device)
                train_limit = (
                    int(train_metadata["expected_steps"])
                    if max_train_steps is None
                    else int(max_train_steps)
                )
                completed_steps = 0
                update_rows: list[dict[str, Any]] = []
                train_action_rows: list[dict[str, Any]] = []
                terminal = False
            else:
                if str(recovery["phase"]) != "TRAINING":
                    raise ValueError(f"Unsupported recovery phase: {recovery['phase']}")
                train_env = recovery["train_env"]
                train_metadata = dict(recovery["train_metadata"])
                observation = np.asarray(recovery["observation"], dtype=np.float32)
                saved_action_dim = int(
                    recovery.get("action_dim", train_env.action_space.shape[0])
                )
                model, trainer = build_model_and_trainer(
                    int(observation.shape[0]), saved_action_dim
                )
                trainer.load_state_dict(recovery["trainer"])
                hidden = recovery["hidden"].to(device)
                train_limit = int(recovery["train_limit"])
                completed_steps = int(recovery["completed_steps"])
                update_rows = list(recovery["update_rows"])
                train_action_rows = list(recovery["train_action_rows"])
                terminal = bool(recovery["terminal"])
                _restore_rng_state(recovery["rng_state"])

            while completed_steps < train_limit and not terminal:
                count = min(int(algorithm["rollout_steps"]), train_limit - completed_steps)
                batch, observation, _ = collect_rollout(
                    train_env,
                    model,
                    observation,
                    hidden,
                    rollout_steps=count,
                    device=device,
                    gamma=float(algorithm["gamma"]),
                    gae_lambda=float(algorithm["gae_lambda"]),
                )
                for offset, transition in enumerate(batch.infos):
                    train_action_rows.append({
                        "step": completed_steps + offset,
                        "time": int(transition["time"]),
                        "executed_action": transition["rl_action"],
                        "reward": float(transition["training_reward"]),
                        "constraint_costs_raw": transition["constraint_costs_raw"],
                    })
                update = trainer.update(batch)
                hidden = trainer.recompute_hidden(
                    batch.observations,
                    burn_in_steps=int(algorithm["burn_in_steps"]),
                )
                completed_steps += batch.steps
                terminal = batch.terminal
                update_rows.append(update)
                write_progress({
                    "state": "TRAINING" if completed_steps < train_limit and not terminal else "EVALUATING",
                    "run_id": run_id,
                    "method_id": method_id,
                    "seed": seed,
                    "completed_train_steps": completed_steps,
                    "planned_train_steps": train_limit,
                    "completed_eval_steps": 0,
                    "planned_eval_steps": None,
                    "progress_fraction": completed_steps / max(1, train_limit),
                    "latest_update": update,
                    "resumed_count": resumed_count,
                })
                must_interrupt = (
                    interrupt_after_updates is not None
                    and trainer.updates >= int(interrupt_after_updates)
                )
                if (
                    trainer.updates == 1
                    or trainer.updates % recovery_interval == 0
                    or completed_steps >= train_limit
                    or terminal
                    or must_interrupt
                ):
                    _atomic_torch_save(recovery_path, {
                        "schema_version": RECOVERY_SCHEMA,
                        "phase": "TRAINING",
                        "run_id": run_id,
                        "method_id": method_id,
                        "seed": seed,
                        "suite_id": suite["suite_id"],
                        "suite_sha256": suite_sha256,
                        "algorithm": algorithm,
                        "smoke_forecast": bool(smoke_forecast),
                        "train_env": train_env,
                        "train_metadata": train_metadata,
                        "observation": observation,
                        "hidden": hidden,
                        "action_dim": int(train_env.action_space.shape[0]),
                        "trainer": trainer.state_dict(),
                        "train_limit": train_limit,
                        "eval_limit": eval_limit_requested,
                        "bounded_smoke": bounded_smoke,
                        "completed_steps": completed_steps,
                        "terminal": terminal,
                        "update_rows": update_rows,
                        "train_action_rows": train_action_rows,
                        "rng_state": _rng_state(),
                        "elapsed_wall_seconds": elapsed_before + time.perf_counter() - started,
                        "resumed_count": resumed_count,
                    })
                if must_interrupt:
                    raise RuntimeError("Planned recovery interruption after checkpoint.")

            if not terminal:
                train_env.terminate_external()
            train_summary = {
                "base_policy": train_env.base_manifest,
                "risk_budgets": train_env.risk_budgets.manifest(),
                "safety": train_env.safety.manifest(),
            }
            checkpoint = {
                "schema_version": "safe-residual-quota-rl-1.0",
                "method_id": method_id,
                "seed": seed,
                "suite_id": suite["suite_id"],
                "suite_sha256": suite_sha256,
                "algorithm": algorithm,
                "model": model.manifest(),
                "observation_features": list(train_env.observation_feature_names),
                "trainer": trainer.state_dict(),
                **train_summary,
                "action_transform": str(method_config.get("action_mode", "legacy_residual")),
                "test_outcomes_accessed": False,
                "git": _git_state(),
            }
            _atomic_torch_save(checkpoint_path, checkpoint)
            _atomic_json(run_dir / "train_updates.json", {"updates": update_rows})
            _atomic_json(run_dir / "train_actions.json", {"actions": train_action_rows})
            eval_limit_saved = eval_limit_requested
            _atomic_torch_save(recovery_path, {
                "schema_version": RECOVERY_SCHEMA,
                "phase": "TRAINED",
                "run_id": run_id,
                "method_id": method_id,
                "seed": seed,
                "suite_id": suite["suite_id"],
                "suite_sha256": suite_sha256,
                "algorithm": algorithm,
                "smoke_forecast": bool(smoke_forecast),
                "train_metadata": train_metadata,
                "train_summary": train_summary,
                "train_limit": train_limit,
                "completed_steps": completed_steps,
                "eval_limit": eval_limit_saved,
                "bounded_smoke": bounded_smoke,
                "elapsed_wall_seconds": elapsed_before + time.perf_counter() - started,
                "resumed_count": resumed_count,
            })

        if train_only:
            elapsed = elapsed_before + time.perf_counter() - started
            manifest = {
                "schema_version": "1.0",
                "run_id": run_id,
                "method_id": method_id,
                "method": dict(suite["methods"][method_id]),
                "seed": seed,
                "seed_master": int(suite["randomness"]["master_seed"]),
                "seed_derivation": suite["randomness"]["stochastic_seed_derivation"],
                "train": train_metadata,
                "validation": None,
                "algorithm": algorithm,
                "model": model.manifest(),
                "base_policy": train_summary["base_policy"],
                "test_outcomes_accessed": False,
                "git": _git_state(),
                "runtime": {
                    **_hardware(),
                    **thread_manifest,
                    "policy_device": str(device),
                    "simulator_device": "cpu",
                    "elapsed_wall_seconds": elapsed,
                    "resumed_count": resumed_count,
                    "recovery_checkpoint": str(recovery_path),
                },
                "bounded_smoke": True,
                "train_only": True,
            }
            _atomic_json(run_dir / "manifest.json", manifest)
            _atomic_json(status_path, {
                "state": "TRAIN_ONLY_COMPLETED",
                "run_id": run_id,
                "method_id": method_id,
                "seed": seed,
                "checkpoint": str(checkpoint_path),
                "elapsed_wall_seconds": elapsed,
                "resumed_count": resumed_count,
            })
            write_progress({
                "state": "TRAIN_ONLY_COMPLETED",
                "run_id": run_id,
                "method_id": method_id,
                "seed": seed,
                "completed_train_steps": completed_steps,
                "planned_train_steps": train_limit,
                "completed_eval_steps": 0,
                "planned_eval_steps": 0,
                "progress_fraction": 1.0,
                "elapsed_wall_seconds": elapsed,
                "resumed_count": resumed_count,
            })
            return run_dir

        eval_env, eval_metadata = build_quota_rl_env(
            suite,
            method_id=method_id,
            split="val",
            seed=seed,
            run_dir=run_dir / "val",
            smoke_forecast=smoke_forecast,
        )
        expected_eval_steps = int(eval_metadata["expected_steps"])
        eval_limit = expected_eval_steps if eval_limit_saved < 0 else eval_limit_saved

        def report_evaluation(eval_steps: int) -> None:
            progress_every = int(suite["resources"]["progress_every_steps"])
            if eval_steps % progress_every != 0 and eval_steps != eval_limit:
                return
            write_progress({
                "state": "EVALUATING",
                "run_id": run_id,
                "method_id": method_id,
                "seed": seed,
                "completed_train_steps": completed_steps,
                "planned_train_steps": train_limit,
                "completed_eval_steps": eval_steps,
                "planned_eval_steps": eval_limit,
                "progress_fraction": (
                    completed_steps + eval_steps
                ) / max(1, train_limit + eval_limit),
                "resumed_count": resumed_count,
            })
        evaluation = _evaluate(
            eval_env,
            model,
            seed=seed,
            device=device,
            max_steps=None if eval_limit == expected_eval_steps else eval_limit,
            episode_id=f"{run_id}_val",
            progress_callback=report_evaluation,
        )
        elapsed = elapsed_before + time.perf_counter() - started
        manifest = {
            "schema_version": "1.0",
            "run_id": run_id,
            "method_id": method_id,
            "method": dict(suite["methods"][method_id]),
            "seed": seed,
            "seed_master": int(suite["randomness"]["master_seed"]),
            "seed_derivation": suite["randomness"]["stochastic_seed_derivation"],
            "train": train_metadata,
            "validation": eval_metadata,
            "algorithm": algorithm,
            "model": model.manifest(),
            "base_policy": train_summary["base_policy"],
            "test_outcomes_accessed": False,
            "git": _git_state(),
            "runtime": {
                **_hardware(),
                **thread_manifest,
                "policy_device": str(device),
                "simulator_device": "cpu",
                "elapsed_wall_seconds": elapsed,
                "resumed_count": resumed_count,
                "recovery_checkpoint": str(recovery_path),
            },
            "bounded_smoke": bounded_smoke,
        }
        _atomic_json(run_dir / "manifest.json", manifest)
        _atomic_json(run_dir / "validation.json", evaluation)
        final_state = "SMOKE_COMPLETED" if manifest["bounded_smoke"] else "COMPLETED"
        _atomic_json(status_path, {
            "state": final_state,
            "run_id": run_id,
            "method_id": method_id,
            "seed": seed,
            "checkpoint": str(checkpoint_path),
            "validation": str(run_dir / "validation.json"),
            "elapsed_wall_seconds": elapsed,
            "resumed_count": resumed_count,
        })
        write_progress({
            "state": final_state,
            "run_id": run_id,
            "method_id": method_id,
            "seed": seed,
            "completed_train_steps": completed_steps,
            "planned_train_steps": train_limit,
            "completed_eval_steps": evaluation["steps"],
            "planned_eval_steps": eval_limit,
            "progress_fraction": 1.0,
            "elapsed_wall_seconds": elapsed,
            "resumed_count": resumed_count,
        })
        return run_dir
    except Exception as exc:
        previous_progress = (
            json.loads(progress_path.read_text(encoding="utf-8"))
            if progress_path.is_file()
            else {}
        )
        failure = {
            **previous_progress,
            "state": "FAILED",
            "run_id": run_id,
            "method_id": method_id,
            "seed": seed,
            "error": f"{type(exc).__name__}: {exc}",
            "resumed_count": resumed_count,
        }
        _atomic_json(status_path, failure)
        write_progress(failure)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=BENCHMARK_ROOT / "rl_suite.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list")
    subparsers.add_parser("prepare")
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--methods", nargs="+")
    smoke.add_argument("--train-steps", type=int, default=4)
    smoke.add_argument("--eval-steps", type=int, default=4)
    run = subparsers.add_parser("run")
    run.add_argument("--method", required=True)
    run.add_argument("--seed", type=int)
    run.add_argument("--max-train-steps", type=int)
    run.add_argument("--max-eval-steps", type=int)
    run.add_argument("--smoke-forecast", action="store_true")
    run.add_argument("--progress-file", type=Path)
    run.add_argument("--checkpoint-every-updates", type=int)
    run.add_argument("--interrupt-after-updates", type=int, help=argparse.SUPPRESS)
    train_only = subparsers.add_parser("train-only")
    train_only.add_argument("--method", required=True)
    train_only.add_argument("--seed", type=int)
    train_only.add_argument("--steps", type=int, default=256)
    train_only.add_argument("--smoke-forecast", action="store_true")
    resume = subparsers.add_parser("resume")
    resume.add_argument("--run-dir", type=Path, required=True)
    resume.add_argument("--progress-file", type=Path)
    parity = subparsers.add_parser("parity")
    parity.add_argument("--methods", nargs="+")
    parity.add_argument("--steps", type=int, default=4)
    parity.add_argument("--seed", type=int)
    args = parser.parse_args()
    suite = load_rl_suite(args.suite)
    method_choices = tuple(suite["methods"])
    if hasattr(args, "method") and args.method is not None and args.method not in method_choices:
        raise SystemExit(f"Unknown method {args.method!r}. Choose from {list(method_choices)}")
    if hasattr(args, "methods"):
        if args.methods is None:
            args.methods = list(method_choices)
        unknown = sorted(set(args.methods) - set(method_choices))
        if unknown:
            raise SystemExit(f"Unknown methods: {unknown}. Choose from {list(method_choices)}")
    seeds = stochastic_seeds_from_suite(suite)
    if args.command == "list":
        print(json.dumps({
            "suite_id": suite["suite_id"],
            "algorithm": suite["algorithm"],
            "methods": suite["methods"],
            "seeds": list(seeds),
            "test_policy": suite["environment"]["test_policy"],
        }, indent=2, sort_keys=True))
    elif args.command == "prepare":
        result = preflight(suite)
        result["full_commands"] = [
            f"python benchmark/rl_run.py run --method {method} --seed {seed}"
            for method in suite["methods"]
            for seed in seeds
        ]
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "smoke":
        preflight(suite)
        run_dirs = []
        for method in args.methods:
            run_dirs.append(str(run_method(
                suite,
                method_id=method,
                seed=seeds[0],
                max_train_steps=args.train_steps,
                max_eval_steps=args.eval_steps,
                smoke_forecast=True,
                verify=False,
                external_progress_file=None,
            )))
        print(json.dumps({"status": "PASS", "run_dirs": run_dirs}, indent=2))
    elif args.command == "parity":
        seed = seeds[0] if args.seed is None else int(args.seed)
        if seed not in seeds:
            raise SystemExit(f"Seed {seed} is not generated by master seed 17: {list(seeds)}")
        result_path = run_zero_action_parity(
            suite,
            method_ids=list(args.methods),
            seed=seed,
            steps=int(args.steps),
            verify=True,
        )
        print(json.dumps({"status": "PASS", "report": str(result_path)}, indent=2))
    elif args.command == "resume":
        progress_path = args.run_dir.resolve() / "progress.json"
        if not progress_path.is_file():
            raise SystemExit(f"Resume progress file is missing: {progress_path}")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        method_id = str(progress["method_id"])
        seed = int(progress["seed"])
        if method_id not in method_choices:
            raise SystemExit(f"Unknown method in recovery progress: {method_id}")
        if seed not in seeds:
            raise SystemExit(f"Seed {seed} is not generated by master seed 17: {list(seeds)}")
        run_dir = run_method(
            suite,
            method_id=method_id,
            seed=seed,
            max_train_steps=None,
            max_eval_steps=None,
            smoke_forecast=bool(progress.get("smoke_forecast", False)),
            verify=True,
            external_progress_file=args.progress_file,
            resume_run_dir=args.run_dir,
        )
        print(json.dumps({"status": "PASS", "run_dir": str(run_dir)}, indent=2))
    elif args.command == "train-only":
        seed = seeds[0] if args.seed is None else int(args.seed)
        if seed not in seeds:
            raise SystemExit(f"Seed {seed} is not generated by master seed 17: {list(seeds)}")
        if args.steps < 2:
            raise SystemExit("Bounded training requires at least two steps.")
        run_dir = run_method(
            suite,
            method_id=args.method,
            seed=seed,
            max_train_steps=int(args.steps),
            max_eval_steps=None,
            smoke_forecast=bool(args.smoke_forecast),
            verify=True,
            train_only=True,
        )
        print(json.dumps({"status": "PASS", "run_dir": str(run_dir)}, indent=2))
    else:
        seed = seeds[0] if args.seed is None else int(args.seed)
        if seed not in seeds:
            raise SystemExit(f"Seed {seed} is not generated by master seed 17: {list(seeds)}")
        if args.max_train_steps is not None and args.max_train_steps < 2:
            raise SystemExit("Bounded training requires at least two steps.")
        if args.max_eval_steps is not None and args.max_eval_steps < 1:
            raise SystemExit("Bounded evaluation requires at least one step.")
        run_dir = run_method(
            suite,
            method_id=args.method,
            seed=seed,
            max_train_steps=args.max_train_steps,
            max_eval_steps=args.max_eval_steps,
            smoke_forecast=bool(args.smoke_forecast),
            verify=True,
            external_progress_file=args.progress_file,
            checkpoint_every_updates=args.checkpoint_every_updates,
            interrupt_after_updates=args.interrupt_after_updates,
        )
        print(json.dumps({"status": "PASS", "run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
