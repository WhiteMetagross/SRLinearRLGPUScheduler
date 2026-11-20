"""Run auditable fixed baselines on the unified A100 control environment."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


BENCHMARK_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = BENCHMARK_ROOT.parent
ENVIRONMENT_ROOT = REPOSITORY_ROOT / "environment"
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(ENVIRONMENT_ROOT / "src"))

from research_sim.cli import (  # noqa: E402
    finalize_run_artifacts,
    get_system_manifest,
    write_run_artifacts,
)
from research_sim.control_env import UnifiedControlEnv  # noqa: E402
from research_sim.measurement import reduce_replay_events  # noqa: E402
from benchmark.protocol_helpers import (  # noqa: E402
    _paper_manifest_for,
    _validate_trace_hashes,
    build_spec_from_dict,
    fit_scenarios_from_executed_history,
    load_yaml,
    validate_fit_report,
)
from research_sim.trace import TraceStore  # noqa: E402

from benchmark.policies import BaselinePolicy, build_policy  # noqa: E402
from benchmark.forecast_interventions import (  # noqa: E402
    activate_forecast_at_score_boundary,
    build_forecast_intervention,
)
from benchmark.publication_metrics import derive_publication_metrics  # noqa: E402
from benchmark.seeds import (  # noqa: E402
    DETERMINISTIC_SEED_DERIVATION,
    STOCHASTIC_SEED_DERIVATION,
    deterministic_seed_from_suite,
    stochastic_seeds_from_suite,
)


def _resolve(base: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (base / path).resolve()


def load_suite(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != "1.0":
        raise ValueError("Benchmark suite requires schema_version '1.0'.")
    if not isinstance(raw.get("baselines"), dict) or not raw["baselines"]:
        raise ValueError("Benchmark suite requires a nonempty baselines mapping.")
    raw["_suite_path"] = str(resolved)
    deterministic_seed_from_suite(raw)
    stochastic_seeds_from_suite(raw)
    return raw


def _suite_base(suite: dict[str, Any]) -> Path:
    return Path(suite["_suite_path"]).parent


def _environment_config(suite: dict[str, Any], split: str) -> Path:
    configs = suite["environment"]["configs"]
    if split not in configs:
        raise ValueError(f"Split {split!r} is not exposed by this benchmark suite.")
    return _resolve(_suite_base(suite), str(configs[split]))


def _configure_numeric_threads(suite: dict[str, Any]) -> dict[str, Any]:
    """Enforce and report the suite's per-process numerical thread budget."""
    declared = int(suite["resources"]["threads_per_worker"])
    if declared < 1:
        raise ValueError("resources.threads_per_worker must be positive.")
    thread_value = str(declared)
    names = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    for name in names:
        os.environ[name] = thread_value
    torch.set_num_threads(declared)
    actual_torch = int(torch.get_num_threads())
    if actual_torch != declared:
        raise RuntimeError(
            f"PyTorch numerical threads differ from suite: {actual_torch} != {declared}."
        )
    return {
        "declared_threads_per_worker": declared,
        "torch_num_threads": actual_torch,
        "environment": {name: os.environ[name] for name in names},
    }


def _run_checked(command: list[str]) -> None:
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


class _EventDictView(Sequence[dict[str, Any]]):
    """Expose ledger events as dictionaries without materializing another list."""

    def __init__(self, events: Sequence[Any]):
        self._events = events
        self._last = self._convert(events[-1]) if events else None

    @staticmethod
    def _convert(event: Any) -> dict[str, Any]:
        return event if isinstance(event, dict) else event.to_dict()

    def __len__(self) -> int:
        return len(self._events)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index == -1 or index == len(self._events) - 1:
            if self._last is None:
                raise IndexError(index)
            return self._last
        return self._convert(self._events[index])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        last_index = len(self._events) - 1
        for index, event in enumerate(self._events):
            yield self._last if index == last_index else self._convert(event)


def verify_workspace(
    suite: dict[str, Any],
    splits: tuple[str, ...] = ("train", "val"),
) -> dict[str, Any]:
    if suite["environment"].get("test_policy") != "locked":
        raise ValueError("Benchmark suite must keep the final test split locked.")
    if (REPOSITORY_ROOT / "test_access_grant.json").exists():
        raise ValueError("Test access grant exists. Baseline preparation must not inspect test outcomes.")

    _run_checked([sys.executable, "environment/tools/verify_freeze.py"])
    checked = []
    for split in splits:
        config = _environment_config(suite, split)
        _run_checked([
            sys.executable,
            "-m",
            "benchmark.protocol_helpers",
            "preflight",
            "--config",
            str(config),
        ])
        checked.append(str(config.relative_to(REPOSITORY_ROOT)))
    return {
        "status": "PASS",
        "suite_id": suite["suite_id"],
        "configs": checked,
        "test_locked": True,
    }


def _baseline_action(suite: dict[str, Any], baseline_id: str) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        baseline = dict(suite["baselines"][baseline_id])
    except KeyError as exc:
        raise ValueError(f"Unknown baseline {baseline_id!r}.") from exc
    if baseline.get("status") != "ready":
        artifact = baseline.get("action_artifact")
        raise ValueError(
            f"Baseline {baseline_id!r} is {baseline.get('status')!r}. "
            f"Fit it on train and write {artifact!r} before evaluation."
        )
    action = np.asarray(baseline.get("action"), dtype=np.float32)
    if action.shape != (7,) or not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1.0):
        raise ValueError(f"Baseline {baseline_id!r} has an invalid fixed action.")
    return action, baseline


def _baseline_policy(
    suite: dict[str, Any],
    baseline_id: str,
    *,
    allow_smoke_artifact: bool,
) -> tuple[BaselinePolicy, dict[str, Any], dict[str, Any]]:
    try:
        baseline = dict(suite["baselines"][baseline_id])
    except KeyError as exc:
        raise ValueError(f"Unknown baseline {baseline_id!r}.") from exc
    status = baseline.get("status")
    if status not in {"ready", "requires_train_fit", "requires_validation_selection"}:
        raise ValueError(f"Baseline {baseline_id!r} has unsupported status {status!r}.")
    try:
        policy, policy_manifest = build_policy(
            baseline,
            suite_base=_suite_base(suite),
            allow_smoke_artifact=allow_smoke_artifact,
        )
    except (FileNotFoundError, ValueError) as exc:
        if status in {"requires_train_fit", "requires_validation_selection"}:
            raise ValueError(
                f"Baseline {baseline_id!r} is {status!r}. {exc}"
            ) from exc
        raise
    return policy, baseline, policy_manifest


def _set_process_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _fit_required_duration_estimator(
    baseline: dict[str, Any],
    cfg: dict[str, Any],
    spec: Any,
    view: Any,
    run_dir: Path,
) -> tuple[Any | None, dict[str, Any] | None]:
    """Fit a causal duration estimator only for baselines that declare it."""
    if not bool(baseline.get("requires_duration_estimator", False)):
        return None, None
    fit_end = int(spec.score_start)
    if fit_end > int(spec.score_start):
        raise ValueError(
            f"Duration fit cutoff {fit_end} exceeds score start {spec.score_start}."
        )
    estimator = fit_scenarios_from_executed_history(
        cfg,
        spec,
        fit_end,
        run_dir=str(run_dir),
    )
    validate_fit_report(estimator, require_spot=True)
    spot_rows = view.job_df[view.job_df["job_type"] == "Spot"]
    shapes = {
        (
            str(row.gpu_model),
            int(row.worker_num),
            int(row.gpu_request),
        )
        for row in spot_rows.itertuples()
    }
    estimates = {
        shape: int(estimator.estimate_duration(
            shape[1],
            shape[2],
            is_spot=True,
            gpu_model=shape[0],
        ))
        for shape in sorted(shapes)
    }
    distinct = sorted(set(estimates.values()))
    if len(distinct) < 2:
        raise ValueError(
            "Duration-ranked baseline requires at least two distinct causal Spot estimates."
        )
    report = {
        "fit_cutoff": fit_end,
        "fit_report": estimator.fit_report,
        "shape_count": len(estimates),
        "distinct_estimate_count": len(distinct),
        "minimum_estimate_seconds": min(distinct),
        "maximum_estimate_seconds": max(distinct),
    }
    return estimator, report


def _run_directory(
    output_root: Path,
    baseline_id: str,
    split: str,
    run_seed: int,
    run_id: str,
) -> Path:
    """Give each seed an isolated parent for the replay audit prefix cache."""
    return output_root / baseline_id / split / f"seed{int(run_seed)}" / run_id


def _git_state() -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    return {"commit": commit, "dirty": bool(status)}


def run_baseline(
    suite: dict[str, Any],
    baseline_id: str,
    split: str,
    max_control_steps: int | None,
    run_seed: int,
    verify: bool = True,
    progress_file: Path | None = None,
    progress_every_steps: int = 12,
    forecast_smoke_hyperparameters: bool = True,
) -> Path:
    if progress_every_steps < 1:
        raise ValueError("progress_every_steps must be positive.")
    numeric_threads = _configure_numeric_threads(suite)
    if verify:
        verify_workspace(suite, splits=(split,))
    policy, baseline, policy_manifest = _baseline_policy(
        suite,
        baseline_id,
        allow_smoke_artifact=max_control_steps is not None,
    )
    config_path = _environment_config(suite, split)
    cfg = load_yaml(str(config_path))
    base_spec = build_spec_from_dict(cfg)
    spec = dataclasses.replace(base_spec, policy_seed=int(run_seed))
    _set_process_seed(int(run_seed))
    paper_manifest = _paper_manifest_for(str(config_path), cfg, spec)
    if paper_manifest is None:
        raise ValueError("Benchmark runs require a frozen paper config.")
    if spec.split_id == "test":
        raise ValueError("The final test split is not available during baseline development.")

    trace_dir = _resolve(REPOSITORY_ROOT, str(cfg["trace_dir"]))
    view = TraceStore.get_instance(str(trace_dir)).create_episode_view(spec)
    _validate_trace_hashes(paper_manifest, view)
    snapshot_dir = _resolve(REPOSITORY_ROOT, str(cfg["snapshot_dir"]))

    output_root = _resolve(_suite_base(suite), str(suite["environment"]["output_root"]))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{baseline_id}_{split}_seed{spec.policy_seed}_{timestamp}"
    run_dir = _run_directory(
        output_root,
        baseline_id,
        split,
        spec.policy_seed,
        run_id,
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    duration_estimator, duration_fit_report = _fit_required_duration_estimator(
        baseline,
        cfg,
        spec,
        view,
        run_dir,
    )
    progress_paths = [run_dir / "progress.json"]
    if progress_file is not None and progress_file.resolve() != progress_paths[0].resolve():
        progress_paths.append(progress_file.resolve())
    expected_steps = int(math.ceil(
        (spec.score_end - spec.score_start) / spec.control_interval_seconds
    ))
    progress_started = datetime.now(timezone.utc).isoformat()

    def record_progress(state: str, *, current_time: int, completed_steps: int) -> None:
        bounded_steps = min(completed_steps, expected_steps)
        payload = {
            "schema_version": "1.0",
            "state": state,
            "baseline_id": baseline_id,
            "split": split,
            "run_seed": int(run_seed),
            "run_id": run_id,
            "run_dir": str(run_dir),
            "started_at": progress_started,
            "heartbeat_at": datetime.now(timezone.utc).isoformat(),
            "score_start": int(spec.score_start),
            "score_end": int(spec.score_end),
            "current_time": int(current_time),
            "completed_steps": int(completed_steps),
            "expected_steps": expected_steps,
            "progress_fraction": (
                bounded_steps / expected_steps if expected_steps else 1.0
            ),
        }
        for path in progress_paths:
            _write_json_atomic(path, payload)

    started = time.perf_counter()
    env = UnifiedControlEnv(
        spec,
        view,
        scheduler_profile=str(baseline.get("scheduler_profile", "native_gfs")),
        duration_estimator=duration_estimator,
        use_warmup_snapshot=True,
        snapshot_dir=str(snapshot_dir),
        snapshot_missing_mode="raise",
        log_dir=str(run_dir / "logs"),
        run_id=run_id,
    )
    _, reset_info = env.reset(seed=spec.policy_seed, options={"episode_id": run_id})
    forecast_provider = None
    forecast_declaration = baseline.get("forecast_intervention")
    if forecast_declaration is not None:
        if not isinstance(forecast_declaration, dict):
            raise ValueError("forecast_intervention must be a mapping.")
        forecast_provider = build_forecast_intervention(
            forecast_declaration,
            suite_base=_suite_base(suite),
            simulator=env.sim,
            run_dir=run_dir,
            seed=int(run_seed),
            smoke=(
                max_control_steps is not None and forecast_smoke_hyperparameters
            ),
        )
        bind_provider_fn = getattr(policy, "bind_forecast_provider", None)
        if callable(bind_provider_fn):
            bind_provider_fn(forecast_provider)
        spec = dataclasses.replace(
            spec,
            forecast_mode="external_causal",
            forecast_provider_id=forecast_provider.provider_id,
            forecast_provider_version=forecast_provider.provider_version,
        )
        env.sim.spec = spec
        env.sim.quota_provider = forecast_provider
        env.episode_spec = spec
        env.current_context = activate_forecast_at_score_boundary(
            env.sim, forecast_provider
        )
    if env.current_context is None:
        raise ValueError("Baseline reset did not reach a control decision.")
    total_capacity_gpu = float(sum(env.current_context.capacity_by_pool.values()))
    record_progress(
        "RUNNING",
        current_time=int(env.current_context.time),
        completed_steps=0,
    )
    terminated = bool(reset_info["terminated"])
    truncated = bool(reset_info["truncated"])
    steps = 0
    reward_sum = 0.0
    action_rows = []
    while not (terminated or truncated):
        if max_control_steps is not None and steps >= max_control_steps:
            env.sim.terminate(
                reason="max_control_steps_reached",
                externally_truncated=True,
            )
            truncated = True
            break
        decision_time = int(env.current_context.time)
        pools = tuple(env.current_context.capacity_by_pool)
        if len(pools) != 1:
            raise ValueError("Current benchmark action audit requires one GPU pool.")
        pool = pools[0]
        decision_capacity = float(env.current_context.capacity_by_pool[pool])
        decision_base_quota = float(min(
            env.current_context.base_quota_by_pool[pool].values()
        ))
        action = policy.action(env.current_context)
        effective_target_quota = float(np.clip(
            decision_base_quota + float(action[0]) * decision_capacity,
            0.0,
            decision_capacity,
        ))
        decision_diagnostics_fn = getattr(policy, "decision_diagnostics", None)
        decision_diagnostics = (
            decision_diagnostics_fn() if callable(decision_diagnostics_fn) else None
        )
        _, reward, terminated, truncated, info = env.step(action)
        observe_transition_fn = getattr(policy, "observe_transition", None)
        if callable(observe_transition_fn):
            observe_transition_fn(info)
        reward_sum += float(reward)
        action_row = {
            "step": steps,
            "decision_time": decision_time,
            "action": action.tolist(),
            "base_quota": decision_base_quota,
            "effective_target_quota": effective_target_quota,
            "reward": float(reward),
            "reward_components": dict(info["reward_components"]),
            "next_time": int(info["time"]),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        if decision_diagnostics is not None:
            action_row["policy_decision"] = decision_diagnostics
        action_rows.append(action_row)
        steps += 1
        if steps % progress_every_steps == 0 or terminated or truncated:
            record_progress(
                "RUNNING" if not (terminated or truncated) else "FINALIZING",
                current_time=int(info["time"]),
                completed_steps=steps,
            )

    events = env.sim.ledger.events
    event_dicts = _EventDictView(events)
    reduced = reduce_replay_events(
        event_dicts,
        spec,
        expected_run_id=run_id,
        expected_episode_id=run_id,
    )
    publication_metrics = derive_publication_metrics(
        event_dicts,
        spec,
        total_capacity_gpu=total_capacity_gpu,
    )
    elapsed = time.perf_counter() - started
    admission_diagnostics = {
        "queue_scans": int(getattr(env.sim, "admission_queue_scans", 0)),
        "reordered_scans": int(getattr(env.sim, "admission_reordered_scans", 0)),
        "duration_nonconstant_scans": int(
            getattr(env.sim, "admission_duration_nonconstant_scans", 0)
        ),
    }
    runtime_diagnostics_fn = getattr(policy, "runtime_diagnostics", None)
    policy_runtime_diagnostics = (
        runtime_diagnostics_fn() if callable(runtime_diagnostics_fn) else None
    )
    if (
        bool(baseline.get("requires_duration_estimator", False))
        and max_control_steps is None
        and admission_diagnostics["reordered_scans"] <= 0
    ):
        raise ValueError(
            "Duration-ranked full replay never changed native queue order."
        )
    git = _git_state()
    manifest = get_system_manifest()
    manifest.update({
        "status": "AUDITING",
        "command": "benchmark_baseline",
        "suite_id": suite["suite_id"],
        "suite_file": str(Path(suite["_suite_path"]).relative_to(REPOSITORY_ROOT)),
        "suite_sha256": _sha256(Path(suite["_suite_path"])),
        "environment_config": str(config_path.relative_to(REPOSITORY_ROOT)),
        "environment_config_sha256": _sha256(config_path),
        "environment_protocol": paper_manifest["protocol_id"],
        "environment_protocol_version": paper_manifest["protocol_version"],
        "baseline_id": baseline_id,
        "baseline_family": baseline["family"],
        "baseline_policy": policy_manifest,
        "split": split,
        "policy_seed": spec.policy_seed,
        "seed_master": int(suite["randomness"]["master_seed"]),
        "seed_count": 1,
        "planned_seed_count": (
            int(baseline.get(
                "seed_count_override",
                suite["randomness"]["stochastic_seed_count"],
            ))
            if bool(baseline.get("stochastic", False)) else 1
        ),
        "seed_derivation": (
            STOCHASTIC_SEED_DERIVATION
            if bool(baseline.get("stochastic", False))
            else DETERMINISTIC_SEED_DERIVATION
        ),
        "seed_role": (
            "stochastic_generated_method_seed"
            if bool(baseline.get("stochastic", False))
            else "deterministic_provenance"
        ),
        "environment_seed": spec.environment_seed,
        "epochs": 0,
        "iterations": 0,
        "complete_replays_per_split": 1,
        "control_steps": steps,
        "max_control_steps": max_control_steps,
        "invalid_actions": 0,
        "reward_sum": reward_sum,
        "test_outcomes_accessed": False,
        "duration_estimator": duration_fit_report,
        "forecast_intervention": (
            forecast_provider.manifest() if forecast_provider is not None else None
        ),
        "numeric_threads": numeric_threads,
        "admission_diagnostics": admission_diagnostics,
        "git_commit": git["commit"],
        "git_dirty_at_run": git["dirty"],
        "raw_trace_hashes": view.hashes,
        "workload_sha256": view.workload_sha256,
        "prefix_workload_sha256": view.prefix_workload_sha256,
    })
    timing = {
        "elapsed_wall_seconds": round(elapsed, 6),
        "control_steps": steps,
        "control_steps_per_second": round(steps / max(elapsed, 1e-9), 6),
        "simulator_device": "cpu",
        "threads_per_worker": int(numeric_threads["torch_num_threads"]),
    }
    write_run_artifacts(
        str(run_dir),
        spec,
        manifest,
        timing,
        metrics=reduced,
        events=events,
        decision_timing=env.sim.decision_timing_records,
        extra_files={
            "baseline.json": {
                "baseline_id": baseline_id,
                **baseline,
                "policy_manifest": policy_manifest,
                "run_seed": run_seed,
                "forecast_intervention": (
                    forecast_provider.manifest() if forecast_provider is not None else None
                ),
            },
            "control_actions.json": action_rows,
            "publication_metrics.json": publication_metrics,
            "admission_diagnostics.json": admission_diagnostics,
            **(
                {"policy_runtime_diagnostics.json": policy_runtime_diagnostics}
                if policy_runtime_diagnostics is not None else {}
            ),
            **(
                {"duration_estimator.json": duration_fit_report}
                if duration_fit_report is not None else {}
            ),
        },
    )
    final_status = "EXTERNALLY_TRUNCATED" if truncated and not terminated else "COMPLETED"
    terminal_time = int(events[-1].time)
    del event_dicts, events, env, view, policy
    gc.collect()
    finalize_run_artifacts(str(run_dir), status=final_status)
    record_progress(
        final_status,
        current_time=terminal_time,
        completed_steps=steps,
    )
    return run_dir


def _smoke_task(task: tuple[str, str, int, int, str]) -> str:
    baseline_id, split, steps, run_seed, suite_path = task
    suite = load_suite(Path(suite_path))
    return str(run_baseline(
        suite,
        baseline_id,
        split,
        steps,
        run_seed,
        verify=False,
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=BENCHMARK_ROOT / "suite.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List declared baseline status and scope.")
    subparsers.add_parser("preflight", help="Verify the frozen environment and train or validation configs.")
    run_parser = subparsers.add_parser("run", help="Run one fixed baseline.")
    run_parser.add_argument("--baseline", required=True)
    run_parser.add_argument("--split", choices=("train", "val"), required=True)
    run_parser.add_argument("--max-control-steps", type=int)
    run_parser.add_argument("--progress-file", type=Path)
    run_parser.add_argument("--progress-every-steps", type=int, default=12)
    run_parser.add_argument(
        "--run-seed", type=int,
        help="Explicit method seed. Omit for deterministic master seed provenance.",
    )
    run_parser.add_argument(
        "--full-forecast-model",
        action="store_true",
        help="Use publication forecast hyperparameters in a bounded integration run.",
    )
    smoke_parser = subparsers.add_parser(
        "smoke",
        help="Run bounded baselines and generated seeds in independent CPU processes.",
    )
    smoke_parser.add_argument("--split", choices=("train", "val"), default="train")
    smoke_parser.add_argument("--max-control-steps", type=int, default=3)
    smoke_parser.add_argument("--baselines", nargs="+")
    args = parser.parse_args()

    suite = load_suite(args.suite)
    deterministic_seed = deterministic_seed_from_suite(suite)
    stochastic_seeds = stochastic_seeds_from_suite(suite)
    if args.command == "list":
        print(json.dumps({
            "baselines": suite["baselines"],
            "seed_contract": suite["randomness"],
            "deterministic_seed": deterministic_seed,
            "stochastic_generated_seeds": list(stochastic_seeds),
            "deterministic_execution": suite["deterministic_execution"],
        }, indent=2, sort_keys=True))
    elif args.command == "preflight":
        print(json.dumps(verify_workspace(suite), indent=2, sort_keys=True))
    elif args.command == "smoke":
        if args.max_control_steps < 1:
            raise SystemExit("--max-control-steps must be positive.")
        baseline_ids = args.baselines or [
            baseline_id
            for baseline_id, baseline in suite["baselines"].items()
            if baseline.get("status") in {"ready", "requires_train_fit"}
        ]
        verify_workspace(suite, splits=(args.split,))
        tasks = [
            (
                baseline_id,
                args.split,
                args.max_control_steps,
                deterministic_seed,
                suite["_suite_path"],
            )
            for baseline_id in baseline_ids
        ]
        workers = min(int(suite["resources"]["independent_workers"]), len(tasks))
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            run_dirs = list(executor.map(_smoke_task, tasks))
        print(json.dumps({"status": "PASS", "run_dirs": run_dirs}, indent=2))
    else:
        if args.max_control_steps is not None and args.max_control_steps < 1:
            raise SystemExit("--max-control-steps must be positive.")
        run_dir = run_baseline(
            suite,
            args.baseline,
            args.split,
            args.max_control_steps,
            deterministic_seed if args.run_seed is None else int(args.run_seed),
            progress_file=args.progress_file,
            progress_every_steps=args.progress_every_steps,
            forecast_smoke_hyperparameters=not args.full_forecast_model,
        )
        print(json.dumps({"status": "PASS", "run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
