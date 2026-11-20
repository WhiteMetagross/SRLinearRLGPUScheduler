"""Prepare, launch, and monitor detached deterministic baseline batches."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


BENCHMARK_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = BENCHMARK_ROOT.parent
DEFAULT_PLAN = BENCHMARK_ROOT / "experiment_plan.yaml"
BATCH_ROOT = REPOSITORY_ROOT / "runs" / "benchmark_batches"
TERMINAL_JOB_STATES = {"COMPLETED", "FAILED", "HALTED", "CANCELLED"}
TERMINAL_BATCH_STATES = {"COMPLETED", "FAILED", "HALTED", "CANCELLED"}
sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark.run import (  # noqa: E402
    _baseline_policy,
    _suite_base,
    _write_json_atomic,
    load_suite,
    verify_workspace,
)
from benchmark.seeds import (  # noqa: E402
    deterministic_seed_from_suite,
    stochastic_seeds_from_suite,
)


def _load_plan(path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    resolved = path.resolve()
    plan = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or plan.get("schema_version") != "1.0":
        raise ValueError("Experiment plan requires schema_version '1.0'.")
    suite_path = Path(str(plan["suite"]))
    if not suite_path.is_absolute():
        suite_path = (resolved.parent / suite_path).resolve()
    return plan, resolved, load_suite(suite_path)


def _validation_baselines(plan: dict[str, Any]) -> tuple[str, ...]:
    values = plan.get("validation", {}).get("baselines")
    if not isinstance(values, list) or not values:
        raise ValueError("Experiment plan requires nonempty validation baselines.")
    result = tuple(str(value) for value in values)
    if len(set(result)) != len(result):
        raise ValueError("Validation baseline IDs must be unique.")
    return result


def _full_run_blockers(
    plan: dict[str, Any],
    suite: dict[str, Any],
    baseline_ids: tuple[str, ...] | None = None,
) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    selected = baseline_ids or _validation_baselines(plan)
    if len(set(selected)) != len(selected):
        raise ValueError("Selected baseline IDs must be unique.")
    for baseline_id in selected:
        try:
            _baseline_policy(suite, baseline_id, allow_smoke_artifact=False)
        except (KeyError, ValueError, FileNotFoundError) as exc:
            blockers.append({"baseline_id": baseline_id, "reason": str(exc)})
            continue
        declaration = suite["baselines"][baseline_id].get("forecast_intervention")
        if declaration is not None:
            for field in ("forecast_suite", "calibration_artifact"):
                raw = declaration.get(field)
                if not raw:
                    blockers.append({
                        "baseline_id": baseline_id,
                        "reason": f"Forecast intervention requires {field}.",
                    })
                    continue
                path = Path(str(raw))
                if not path.is_absolute():
                    path = (_suite_base(suite) / path).resolve()
                if not path.is_file():
                    blockers.append({
                        "baseline_id": baseline_id,
                        "reason": f"Missing forecast intervention file: {path}",
                    })
            calibration_raw = declaration.get("calibration_artifact")
            if calibration_raw:
                calibration_path = Path(str(calibration_raw))
                if not calibration_path.is_absolute():
                    calibration_path = (_suite_base(suite) / calibration_path).resolve()
                if calibration_path.is_file():
                    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
                    if calibration.get("qualification") != "publication":
                        blockers.append({
                            "baseline_id": baseline_id,
                            "reason": "Forecast calibration is not publication qualified.",
                        })
                    if int(calibration.get("origin_cadence_hours", 24)) != 1:
                        blockers.append({
                            "baseline_id": baseline_id,
                            "reason": (
                                "Forecast calibration cadence is not one hour. "
                                "Generate the hourly train calibration artifact first."
                            ),
                        })
    return blockers


def _git_state() -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPOSITORY_ROOT, text=True
    ).strip())
    return {"commit": commit, "dirty": dirty}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bar(fraction: float, width: int = 30) -> str:
    bounded = min(1.0, max(0.0, float(fraction)))
    filled = int(round(width * bounded))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _resolve_batch(raw: str) -> Path:
    if raw != "latest":
        path = Path(raw)
        if not path.is_absolute():
            path = (REPOSITORY_ROOT / path).resolve()
        return path
    pointer = BATCH_ROOT / "latest.json"
    if not pointer.is_file():
        raise FileNotFoundError("No baseline batch has been launched.")
    return Path(json.loads(pointer.read_text(encoding="utf-8"))["batch_dir"])


def _initial_progress(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "state": "QUEUED",
        "baseline_id": job["baseline_id"],
        "split": job["split"],
        "run_seed": job["run_seed"],
        "completed_steps": 0,
        "expected_steps": job["expected_steps"],
        "progress_fraction": 0.0,
        "heartbeat_at": None,
    }


def _render_status(batch_dir: Path) -> tuple[str, bool]:
    batch = json.loads((batch_dir / "batch.json").read_text(encoding="utf-8"))
    state_path = batch_dir / "status.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {
        "state": "QUEUED"
    }
    rows = []
    fractions = []
    terminal = True
    for job in batch["jobs"]:
        progress_path = Path(job["progress_file"])
        progress = (
            json.loads(progress_path.read_text(encoding="utf-8"))
            if progress_path.is_file() else _initial_progress(job)
        )
        fraction = float(progress.get("progress_fraction", 0.0))
        fractions.append(fraction)
        job_state = str(progress.get("state", "QUEUED"))
        if job_state not in TERMINAL_JOB_STATES:
            terminal = False
        rows.append(
            f"{job['baseline_id'] + '#' + str(job['run_seed']):<43} {_bar(fraction)} "
            f"{100.0 * fraction:6.2f}% "
            f"{int(progress.get('completed_steps', 0)):>5}/"
            f"{int(progress.get('expected_steps', job['expected_steps'])):<5} {job_state}"
        )
    overall = sum(fractions) / len(fractions) if fractions else 1.0
    if state.get("state") in TERMINAL_BATCH_STATES:
        terminal = True
    header = (
        f"batch={batch['batch_id']} state={state.get('state', 'UNKNOWN')} "
        f"overall={_bar(overall)} {100.0 * overall:.2f}%"
    )
    return "\n".join((header, *rows)), terminal


def _execute_job(
    job: dict[str, Any],
    suite_path: str,
    progress_every: int,
    threads_per_worker: int,
) -> dict[str, Any]:
    progress_path = Path(job["progress_file"])
    job_dir = progress_path.parent
    command = [
        sys.executable,
        str(BENCHMARK_ROOT / "run.py"),
        "--suite",
        suite_path,
        "run",
        "--baseline",
        job["baseline_id"],
        "--split",
        job["split"],
        "--progress-file",
        str(progress_path),
        "--progress-every-steps",
        str(progress_every),
        "--run-seed",
        str(job["run_seed"]),
    ]
    environment = os.environ.copy()
    thread_count = str(int(threads_per_worker))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = thread_count
    started = _utc_now()
    with (job_dir / "stdout.log").open("wb") as stdout, (job_dir / "stderr.log").open("wb") as stderr:
        result = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    payload = {
        "baseline_id": job["baseline_id"],
        "state": "COMPLETED" if result.returncode == 0 else "FAILED",
        "returncode": result.returncode,
        "started_at": started,
        "finished_at": _utc_now(),
        "command": command,
    }
    _write_json_atomic(job_dir / "result.json", payload)
    if result.returncode != 0:
        current = (
            json.loads(progress_path.read_text(encoding="utf-8"))
            if progress_path.is_file() else _initial_progress(job)
        )
        current["state"] = "FAILED"
        current["heartbeat_at"] = _utc_now()
        _write_json_atomic(progress_path, current)
    return payload


def _execute_batch(batch_dir: Path) -> int:
    batch = json.loads((batch_dir / "batch.json").read_text(encoding="utf-8"))
    _write_json_atomic(batch_dir / "status.json", {
        "state": "RUNNING",
        "supervisor_pid": os.getpid(),
        "started_at": _utc_now(),
        "completed_jobs": 0,
        "failed_jobs": 0,
        "total_jobs": len(batch["jobs"]),
    })
    completed = 0
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(batch["workers"])) as executor:
        futures = {
            executor.submit(
                _execute_job,
                job,
                batch["suite_path"],
                int(batch["progress_every_steps"]),
                int(batch.get("threads_per_worker", 1)),
            ): job
            for job in batch["jobs"]
        }
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover: defensive supervisor boundary.
                result = {
                    "baseline_id": job["baseline_id"],
                    "state": "FAILED",
                    "returncode": None,
                    "finished_at": _utc_now(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                progress_path = Path(job["progress_file"])
                progress = (
                    json.loads(progress_path.read_text(encoding="utf-8"))
                    if progress_path.is_file() else _initial_progress(job)
                )
                progress["state"] = "FAILED"
                progress["heartbeat_at"] = _utc_now()
                _write_json_atomic(progress_path, progress)
                _write_json_atomic(progress_path.parent / "result.json", result)
            completed += 1
            failed += int(result["state"] == "FAILED")
            _write_json_atomic(batch_dir / "status.json", {
                "state": "RUNNING",
                "supervisor_pid": os.getpid(),
                "started_at": batch["created_at"],
                "heartbeat_at": _utc_now(),
                "completed_jobs": completed,
                "failed_jobs": failed,
                "total_jobs": len(batch["jobs"]),
            })
    final_state = "COMPLETED" if failed == 0 else "FAILED"
    _write_json_atomic(batch_dir / "status.json", {
        "state": final_state,
        "supervisor_pid": os.getpid(),
        "started_at": batch["created_at"],
        "finished_at": _utc_now(),
        "completed_jobs": completed,
        "failed_jobs": failed,
        "total_jobs": len(batch["jobs"]),
    })
    return int(failed > 0)


def _prepare_payload(
    plan: dict[str, Any], plan_path: Path, suite: dict[str, Any]
) -> dict[str, Any]:
    blockers = _full_run_blockers(plan, suite)
    python_prefix = "conda run --no-capture-output -n trainenv python"
    commands = {
        "launch_validation": (
            f"{python_prefix} benchmark/experiments.py --plan {plan_path} launch --split val"
        ),
        "monitor": (
            f"{python_prefix} benchmark/experiments.py --plan {plan_path} "
            "status --batch latest --watch"
        ),
    }
    generated = stochastic_seeds_from_suite(suite)
    planned_jobs = 0
    for baseline_id in _validation_baselines(plan):
        baseline = suite["baselines"][baseline_id]
        if bool(baseline.get("stochastic", False)):
            planned_jobs += int(baseline.get("seed_count_override", len(generated)))
        else:
            planned_jobs += 1
    return {
        "status": "READY" if not blockers else "BLOCKED",
        "deterministic_repetitions_per_split": 1,
        "deterministic_seed": deterministic_seed_from_suite(suite),
        "baseline_count": len(_validation_baselines(plan)),
        "planned_job_count": planned_jobs,
        "blockers": blockers,
        "commands": commands,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    launch = commands.add_parser("launch")
    launch.add_argument("--split", choices=("val",), default="val")
    launch.add_argument(
        "--baselines",
        nargs="+",
        help="Optional exact subset of predeclared validation baselines.",
    )
    status = commands.add_parser("status")
    status.add_argument("--batch", default="latest")
    status.add_argument("--watch", action="store_true")
    status.add_argument("--interval", type=float, default=15.0)
    execute = commands.add_parser("_execute")
    execute.add_argument("--batch-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "_execute":
        raise SystemExit(_execute_batch(args.batch_dir.resolve()))

    plan, plan_path, suite = _load_plan(args.plan)
    if args.command == "prepare":
        verify_workspace(suite, splits=("train", "val"))
        print(json.dumps(_prepare_payload(plan, plan_path, suite), indent=2))
        return
    if args.command == "status":
        if args.interval <= 0:
            raise SystemExit("--interval must be positive.")
        batch_dir = _resolve_batch(args.batch)
        while True:
            rendered, terminal = _render_status(batch_dir)
            if args.watch and sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print(rendered, flush=True)
            if terminal or not args.watch:
                return
            time.sleep(args.interval)

    baseline_ids = (
        tuple(args.baselines) if args.baselines else _validation_baselines(plan)
    )
    declared = set(_validation_baselines(plan))
    unknown = sorted(set(baseline_ids) - declared)
    if unknown:
        raise SystemExit(f"Baselines are not declared in the experiment plan: {unknown}")
    blockers = _full_run_blockers(plan, suite, baseline_ids)
    if blockers:
        print(json.dumps({"status": "BLOCKED", "blockers": blockers}, indent=2))
        raise SystemExit(2)
    git = _git_state()
    if git["dirty"]:
        raise SystemExit("Full experiments require a clean Git worktree.")
    verify_workspace(suite, splits=(args.split,))
    deterministic_seed = deterministic_seed_from_suite(suite)
    stochastic_seeds = stochastic_seeds_from_suite(suite)
    config_path = Path(suite["_suite_path"]).parent / suite["environment"]["configs"][args.split]
    config = yaml.safe_load(config_path.resolve().read_text(encoding="utf-8"))
    expected_steps = int(
        (int(config["score_end"]) - int(config["score_start"]))
        // int(config["control_interval_seconds"])
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_id = f"{plan['plan_id']}_{args.split}_{timestamp}"
    batch_dir = BATCH_ROOT / batch_id
    batch_dir.mkdir(parents=True, exist_ok=False)
    jobs = []
    for baseline_id in baseline_ids:
        baseline = suite["baselines"][baseline_id]
        if bool(baseline.get("stochastic", False)):
            seed_count = int(baseline.get("seed_count_override", len(stochastic_seeds)))
            if not 1 <= seed_count <= len(stochastic_seeds):
                raise ValueError(
                    f"Invalid seed_count_override for {baseline_id!r}: {seed_count}."
                )
            seeds = stochastic_seeds[:seed_count]
        else:
            seeds = (deterministic_seed,)
        for run_seed in seeds:
            job_dir = batch_dir / "jobs" / baseline_id / f"seed{int(run_seed)}"
            job_dir.mkdir(parents=True)
            job = {
                "baseline_id": baseline_id,
                "split": args.split,
                "run_seed": int(run_seed),
                "expected_steps": expected_steps,
                "progress_file": str((job_dir / "progress.json").resolve()),
            }
            jobs.append(job)
            _write_json_atomic(job_dir / "progress.json", _initial_progress(job))
    batch = {
        "schema_version": "1.0",
        "batch_id": batch_id,
        "created_at": _utc_now(),
        "plan_path": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "suite_path": suite["_suite_path"],
        "suite_sha256": _sha256(Path(suite["_suite_path"])),
        "git_commit": git["commit"],
        "split": args.split,
        "workers": int(plan["deterministic_protocol"]["independent_workers"]),
        "threads_per_worker": int(
            plan["deterministic_protocol"]["threads_per_worker"]
        ),
        "progress_every_steps": int(plan["deterministic_protocol"]["progress_every_steps"]),
        "jobs": jobs,
    }
    _write_json_atomic(batch_dir / "batch.json", batch)
    _write_json_atomic(batch_dir / "status.json", {"state": "QUEUED", "total_jobs": len(jobs)})
    _write_json_atomic(BATCH_ROOT / "latest.json", {"batch_dir": str(batch_dir.resolve())})
    stdout = (batch_dir / "supervisor.stdout.log").open("ab")
    stderr = (batch_dir / "supervisor.stderr.log").open("ab")
    try:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "_execute", "--batch-dir", str(batch_dir)],
            cwd=REPOSITORY_ROOT,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    finally:
        stdout.close()
        stderr.close()
    _write_json_atomic(batch_dir / "launcher.json", {
        "supervisor_pid": process.pid,
        "launched_at": _utc_now(),
    })
    print(json.dumps({
        "status": "LAUNCHED",
        "batch_dir": str(batch_dir.resolve()),
        "supervisor_pid": process.pid,
        "monitor_command": (
            "conda run --no-capture-output -n trainenv python "
            "benchmark/experiments.py status --batch latest --watch"
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
