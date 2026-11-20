"""Prepare, launch, and monitor declared quota RL methods."""

from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import yaml


BENCHMARK_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = BENCHMARK_ROOT.parent
DEFAULT_SUITE = BENCHMARK_ROOT / "rl_suite.yaml"
BATCH_ROOT = REPOSITORY_ROOT / "runs" / "quota_rl_batches"
sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark.rl_run import load_rl_suite, preflight  # noqa: E402
from benchmark.run import _environment_config, load_suite  # noqa: E402
from benchmark.seeds import stochastic_seeds_from_suite  # noqa: E402


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state() -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    return {"commit": commit, "dirty": bool(status)}


def _expected_steps(suite: Mapping[str, Any], split: str) -> int:
    base_path = Path(str(suite["environment"]["base_suite"]))
    if not base_path.is_absolute():
        base_path = (Path(str(suite["_suite_path"])).parent / base_path).resolve()
    base_suite = load_suite(base_path)
    config_path = _environment_config(base_suite, split)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    duration = int(config["score_end"]) - int(config["score_start"])
    interval = int(config["control_interval_seconds"])
    return (duration + interval - 1) // interval


def _jobs(
    suite: Mapping[str, Any],
    seed_limit: int | None = None,
    method_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    train_steps = _expected_steps(suite, "train")
    validation_steps = _expected_steps(suite, "val")
    seeds = stochastic_seeds_from_suite(suite)
    if seed_limit is not None:
        if seed_limit < 1 or seed_limit > len(seeds):
            raise ValueError("seed_limit must select one or more declared seeds.")
        seeds = seeds[:seed_limit]
    method_order = tuple(suite["resources"]["method_execution_order"])
    if len(method_order) != len(set(method_order)):
        raise ValueError("RL method execution order contains duplicates.")
    if set(method_order) != set(suite["methods"]):
        raise ValueError("RL method execution order must contain every method exactly once.")
    if method_ids is not None:
        requested = set(method_ids)
        unknown = sorted(requested - set(method_order))
        if unknown:
            raise ValueError(f"Unknown requested RL methods: {unknown}.")
        if len(requested) != len(method_ids):
            raise ValueError("Requested RL methods contain duplicates.")
        method_order = tuple(method for method in method_order if method in requested)
    jobs = []
    for method in method_order:
        for seed in seeds:
            jobs.append({
                "method_id": method,
                "seed": int(seed),
                "expected_train_steps": train_steps,
                "expected_eval_steps": validation_steps,
            })
    return jobs


def _bar(fraction: float, width: int = 26) -> str:
    value = min(1.0, max(0.0, float(fraction)))
    filled = int(round(value * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _resolve_batch(value: str) -> Path:
    if value != "latest":
        path = Path(value)
        return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()
    pointer = BATCH_ROOT / "latest.json"
    if not pointer.is_file():
        raise FileNotFoundError("No quota RL batch has been launched.")
    return Path(json.loads(pointer.read_text(encoding="utf-8"))["batch_dir"])


def _job_progress(job: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(job["progress_file"]))
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "state": "QUEUED",
        "completed_train_steps": 0,
        "completed_eval_steps": 0,
    }


def _supervisor_is_active(batch_dir: Path) -> bool:
    launcher_path = batch_dir / "launcher.json"
    if not launcher_path.is_file():
        return False
    try:
        pid = int(json.loads(launcher_path.read_text(encoding="utf-8"))["supervisor_pid"])
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\x00", b" ")
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        b"rl_experiments.py" in command
        and str(batch_dir.resolve()).encode("utf-8") in command
    )


def _render_status(batch_dir: Path) -> str:
    batch = json.loads((batch_dir / "batch.json").read_text(encoding="utf-8"))
    rows = []
    fractions = []
    for job in batch["jobs"]:
        progress = _job_progress(job)
        train_done = int(progress.get("completed_train_steps", 0))
        eval_done = int(progress.get("completed_eval_steps", 0))
        total = int(job["expected_train_steps"]) + int(job["expected_eval_steps"])
        fraction = (train_done + eval_done) / max(1, total)
        if str(progress.get("state")) in {"COMPLETED", "SMOKE_COMPLETED"}:
            fraction = 1.0
        fractions.append(fraction)
        label = f"{job['method_id']}#{job['seed']}"
        rows.append(
            f"{label:<28} {_bar(fraction)} {100.0 * fraction:6.2f}% "
            f"train {train_done:>5}/{job['expected_train_steps']:<5} "
            f"val {eval_done:>4}/{job['expected_eval_steps']:<4} "
            f"{progress.get('state', 'QUEUED')}"
        )
    overall = sum(fractions) / max(1, len(fractions))
    state_path = batch_dir / "status.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    display_state = str(state.get("state", "QUEUED"))
    if display_state == "RUNNING" and not _supervisor_is_active(batch_dir):
        display_state = "STALE_INTERRUPTED"
    header = (
        f"batch={batch['batch_id']} state={display_state} "
        f"overall={_bar(overall)} {100.0 * overall:.2f}%"
    )
    return "\n".join((header, *rows))


def _execute_job(job: Mapping[str, Any], suite_path: str, threads: int) -> dict[str, Any]:
    progress_file = Path(str(job["progress_file"]))
    job_dir = progress_file.parent
    progress = _job_progress(job)
    recovery_path = Path(str(progress.get("recovery_checkpoint", "")))
    run_dir_value = progress.get("run_dir")
    resume = bool(run_dir_value and recovery_path.is_file())
    command = [sys.executable, str(BENCHMARK_ROOT / "rl_run.py"), "--suite", suite_path]
    if resume:
        command.extend([
            "resume",
            "--run-dir",
            str(run_dir_value),
            "--progress-file",
            str(progress_file),
        ])
    else:
        command.extend([
            "run",
            "--method",
            str(job["method_id"]),
            "--seed",
            str(job["seed"]),
            "--progress-file",
            str(progress_file),
        ])
    environment = os.environ.copy()
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = str(int(threads))
    with (job_dir / "stdout.log").open("ab") as stdout, (job_dir / "stderr.log").open("ab") as stderr:
        result = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    payload = {
        "state": "COMPLETED" if result.returncode == 0 else "FAILED",
        "returncode": result.returncode,
        "method_id": job["method_id"],
        "seed": job["seed"],
        "finished_at": _utc_now(),
        "command": command,
        "resumed": resume,
    }
    _write_json(job_dir / "result.json", payload)
    if result.returncode != 0:
        progress = _job_progress(job)
        progress["state"] = "FAILED"
        progress["returncode"] = result.returncode
        _write_json(progress_file, progress)
    return payload


def _execute_batch(batch_dir: Path) -> int:
    batch = json.loads((batch_dir / "batch.json").read_text(encoding="utf-8"))
    suite_path = Path(str(batch["suite_path"]))
    if _sha256(suite_path) != str(batch["suite_sha256"]):
        raise RuntimeError("RL suite changed after batch creation. Resume refused.")
    git = _git_state()
    if git["dirty"] or git["commit"] != str(batch["git_commit"]):
        raise RuntimeError("Git state differs from the batch manifest. Resume refused.")
    already_completed = sum(
        str(_job_progress(job).get("state")) in {"COMPLETED", "SMOKE_COMPLETED"}
        for job in batch["jobs"]
    )
    old_status_path = batch_dir / "status.json"
    old_status = (
        json.loads(old_status_path.read_text(encoding="utf-8"))
        if old_status_path.is_file()
        else {}
    )
    is_resume = str(old_status.get("state")) in {"RUNNING", "FAILED", "STALE_INTERRUPTED"}
    resume_count = int(old_status.get("resume_count", 0)) + int(is_resume)
    _write_json(batch_dir / "status.json", {
        "state": "RUNNING",
        "started_at": old_status.get("started_at", _utc_now()),
        "resumed_at": _utc_now() if is_resume else None,
        "resume_count": resume_count,
        "total_jobs": len(batch["jobs"]),
        "completed_jobs": already_completed,
        "failed_jobs": 0,
    })
    completed = already_completed
    failed = 0
    method_order = list(batch["method_execution_order"])
    jobs_by_method = {
        method_id: [
            job for job in batch["jobs"] if job["method_id"] == method_id
        ]
        for method_id in method_order
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(batch["workers"])) as executor:
        execution_groups = (
            [method_order]
            if bool(batch.get("parallel_across_methods", False))
            else [[method_id] for method_id in method_order]
        )
        for method_group in execution_groups:
            pending_jobs = [
                job
                for method_id in method_group
                for job in jobs_by_method[method_id]
                if str(_job_progress(job).get("state")) not in {"COMPLETED", "SMOKE_COMPLETED"}
            ]
            futures = {
                executor.submit(
                    _execute_job,
                    job,
                    batch["suite_path"],
                    int(batch["threads_per_worker"]),
                ): job
                for job in pending_jobs
            }
            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"state": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
                    progress_file = Path(str(job["progress_file"]))
                    progress = _job_progress(job)
                    progress["state"] = "FAILED"
                    progress["error"] = result["error"]
                    _write_json(progress_file, progress)
                    _write_json(progress_file.parent / "result.json", result)
                completed += 1
                failed += int(result["state"] == "FAILED")
                _write_json(batch_dir / "status.json", {
                    "state": "RUNNING",
                    "active_methods": list(method_group),
                    "heartbeat_at": _utc_now(),
                    "total_jobs": len(batch["jobs"]),
                    "completed_jobs": completed,
                    "failed_jobs": failed,
                    "resume_count": resume_count,
                })
    state = "COMPLETED" if failed == 0 else "FAILED"
    _write_json(batch_dir / "status.json", {
        "state": state,
        "finished_at": _utc_now(),
        "total_jobs": len(batch["jobs"]),
        "completed_jobs": completed,
        "failed_jobs": failed,
        "resume_count": resume_count,
    })
    return int(failed > 0)


def _launch_supervisor(batch_dir: Path) -> subprocess.Popen[bytes]:
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
    _write_json(batch_dir / "launcher.json", {
        "supervisor_pid": process.pid,
        "launched_at": _utc_now(),
    })
    return process


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--seed-limit", type=int)
    prepare.add_argument("--methods", nargs="+")
    launch = commands.add_parser("launch")
    launch.add_argument("--seed-limit", type=int)
    launch.add_argument("--methods", nargs="+")
    resume = commands.add_parser("resume")
    resume.add_argument("--batch", default="latest")
    status = commands.add_parser("status")
    status.add_argument("--batch", default="latest")
    execute = commands.add_parser("_execute")
    execute.add_argument("--batch-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "_execute":
        raise SystemExit(_execute_batch(args.batch_dir.resolve()))
    if args.command == "status":
        print(_render_status(_resolve_batch(args.batch)))
        return
    if args.command == "resume":
        batch_dir = _resolve_batch(args.batch)
        if _supervisor_is_active(batch_dir):
            raise SystemExit(f"Quota RL supervisor is already active: {batch_dir}")
        batch = json.loads((batch_dir / "batch.json").read_text(encoding="utf-8"))
        suite = load_rl_suite(Path(str(batch["suite_path"])))
        preflight(suite)
        git = _git_state()
        if git["dirty"] or git["commit"] != str(batch["git_commit"]):
            raise SystemExit("Git state differs from the batch manifest. Resume refused.")
        if _sha256(Path(str(batch["suite_path"]))) != str(batch["suite_sha256"]):
            raise SystemExit("RL suite changed after batch creation. Resume refused.")
        process = _launch_supervisor(batch_dir)
        print(json.dumps({
            "status": "RESUMED",
            "batch_dir": str(batch_dir.resolve()),
            "supervisor_pid": process.pid,
            "monitor_command": (
                "conda run --no-capture-output -n trainenv python "
                "benchmark/rl_experiments.py status --batch latest"
            ),
        }, indent=2))
        return

    suite = load_rl_suite(args.suite)
    check = preflight(suite)
    jobs = _jobs(suite, seed_limit=args.seed_limit, method_ids=args.methods)
    selected_methods = list(dict.fromkeys(job["method_id"] for job in jobs))
    if args.command == "prepare":
        print(json.dumps({
            "status": "READY",
            "preflight": check,
            "workers": int(suite["resources"]["independent_run_workers"]),
            "threads_per_worker": int(suite["resources"]["threads_per_worker"]),
            "policy_device": suite["resources"]["policy_device"],
            "method_execution_order": selected_methods,
            "job_count": len(jobs),
            "seed_limit": args.seed_limit,
            "jobs": jobs,
            "launch_command": (
                "conda run --no-capture-output -n trainenv python "
                f"benchmark/rl_experiments.py --suite {args.suite} launch"
                + ("" if args.seed_limit is None else f" --seed-limit {args.seed_limit}")
                + ("" if args.methods is None else f" --methods {' '.join(args.methods)}")
            ),
        }, indent=2, sort_keys=True))
        return

    git = _git_state()
    if git["dirty"]:
        raise SystemExit("Full quota RL experiments require a clean Git worktree.")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_id = f"{suite['suite_id']}_{timestamp}"
    batch_dir = BATCH_ROOT / batch_id
    batch_dir.mkdir(parents=True, exist_ok=False)
    materialized = []
    for job in jobs:
        job_dir = batch_dir / "jobs" / str(job["method_id"]) / f"seed{job['seed']}"
        job_dir.mkdir(parents=True)
        payload = {**job, "progress_file": str((job_dir / "progress.json").resolve())}
        materialized.append(payload)
        _write_json(job_dir / "progress.json", {
            "state": "QUEUED",
            "completed_train_steps": 0,
            "completed_eval_steps": 0,
        })
    batch = {
        "schema_version": "1.0",
        "batch_id": batch_id,
        "created_at": _utc_now(),
        "suite_path": str(Path(str(suite["_suite_path"])).resolve()),
        "suite_sha256": _sha256(Path(str(suite["_suite_path"]))),
        "git_commit": git["commit"],
        "workers": int(suite["resources"]["independent_run_workers"]),
        "threads_per_worker": int(suite["resources"]["threads_per_worker"]),
        "method_execution_order": selected_methods,
        "parallel_across_methods": bool(
            suite["resources"].get("parallel_across_methods", False)
        ),
        "seed_limit": args.seed_limit,
        "jobs": materialized,
    }
    _write_json(batch_dir / "batch.json", batch)
    _write_json(batch_dir / "status.json", {"state": "QUEUED", "total_jobs": len(materialized)})
    _write_json(BATCH_ROOT / "latest.json", {"batch_dir": str(batch_dir.resolve())})
    process = _launch_supervisor(batch_dir)
    print(json.dumps({
        "status": "LAUNCHED",
        "batch_dir": str(batch_dir.resolve()),
        "supervisor_pid": process.pid,
        "monitor_command": (
            "conda run --no-capture-output -n trainenv python "
            "benchmark/rl_experiments.py status --batch latest"
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
