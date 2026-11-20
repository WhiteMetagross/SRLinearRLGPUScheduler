"""
cli.py: Main entry points and subcommands for the research simulator:
- audit-trace
- replay
- compare-parity
- profile
Writes structured run folders with manifest, config, timing, metrics, and events.
"""

import sys
import os
import time
import json
import hashlib
import re
import subprocess
import yaml
import platform
import resource
import argparse
from datetime import datetime
from pathlib import Path
import dataclasses
from dataclasses import asdict, replace
from typing import Dict, Any, Iterator, List, Optional

import pandas as pd
import numpy as np

from .types import EpisodeSpec, InvalidConfigError, ResearchSimError
from .trace import TraceStore, EpisodeTraceView
from .core import ResearchSimulator
from .verifier import NoCheckVerifier
from .metrics import EpisodeMetrics
from .measurement import (
    SCHEMA_VERSION,
    validate_event_dict,
    reduce_replay_events,
    reconcile_metrics
)
from .warmup_snapshot import (
    get_snapshot_path,
    has_warmup_snapshot,
    load_warmup_snapshot,
    generate_warmup_snapshot,
    DEFAULT_SNAPSHOT_DIR
)


UPSTREAM_COMMIT = "e998d5453e626a0b743b3fd5137c54c987db780b"

AUDIT_PREFIX_CACHE_VERSION = "1.0"


def _audit_prefix_cache_key(spec: EpisodeSpec, manifest: Dict[str, Any]) -> str:
    """Build a provenance key for the immutable warmup event prefix."""
    trace_hashes = manifest.get("raw_trace_hashes") or manifest.get("data_hashes") or {}
    payload = {
        "cache_version": AUDIT_PREFIX_CACHE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "code_tree_sha256": manifest.get("code_tree_sha256"),
        "node_sha256": trace_hashes.get("node_sha256"),
        "job_sha256": trace_hashes.get("job_sha256"),
        "spec": asdict(spec),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_audit_prefix_event(event: Dict[str, Any]) -> bytes:
    """Canonicalize one prefix event while allowing per run identity fields."""
    canonical = dict(event)
    canonical["run_id"] = "<RUN_ID>"
    canonical["episode_id"] = "<EPISODE_ID>"
    return (json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


_RUN_ID_JSON_FIELD = re.compile(rb'"run_id":"(?:\\.|[^"\\])*"')
_EPISODE_ID_JSON_FIELD = re.compile(rb'"episode_id":"(?:\\.|[^"\\])*"')


def _canonical_audit_prefix_line(line: bytes) -> bytes:
    """Fast canonicalization for writer emitted compact JSON lines."""
    line = _RUN_ID_JSON_FIELD.sub(b'"run_id":"<RUN_ID>"', line)
    return _EPISODE_ID_JSON_FIELD.sub(b'"episode_id":"<EPISODE_ID>"', line)


def _load_audit_prefix_cache(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("cache_version") != AUDIT_PREFIX_CACHE_VERSION:
            return {"cache_version": AUDIT_PREFIX_CACHE_VERSION, "entries": {}}
        return data if isinstance(data.get("entries"), dict) else {"cache_version": AUDIT_PREFIX_CACHE_VERSION, "entries": {}}
    except (OSError, json.JSONDecodeError):
        return {"cache_version": AUDIT_PREFIX_CACHE_VERSION, "entries": {}}


def _store_audit_prefix_cache(path: Path, cache: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, sort_keys=True, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def _validate_cached_prefix_identity(
    event: Dict[str, Any], run_id: str, episode_id: str, previous: Optional[Dict[str, Any]]
) -> None:
    """Lightweight checks used only after a validated canonical hash match."""
    if event.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Cached prefix event schema version mismatch.")
    if event.get("run_id") != run_id or event.get("episode_id") != episode_id:
        raise ValueError("Cached prefix event has wrong run or episode identity.")
    if not isinstance(event.get("event_seq"), int) or event["event_seq"] < 0:
        raise ValueError("Cached prefix event has invalid event_seq.")
    if not isinstance(event.get("time"), (int, float)) or not np.isfinite(event["time"]):
        raise ValueError("Cached prefix event has invalid time.")
    if previous is not None:
        if event["event_seq"] != previous["event_seq"] + 1:
            raise ValueError("Cached prefix event sequence is not contiguous.")
        if event["time"] < previous["time"]:
            raise ValueError("Cached prefix event time moved backwards.")


def _iter_jsonl_events(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield one JSON event at a time without retaining the event file in memory."""
    with path.open("rb") as stream:
        for line_no, raw_line in enumerate(stream, start=1):
            payload = raw_line.rstrip(b"\r\n")
            if not payload.strip():
                raise ValueError(f"Blank event line at line {line_no}.")
            yield json.loads(payload)


def get_system_manifest() -> Dict[str, Any]:
    """Captures runtime environment, OS, hardware, and dependency hashes."""
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    total_ram_gb = 16.0
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_ram_gb = round(float(line.split()[1]) / 1024.0 / 1024.0, 2)
                    break
    except Exception:
        pass

    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    code_files = list((root / "src" / "research_sim").rglob("*.py")) + list((root / "scripts").glob("*.py"))
    for source in sorted(code_files):
        digest.update(source.relative_to(root).as_posix().encode())
        digest.update(hashlib.sha256(source.read_bytes()).digest())
    lock = root / "environment-linux-64.lock"
    upstream = root.parent / "tmp" / "source_review" / "Spot"
    upstream_actual = None
    upstream_status = None
    if upstream.is_dir():
        rev = subprocess.run(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"], capture_output=True, text=True
        )
        status = subprocess.run(
            ["git", "-C", str(upstream), "status", "--porcelain", "-uno"], capture_output=True, text=True
        )
        if rev.returncode == 0:
            upstream_actual = rev.stdout.strip()
            upstream_status = status.stdout.strip() if status.returncode == 0 else None
    try:
        import torch
        framework = {"torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                     "cuda_available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            framework["cuda_device_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
            framework["cuda_device_count"] = torch.cuda.device_count()
    except ImportError:
        framework = {"torch_version": None, "cuda_version": None, "cuda_available": False,
                     "cuda_device_name": None, "cuda_device_count": 0}
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "total_ram_gb": total_ram_gb,
        "upstream_commit": upstream_actual,
        "upstream_expected_commit": UPSTREAM_COMMIT,
        "upstream_available": upstream_actual is not None,
        "upstream_clean": (upstream_actual is not None and not upstream_status),
        "code_tree_sha256": digest.hexdigest(),
        "environment_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
        **framework,
        "peak_rss_mb": round(rusage.ru_maxrss / 1024.0, 2)
    }


def create_run_dir(command: str, policy: str = "none") -> str:
    """Creates a new timestamped run directory under runs/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    clean_pol = policy.replace("-", "_")
    run_id = f"run_{ts}_{command}_{clean_pol}"
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def load_spec_from_config(config_path: str) -> EpisodeSpec:
    """Loads EpisodeSpec from YAML configuration file."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    field_names = {f.name for f in dataclasses.fields(EpisodeSpec)}
    spec_data = {k: v for k, v in data.items() if k in field_names}
    return EpisodeSpec(**spec_data)


def get_run_provenance(config_path: str, view: EpisodeTraceView) -> Dict[str, Any]:
    """Record the exact input selection and user supplied config bytes."""
    jobs_df = view.jobs_df
    return {
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "node_sha256": view.hashes["node_sha256"],
        "job_sha256": view.hashes["job_sha256"],
        "workload_sha256": view.workload_sha256,
        "prefix_workload_sha256": view.prefix_workload_sha256,
        "workload_transform": view.workload_metadata,
        "selected_nodes": int(len(view.node_df)),
        "selected_jobs": int(len(jobs_df)),
        "selected_hp_jobs": int((jobs_df["job_type"] == "HP").sum()),
        "selected_spot_jobs": int((jobs_df["job_type"] == "Spot").sum()),
    }


def write_run_artifacts(
    run_dir: str,
    spec: EpisodeSpec,
    manifest: Dict[str, Any],
    timing: Dict[str, Any],
    metrics: Optional[Dict[str, Any]] = None,
    events: Optional[List[Any]] = None,
    decision_timing: Optional[List[Dict[str, Any]]] = None,
    solver_invocations: Optional[List[Dict[str, Any]]] = None,
    extra_files: Optional[Dict[str, Any]] = None
):
    """Writes standard artifacts to run directory."""
    os.makedirs(run_dir, exist_ok=True)

    # 1. config.yaml and config.json
    spec_dict = asdict(spec)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(spec_dict, f, default_flow_style=False)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(spec_dict, f, indent=2, allow_nan=False)

    # 2. events.jsonl
    events_path = os.path.join(run_dir, "events.jsonl")
    if events is not None:
        hasher = hashlib.sha256()
        event_count = 0
        with open(events_path, "w", encoding="utf-8") as f:
            for ev in events:
                if hasattr(ev, "to_dict"):
                    ev_dict = ev.to_dict()
                elif hasattr(ev, "__dict__"):
                    ev_dict = asdict(ev)
                elif isinstance(ev, dict):
                    ev_dict = ev
                else:
                    ev_dict = dict(ev)
                line = json.dumps(ev_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"
                f.write(line)
                hasher.update(line.encode("utf-8"))
                event_count += 1
        manifest["events_sha256"] = hasher.hexdigest()
        manifest["events_count"] = event_count
        manifest["events_bytes"] = os.path.getsize(events_path)
        if event_count:
            with open(events_path, "r", encoding="utf-8") as f:
                first_event = json.loads(f.readline())
            if manifest.get("run_id") not in (None, first_event["run_id"]):
                raise ValueError("Manifest run_id does not match emitted event run_id.")
            if manifest.get("episode_id") not in (None, first_event["episode_id"]):
                raise ValueError("Manifest episode_id does not match emitted event episode_id.")
            manifest["run_id"] = first_event["run_id"]
            manifest["episode_id"] = first_event["episode_id"]
    elif os.path.exists(events_path):
        events_bytes = Path(events_path).read_bytes()
        manifest["events_sha256"] = hashlib.sha256(events_bytes).hexdigest()
        manifest["events_count"] = len([l for l in events_bytes.splitlines() if l.strip()])

    # 3. manifest.json
    manifest["schema_version"] = SCHEMA_VERSION
    if events is not None:
        manifest.setdefault("status", "AUDITING")
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)

    # 4. timing.json
    with open(os.path.join(run_dir, "timing.json"), "w") as f:
        json.dump(timing, f, indent=2, allow_nan=False)

    # 5. metrics.json
    if metrics is not None:
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, allow_nan=False)

    # 6. decision_timing.jsonl
    with open(os.path.join(run_dir, "decision_timing.jsonl"), "w", encoding="utf-8") as f:
        for row in decision_timing or []:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")

    # 7. solver_invocations.jsonl
    if solver_invocations is not None:
        with open(os.path.join(run_dir, "solver_invocations.jsonl"), "w") as f:
            for row in solver_invocations:
                f.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")

    # 8. extra files (e.g. first_divergence.json)
    if extra_files:
        for fname, fcontent in extra_files.items():
            with open(os.path.join(run_dir, fname), "w") as f:
                    json.dump(fcontent, f, indent=2, allow_nan=False)


# ---------------------------------------------------------------------------
# Command: audit-trace
# ---------------------------------------------------------------------------

def run_audit_trace(config_path: str):
    spec = load_spec_from_config(config_path)
    start_time = time.perf_counter()
    run_dir = create_run_dir("audit_trace", "audit")

    print("=================================================================")
    print("AUDIT-TRACE: Validating Raw Data and Cluster Topology")
    print("=================================================================")

    store = TraceStore.get_instance("data/raw/spot_gpu_v2026")
    node_df = store._raw_nodes
    job_df = store._raw_jobs

    total_nodes = len(node_df)
    total_jobs = len(job_df)
    model_counts = node_df["gpu_model"].value_counts().to_dict()
    a100_nodes = int(model_counts.get("A100-SXM4-80GB", 0))

    # A100 job counts
    a100_jobs = job_df[job_df["gpu_model"] == "A100-SXM4-80GB"]
    total_a100_jobs = len(a100_jobs)
    a100_hp_jobs = len(a100_jobs[a100_jobs["job_type"] == "HP"])
    a100_spot_jobs = len(a100_jobs[a100_jobs["job_type"] == "Spot"])
    a100_fractional = len(a100_jobs[a100_jobs["gpu_request"] < 1.0])

    # Arrivals inside the declared score interval.
    window_jobs = a100_jobs[
        (a100_jobs["submit_time"] >= spec.score_start) &
        (a100_jobs["submit_time"] < spec.score_end)
    ]
    window_a100_arrivals = len(window_jobs)

    audit_summary = {
        "node_sha256": store.hashes["node_sha256"],
        "job_sha256": store.hashes["job_sha256"],
        "total_nodes": total_nodes,
        "a100_nodes": a100_nodes,
        "a100_gpus": a100_nodes * 8,
        "total_jobs": total_jobs,
        "total_a100_jobs": total_a100_jobs,
        "a100_hp_jobs": a100_hp_jobs,
        "a100_spot_jobs": a100_spot_jobs,
        "a100_fractional_jobs": a100_fractional,
        "score_window_a100_arrivals": window_a100_arrivals,
        "gpu_models": model_counts
    }

    elapsed = time.perf_counter() - start_time
    rusage = resource.getrusage(resource.RUSAGE_SELF)

    manifest = get_system_manifest()
    manifest.update({
        "command": "audit-trace",
        "config_file": config_path,
        "data_hashes": store.hashes,
        "audit_summary": audit_summary
    })

    timing = {
        "elapsed_wall_seconds": round(elapsed, 3),
        "peak_rss_mb": round(rusage.ru_maxrss / 1024.0, 2)
    }

    write_run_artifacts(run_dir, spec, manifest, timing, metrics=audit_summary)

    print(f"Node CSV SHA256: {store.hashes['node_sha256']}")
    print(f"Job CSV SHA256:  {store.hashes['job_sha256']}")
    print(f"Total Nodes:     {total_nodes} (A100 Nodes: {a100_nodes}, GPUs: {a100_nodes * 8})")
    print(f"Total Jobs:      {total_jobs}")
    print(f"A100 Total Jobs: {total_a100_jobs} (HP: {a100_hp_jobs}, Spot: {a100_spot_jobs}, Fractional: {a100_fractional})")
    print(f"A100 Score Window Arrivals ({spec.score_start}..{spec.score_end}): {window_a100_arrivals}")
    print(f"Run directory:   {run_dir}")
    print(f"Audit completed in {elapsed:.2f}s.")


# ---------------------------------------------------------------------------
# Command: replay
# ---------------------------------------------------------------------------

def run_replay(
    config_path: str,
    policy: str,
    use_warmup_snapshot: bool = False,
    snapshot_dir: str = DEFAULT_SNAPSHOT_DIR
):
    spec = load_spec_from_config(config_path)
    run_dir = create_run_dir("replay", policy)
    start_wall = time.perf_counter()
    run_id = Path(run_dir).name
    write_run_artifacts(
        run_dir, spec,
        {"command": "replay", "policy": policy, "run_id": run_id, "status": "RUNNING"},
        {"status": "RUNNING"}, decision_timing=[],
    )

    print("=================================================================")
    print(f"REPLAY: Trace Replay (Policy: {policy}, Warmup Snapshot: {use_warmup_snapshot})")
    print(f"Config: {config_path}")
    print(f"Window: [{spec.warmup_start}..{spec.score_start}..{spec.score_end}..{spec.followup_end}]")
    print("=================================================================")

    sim: Optional[ResearchSimulator] = None
    try:
        store = TraceStore.get_instance("data/raw/spot_gpu_v2026")
        view = store.create_episode_view(spec)
        print(f"Retained Jobs in View: {view.total_jobs}, Nodes: {view.total_nodes}")

        if use_warmup_snapshot:
            snap_path = get_snapshot_path(spec, snapshot_dir)
            if not has_warmup_snapshot(spec, snapshot_dir):
                print(f"Warmup snapshot not found at {snap_path}. Generating warmup snapshot...")
                generate_warmup_snapshot(
                    spec=spec,
                    view=view,
                    snapshot_dir=snapshot_dir,
                    log_dir=os.path.join(run_dir, "warmup_generation_logs"),
                    progress_hook=lambda t, done, total: print(
                        f"[Warmup Generation] t={t}, completed={done}/{total}", flush=True
                    )
                )
                print(f"Warmup snapshot generated at {snap_path}.")
            print(f"Restoring simulation state from warmup snapshot: {snap_path}")
            pol_type = "native_gfs" if policy == "native_gfs" else "decision_boundary"
            ver = NoCheckVerifier() if policy == "deterministic_no_check" else None
            sim = load_warmup_snapshot(
                filepath=snap_path,
                policy_type=pol_type,
                verifier=ver,
                log_dir=os.path.join(run_dir, "logs"),
                spec=spec,
                view=view,
            )
        elif policy == "native_gfs":
            sim = ResearchSimulator(
                spec=spec,
                view=view,
                policy_type="native_gfs",
                log_dir=os.path.join(run_dir, "logs")
            )
        elif policy == "deterministic_no_check":
            sim = ResearchSimulator(
                spec=spec,
                view=view,
                policy_type="decision_boundary",
                verifier=NoCheckVerifier(),
                log_dir=os.path.join(run_dir, "logs")
            )
        else:
            raise InvalidConfigError(f"Unsupported policy: {policy}")
    except BaseException as exc:
        abort_manifest = {
            "command": "replay", "policy": policy, "status": "ABORTED",
            "run_id": sim.run_id if sim is not None else run_id,
            "episode_id": sim.episode_id if sim is not None else None,
            "abort_error_type": type(exc).__name__, "abort_error_message": str(exc),
            "last_event_seq": sim.ledger._seq - 1 if sim is not None else None,
            "partial_artifacts": ["logs/"],
        }
        try:
            write_run_artifacts(
                run_dir, spec, abort_manifest,
                {"elapsed_wall_seconds": time.perf_counter() - start_wall, "status": "ABORTED"},
                events=sim.ledger.events if sim is not None else None,
                decision_timing=sim.decision_timing_records if sim is not None else [],
            )
        except Exception as write_exc:
            print(f"Failed to serialize aborted replay setup artifacts: {write_exc}", file=sys.stderr)
        raise

    sim.progress_hook = lambda t, done, total: print(
        f"Replay progress: t={t}, completed={done}/{total}", flush=True
    )

    try:
        if policy == "native_gfs":
            while True:
                context, delta = sim.advance_until_decision()
                if delta.terminated or delta.truncated:
                    break
        elif policy == "deterministic_no_check":
            decisions_count = 0
            while True:
                context, delta = sim.advance_until_decision()
                if context is None:
                    break
                decisions_count += 1
                obs = sim.observe()
                mask = obs["action_mask"]

                # Action slot zero is the safe deferral fallback.
                chosen_action = 0
                for slot in range(1, spec.M + 1):
                    act_idx = 2 * slot
                    if mask[act_idx]:
                        chosen_action = act_idx
                        break
                sim.apply(chosen_action)
            print(f"Total Paused Spot Decisions Handled: {decisions_count}")
    except BaseException as exc:
        abort_manifest = {
            "command": "replay", "policy": policy, "status": "ABORTED",
            "run_id": sim.run_id, "episode_id": sim.episode_id,
            "abort_error_type": type(exc).__name__, "abort_error_message": str(exc),
            "last_event_seq": sim.ledger._seq - 1,
            "partial_artifacts": ["events.jsonl", "decision_timing.jsonl", "logs/"],
        }
        try:
            write_run_artifacts(
                run_dir, spec, abort_manifest,
                {"elapsed_wall_seconds": time.perf_counter() - start_wall, "status": "ABORTED"},
                events=sim.ledger.events, decision_timing=sim.decision_timing_records,
            )
        except Exception as write_exc:
            print(f"Failed to serialize aborted replay artifacts: {write_exc}", file=sys.stderr)
        raise

    elapsed_wall = time.perf_counter() - start_wall
    simulated_seconds = sim.time - (spec.score_start if use_warmup_snapshot else spec.warmup_start)
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss_mb = round(rusage.ru_maxrss / 1024.0, 2)

    metrics = sim.get_metrics()
    metrics_dict = asdict(metrics)

    manifest = get_system_manifest()
    manifest.update({
        "command": "replay",
        "policy": policy,
        "use_warmup_snapshot": use_warmup_snapshot,
        "snapshot_path": str(get_snapshot_path(spec, snapshot_dir)) if use_warmup_snapshot else None,
        "config_file": config_path,
        "data_hashes": store.hashes,
        "spec": asdict(spec),
        "credit_rule": "event_in_score_half_open",
        "source_compatible_credit_rule": "event_in_score_left_open",
        "phase_rule": "warmup:t<score_start;score:score_start<=t<score_end;followup:t>=score_end",
        "checks_enabled": False,
        "status": "AUDITING",
        **get_run_provenance(config_path, view),
    })

    timing = {
        "elapsed_wall_seconds": round(elapsed_wall, 3),
        "simulated_seconds": simulated_seconds,
        "sim_speed_ratio": round(simulated_seconds / max(elapsed_wall, 0.001), 2),
        "peak_rss_mb": peak_rss_mb,
        "use_warmup_snapshot": use_warmup_snapshot
    }

    write_run_artifacts(
        run_dir=run_dir,
        spec=spec,
        manifest=manifest,
        timing=timing,
        metrics=metrics_dict,
        events=sim.ledger.events,
        decision_timing=sim.decision_timing_records
    )
    finalize_run_artifacts(
        run_dir,
        status="EXTERNALLY_TRUNCATED" if metrics.truncated else "COMPLETED",
    )

    print("-----------------------------------------------------------------")
    print(f"Replay Finished: Final Sim Time = {sim.time}, Wall Time = {elapsed_wall:.2f}s")
    print(f"Useful Spot GPU Hours Credited: {metrics.useful_spot_gpu_hours:.4f}")
    print(f"High Priority Gang Arrivals:    {metrics.hp_gang_arrivals_cohort}")
    print(f"High Priority Gang Violations:  {metrics.hp_gang_violations}")
    print(f"Violation Rate:                 {metrics.hp_gang_violation_rate}")
    print(f"Completed Jobs (HP/Spot):       {metrics.completed_hp_jobs} / {metrics.completed_spot_jobs}")
    print(f"Preemption Events:              {metrics.spot_evictions_count}")
    print(f"Total Events Logged:            {len(sim.ledger.events)}")
    print(f"Run directory:                  {run_dir}")
    print("=================================================================")


# ---------------------------------------------------------------------------
# Command: compare-parity
# ---------------------------------------------------------------------------

def run_compare_parity(config_path: str):
    spec = load_spec_from_config(config_path)
    if spec.trace_mode != "source_compatible" or spec.forecast_mode != "native_online":
        raise InvalidConfigError("Source parity requires source_compatible trace and native_online forecast.")
    source_state = get_system_manifest()
    if source_state["upstream_commit"] != UPSTREAM_COMMIT or not source_state["upstream_clean"]:
        raise ResearchSimError("Original GFS checkout differs from the pinned clean revision.")
    run_dir = create_run_dir("compare_parity", "parity")
    start_wall = time.perf_counter()

    print("=================================================================")
    print("COMPARE-PARITY: Differential Parity Between Upstream and Fork")
    print(f"Config: {config_path}")
    print("=================================================================")

    store = TraceStore.get_instance("data/raw/spot_gpu_v2026")
    view = store.create_episode_view(spec)

    # 1. Run maintained native GFS fork
    print("Running maintained native GFS fork...")
    t0 = time.perf_counter()
    fork_sim = ResearchSimulator(
        spec=spec,
        view=view,
        policy_type="native_gfs",
        log_dir=os.path.join(run_dir, "fork_logs")
    )
    fork_sim.progress_hook = lambda t, done, total: print(
        f"Fork progress: t={t}, completed={done}/{total}", flush=True
    )
    while True:
        context, delta = fork_sim.advance_until_decision()
        if delta.terminated or delta.truncated:
            break
    fork_wall = time.perf_counter() - t0
    fork_metrics = fork_sim.get_metrics()
    print(f"Fork finished in {fork_wall:.2f}s. Events: {len(fork_sim.ledger.events)}")

    # 2. Run unmodified upstream GFS via reference harness
    print("Running unmodified upstream GFS scheduler via reference harness...")
    t0 = time.perf_counter()
    from scripts.upstream_reference import run_upstream
    upstream_events, upstream_summary = run_upstream(
        node_df=view.node_df,
        job_dicts=view.get_all_jobs_fresh(),
        spec_dict=asdict(spec),
        log_dir=os.path.join(run_dir, "upstream_logs")
    )
    upstream_wall = time.perf_counter() - t0
    print(f"Upstream finished in {upstream_wall:.2f}s. Events: {len(upstream_events)}")

    # 3. Compare events chronologically
    first_divergence = None
    parity_passed = True

    # Compare the released transition loop with the maintained core, including
    # placement maps, quotas, credit amounts, and event order.
    critical_types = {"arrival", "start", "placement", "preemption", "completion", "quota_update", "credit"}

    def normalize_event(ev):
        if not isinstance(ev, dict):
            ev = asdict(ev)
        kind = ev["event_type"]
        detail = ev.get("details", {})
        if kind in {"arrival", "completion", "start"}:
            payload = {"job_type": detail.get("job_type")}
            if kind == "start":
                payload["nodes"] = detail.get("nodes")
        elif kind == "placement":
            payload = {"nodes": detail.get("nodes")}
        elif kind == "credit":
            if "in_score_window" in detail:
                credited = detail["gpu_seconds"] if detail["in_score_window"] else 0.0
            else:
                credited = detail.get("credited_gpu_seconds", 0.0)
            payload = {"credited_gpu_seconds": credited}
        elif kind == "quota_update":
            payload = {"quotas": detail.get("spot_quota", detail.get("quotas"))}
        else:
            payload = {}
        return {"time": ev["time"], "event_type": kind, "job_id": ev.get("job_id"), "details": payload}

    fork_critical = [normalize_event(ev) for ev in fork_sim.ledger.events if ev.event_type in critical_types]
    upstream_critical = [normalize_event(ev) for ev in upstream_events if ev["event_type"] in critical_types]

    print(f"Comparing critical events: Fork={len(fork_critical)}, Upstream={len(upstream_critical)}")

    min_len = min(len(fork_critical), len(upstream_critical))
    for i in range(min_len):
        fe = fork_critical[i]
        ue = upstream_critical[i]

        if fe != ue:
            parity_passed = False
            first_divergence = {
                "step_index": i,
                "fork_event": fe,
                "upstream_event": ue,
                "reason": "Event type, time, job, placement, quota, or credit mismatch"
            }
            break

    if parity_passed and len(fork_critical) != len(upstream_critical):
        parity_passed = False
        first_divergence = {
            "step_index": min_len,
            "reason": f"Event count mismatch: Fork={len(fork_critical)} vs Upstream={len(upstream_critical)}"
        }

    if parity_passed:
        fork_summary = {
            "end_job_num": fork_sim.end_job_num,
            "total_job_num": fork_sim.total_job_num,
            "spot_gpu_time": dict(fork_sim.spot_gpu_time),
            "guar_gpu_time": dict(fork_sim.guar_gpu_time),
            "remaining_hp_queued": sum(map(len, fork_sim.guar_que_list.values())),
            "remaining_spot_queued": sum(map(len, fork_sim.spot_que_list.values())),
        }
        reference_summary = {k: upstream_summary[k] for k in fork_summary}
        if fork_summary != reference_summary:
            parity_passed = False
            first_divergence = {
                "step_index": len(fork_critical),
                "reason": "Final counter or queue mismatch",
                "fork_summary": fork_summary,
                "upstream_summary": reference_summary,
            }

    elapsed_wall = time.perf_counter() - start_wall
    rusage = resource.getrusage(resource.RUSAGE_SELF)

    manifest = get_system_manifest()
    manifest.update({
        "command": "compare-parity",
        "config_file": config_path,
        "parity_status": "MATCH" if parity_passed else "MISMATCH",
        "fork_critical_events": len(fork_critical),
        "upstream_critical_events": len(upstream_critical),
        **get_run_provenance(config_path, view),
    })

    timing = {
        "elapsed_wall_seconds": round(elapsed_wall, 3),
        "fork_wall_seconds": round(fork_wall, 3),
        "upstream_wall_seconds": round(upstream_wall, 3),
        "peak_rss_mb": round(rusage.ru_maxrss / 1024.0, 2)
    }

    extra_files = {}
    if first_divergence:
        extra_files["first_divergence.json"] = first_divergence

    write_run_artifacts(
        run_dir=run_dir,
        spec=spec,
        manifest=manifest,
        timing=timing,
        metrics=asdict(fork_metrics),
        events=fork_sim.ledger.events,
        extra_files=extra_files
    )

    print("-----------------------------------------------------------------")
    if parity_passed:
        print("PARITY STATUS: EXACT MATCH across all critical transitions!")
    else:
        print(f"PARITY STATUS: DIVERGENCE DETECTED at step {first_divergence.get('step_index')}")
        print(f"Reason: {first_divergence.get('reason')}")
        print(f"Written details to {os.path.join(run_dir, 'first_divergence.json')}")
    print(f"Run directory: {run_dir}")
    print("=================================================================")


# ---------------------------------------------------------------------------
# Command: profile
# ---------------------------------------------------------------------------

def run_profile(
    config_path: str,
    window_hours: int,
    num_trials: int = 3,
    use_warmup_snapshot: bool = False,
    snapshot_dir: str = DEFAULT_SNAPSHOT_DIR
):
    spec = load_spec_from_config(config_path)
    if window_hours is not None and window_hours > 0:
        bounded_score_end = spec.score_start + window_hours * 3600
        bounded_followup_end = bounded_score_end + spec.L_H
        spec = dataclasses.replace(
            spec,
            score_end=bounded_score_end,
            followup_end=bounded_followup_end
        )
    actual_window_hours = (spec.score_end - spec.score_start) / 3600.0
    if window_hours is not None and abs(actual_window_hours - window_hours) > 1e-6:
        raise InvalidConfigError(
            f"Requested window ({window_hours} hours) differs from actual config window ({actual_window_hours} hours)."
        )

    run_dir = create_run_dir("profile", "benchmark")

    print("=================================================================")
    print(f"PROFILE: {num_trials}-Pass Benchmark (Window: {actual_window_hours} hours, Warmup Snapshot: {use_warmup_snapshot})")
    print(f"Config: {config_path}")
    print(f"Window: [{spec.warmup_start}..{spec.score_start}..{spec.score_end}..{spec.followup_end}]")
    print("=================================================================")

    store = TraceStore.get_instance("data/raw/spot_gpu_v2026")
    view = store.create_episode_view(spec)

    snap_path = get_snapshot_path(spec, snapshot_dir)
    if use_warmup_snapshot:
        if not has_warmup_snapshot(spec, snapshot_dir):
            print(f"Warmup snapshot not found at {snap_path}. Generating warmup snapshot before benchmark...")
            generate_warmup_snapshot(
                spec=spec,
                view=view,
                snapshot_dir=snapshot_dir,
                progress_hook=lambda t, done, total: print(
                    f"[Warmup Generation] t={t}, completed={done}/{total}", flush=True
                )
            )
            print(f"Warmup snapshot generated at {snap_path}.")

    wall_times: List[float] = []
    speeds: List[float] = []
    peak_mems: List[float] = []

    last_metrics: Dict[str, Any] = {}
    last_decisions: int = 0
    last_quota_updates: int = 0
    last_forecast_trains: int = 0
    last_sim_time: int = spec.score_end

    for trial in range(1, num_trials + 1):
        print(f"--- Running Trial {trial}/{num_trials} ---")
        t0 = time.perf_counter()
        if use_warmup_snapshot:
            print(f"Restoring simulation state from warmup snapshot: {snap_path}")
            sim = load_warmup_snapshot(
                filepath=snap_path,
                policy_type="native_gfs",
                log_dir=os.path.join(run_dir, f"trial_{trial}_logs"),
                spec=spec
            )
        else:
            sim = ResearchSimulator(
                spec=spec,
                view=view,
                policy_type="native_gfs",
                log_dir=os.path.join(run_dir, f"trial_{trial}_logs")
            )
        sim.progress_hook = lambda t, done, total, tr=trial: print(
            f"[Trial {tr}/{num_trials}] Replay progress: t={t}, completed={done}/{total}", flush=True
        )
        while True:
            context, delta = sim.advance_until_decision()
            if delta.terminated or delta.truncated:
                break
        t_wall = time.perf_counter() - t0
        sim_sec = sim.time - (spec.score_start if use_warmup_snapshot else spec.warmup_start)
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        peak_mb = rusage.ru_maxrss / 1024.0

        speed = sim_sec / max(t_wall, 0.001)
        wall_times.append(t_wall)
        speeds.append(speed)
        peak_mems.append(peak_mb)
        print(
            f"Trial {trial}: Wall={t_wall:.2f}s, SimSec={sim_sec}, Speed={speed:.1f}x, "
            f"Decisions={sim.decisions_count}, QuotaUpdates={sim.quota_update_count}, "
            f"ForecastTrains={sim.forecast_training_count}, PeakRSS={peak_mb:.1f}MB"
        )

        last_metrics = asdict(sim.get_metrics())
        last_decisions = sim.decisions_count
        last_quota_updates = sim.quota_update_count
        last_forecast_trains = sim.forecast_training_count
        last_sim_time = sim.time

        del sim
        import gc
        gc.collect()

    median_wall = float(np.median(wall_times))
    median_speed = float(np.median(speeds))
    max_peak_rss = float(np.max(peak_mems))

    # This is only a linear estimate. Native forecast training frequency and
    # queue size change with horizon, so it is not a measured full run.
    primary_spec = load_spec_from_config(str(Path(__file__).resolve().parents[2] / "configs" / "a100_primary.yaml"))
    full_config_seconds = primary_spec.followup_end - primary_spec.warmup_start + 1
    measured_seconds = max(1, last_sim_time - (spec.score_start if use_warmup_snapshot else spec.warmup_start))
    scale_factor = full_config_seconds / measured_seconds
    estimated_quarter_wall_seconds = median_wall * scale_factor
    estimated_quarter_wall_hours = estimated_quarter_wall_seconds / 3600.0

    # Multi-seed (5 seeds)
    multi_seed_hours = estimated_quarter_wall_hours * 5.0

    episodes_10_cost_hours = (median_wall * 10.0) / 3600.0
    episodes_100_cost_hours = (median_wall * 100.0) / 3600.0

    print("-----------------------------------------------------------------")
    print("PROFILE RESULTS & EXTRAPOLATION:")
    print(f"  Warmup Start:                 {spec.warmup_start}")
    print(f"  Score Start:                  {spec.score_start}")
    print(f"  Score End:                    {spec.score_end}")
    print(f"  Followup End:                 {spec.followup_end}")
    print(f"  Configured Score Duration:    {actual_window_hours} hours")
    print(f"  Actual Replay Duration:       {measured_seconds} simulated seconds")
    print(f"  Warmup Snapshot Enabled:      {use_warmup_snapshot}")
    print(f"  Decisions Handled:            {last_decisions}")
    print(f"  Quota Updates:                {last_quota_updates}")
    print(f"  Forecast Training Calls:      {last_forecast_trains}")
    print(f"  Median Wall Time:             {median_wall:.2f}s")
    print(f"  Median Simulation Speed:      {median_speed:.1f} sim_sec / wall_sec")
    print(f"  Peak Resident Memory (RSS):   {max_peak_rss:.1f} MB")
    print(f"  Training 10 Episodes Cost:    {episodes_10_cost_hours:.2f} hours ({median_wall * 10:.1f}s)")
    print(f"  Training 100 Episodes Cost:   {episodes_100_cost_hours:.2f} hours ({median_wall * 100:.1f}s)")
    print(f"  Linear Full Config Estimate:  {estimated_quarter_wall_hours:.2f} hours ({estimated_quarter_wall_seconds:.1f}s)")
    print(f"  Extrapolated 5-Seed Experiment: {multi_seed_hours:.2f} hours")
    print("-----------------------------------------------------------------")

    profile_metrics = {
        "warmup_start": spec.warmup_start,
        "score_start": spec.score_start,
        "score_end": spec.score_end,
        "followup_end": spec.followup_end,
        "window_hours": actual_window_hours,
        "actual_replay_seconds": measured_seconds,
        "simulated_seconds": measured_seconds,
        "use_warmup_snapshot": use_warmup_snapshot,
        "snapshot_path": str(snap_path) if use_warmup_snapshot else None,
        "decisions_count": last_decisions,
        "quota_updates_count": last_quota_updates,
        "forecast_training_calls": last_forecast_trains,
        "trials_wall_seconds": [round(w, 3) for w in wall_times],
        "median_wall_seconds": round(median_wall, 3),
        "median_speed_sim_per_wall": round(median_speed, 2),
        "peak_rss_mb": round(max_peak_rss, 2),
        "episodes_10_cost_hours": round(episodes_10_cost_hours, 2),
        "episodes_100_cost_hours": round(episodes_100_cost_hours, 2),
        "full_config_seconds": full_config_seconds,
        "linear_full_config_estimate_hours": round(estimated_quarter_wall_hours, 2),
        "extrapolated_5_seed_experiment_hours": round(multi_seed_hours, 2),
        "episode_metrics": last_metrics
    }

    manifest = get_system_manifest()
    manifest.update({
        "command": "profile",
        "window_hours": actual_window_hours,
        "warmup_start": spec.warmup_start,
        "score_start": spec.score_start,
        "score_end": spec.score_end,
        "followup_end": spec.followup_end,
        "simulated_seconds": measured_seconds,
        "use_warmup_snapshot": use_warmup_snapshot,
        "snapshot_path": str(snap_path) if use_warmup_snapshot else None,
        "decisions": last_decisions,
        "quota_updates": last_quota_updates,
        "forecast_training_calls": last_forecast_trains,
        "config_file": config_path,
        "data_hashes": store.hashes,
        **get_run_provenance(config_path, view),
    })

    timing = {
        "median_wall_seconds": round(median_wall, 3),
        "sim_speed_ratio": round(median_speed, 2),
        "peak_rss_mb": round(max_peak_rss, 2),
        "simulated_seconds": measured_seconds,
        "warmup_start": spec.warmup_start,
        "score_start": spec.score_start,
        "score_end": spec.score_end,
        "followup_end": spec.followup_end,
        "use_warmup_snapshot": use_warmup_snapshot,
    }

    write_run_artifacts(
        run_dir=run_dir,
        spec=spec,
        manifest=manifest,
        timing=timing,
        metrics=profile_metrics
    )
    print(f"Run directory: {run_dir}")
    print("=================================================================")


# ---------------------------------------------------------------------------
# Command: parallel-profile (Tier 2 Multi-Worker Benchmark)
# ---------------------------------------------------------------------------

def _parallel_episode_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """Worker process executing an isolated episode rollout."""
    ep_idx = task["episode_idx"]
    config_path = task["config_path"]
    policy = task["policy"]
    use_warmup_snapshot = task["use_warmup_snapshot"]
    snapshot_dir = task["snapshot_dir"]
    run_dir = task["run_dir"]
    enable_forecast_cache = task.get("enable_forecast_cache", False)

    base_spec = load_spec_from_config(config_path)
    warmup_seed = base_spec.policy_seed
    policy_seed = warmup_seed + ep_idx if task.get("vary_seed", False) else warmup_seed
    spec = replace(base_spec, policy_seed=policy_seed)

    t_start = time.perf_counter()
    ep_log_dir = os.path.join(run_dir, f"worker_{ep_idx}")

    if use_warmup_snapshot:
        snap_path = get_snapshot_path(base_spec, snapshot_dir)
        pol_type = "native_gfs" if policy == "native_gfs" else "decision_boundary"
        ver = NoCheckVerifier() if policy == "deterministic_no_check" else None
        sim = load_warmup_snapshot(
            filepath=snap_path,
            policy_type=pol_type,
            verifier=ver,
            log_dir=ep_log_dir,
            spec=spec,
            enable_forecast_cache=enable_forecast_cache
        )
    else:
        store = TraceStore.get_instance("data/raw/spot_gpu_v2026")
        view = store.create_episode_view(spec)
        sim = ResearchSimulator(
            spec=spec,
            view=view,
            policy_type="native_gfs" if policy == "native_gfs" else "decision_boundary",
            verifier=NoCheckVerifier() if policy == "deterministic_no_check" else None,
            log_dir=ep_log_dir
        )

    total_eps = task.get("total_episodes", "")
    ep_tag = f"Ep {ep_idx + 1:2d}/{total_eps}" if total_eps else f"Ep {ep_idx + 1}"
    sim.progress_hook = lambda t, done, total: (
        print(f"  [{ep_tag}] t={t}, completed={done}/{total}", flush=True)
        if t % 14400 == 0 else None
    )

    decisions_count = 0
    actions_count = 0
    if policy == "native_gfs":
        while True:
            context, delta = sim.advance_until_decision()
            if delta.terminated or delta.truncated:
                break
    elif policy == "deterministic_no_check":
        while True:
            context, delta = sim.advance_until_decision()
            if context is None:
                if delta.terminated or delta.truncated:
                    break
                raise ResearchSimError("Decision policy returned None context before episode termination.")
            decisions_count += 1
            obs = sim.observe()
            mask = obs["action_mask"]

            # Select first valid unverified placement candidate (slot >= 1)
            chosen_action = 0  # deferral fallback
            for slot in range(1, spec.M + 1):
                act_idx = 2 * slot + 0
                if mask[act_idx]:
                    chosen_action = act_idx
                    break
            sim.apply(chosen_action)
            actions_count += 1
    else:
        raise InvalidConfigError(f"Unsupported policy: {policy}")

    wall_sec = time.perf_counter() - t_start
    m = sim.get_metrics()
    rusage = resource.getrusage(resource.RUSAGE_SELF)

    quota_digest = hashlib.sha256(
        json.dumps([[str(r[0]), r[1], r[2], float(r[3]), float(r[4])] for r in sim.spot_quota_record]).encode()
    ).hexdigest()
    event_count = len(sim.ledger.events)

    return {
        "episode_idx": ep_idx,
        "warmup_seed": warmup_seed,
        "policy_seed": policy_seed,
        "wall_seconds": round(wall_sec, 3),
        "peak_rss_mb": round(rusage.ru_maxrss / 1024.0, 1),
        "useful_spot_gpu_hours": round(m.useful_spot_gpu_hours, 4),
        "hp_gang_violations": m.hp_gang_violations,
        "completed_hp_jobs": m.completed_hp_jobs,
        "completed_spot_jobs": m.completed_spot_jobs,
        "spot_evictions_count": m.spot_evictions_count,
        "quota_updates": sim.quota_update_count,
        "decisions_count": decisions_count,
        "actions_count": actions_count,
        "quota_digest": quota_digest,
        "event_count": event_count,
        "terminated": delta.terminated,
        "truncated": delta.truncated
    }


def run_parallel_profile(
    config_path: str,
    episodes: int = 10,
    workers: int = 1,
    policy: str = "deterministic_no_check",
    use_warmup_snapshot: bool = True,
    snapshot_dir: str = DEFAULT_SNAPSHOT_DIR,
    enable_forecast_cache: bool = False,
    vary_seed: bool = False
):
    import concurrent.futures
    spec = load_spec_from_config(config_path)
    run_dir = create_run_dir("parallel_profile", f"{workers}w_{episodes}ep_{policy}")
    start_wall = time.perf_counter()

    print("=================================================================")
    print("PARALLEL PROFILE: Multi-Worker RL Rollout Benchmark")
    print(f"Config:           {config_path}")
    print(f"Episodes:         {episodes}")
    print(f"Workers:          {workers}")
    print(f"Policy:           {policy}")
    print(f"Warmup Snapshot:  {use_warmup_snapshot}")
    print(f"Forecast Cache:   {enable_forecast_cache}")
    print(f"Vary Seed:        {vary_seed}")
    print("=================================================================")

    if use_warmup_snapshot and not has_warmup_snapshot(spec, snapshot_dir):
        print("Warmup snapshot missing. Pre-generating warmup snapshot once...")
        store = TraceStore.get_instance("data/raw/spot_gpu_v2026")
        view = store.create_episode_view(spec)
        generate_warmup_snapshot(spec=spec, view=view, snapshot_dir=snapshot_dir)

    tasks = [
        {
            "episode_idx": i,
            "config_path": config_path,
            "policy": policy,
            "use_warmup_snapshot": use_warmup_snapshot,
            "snapshot_dir": snapshot_dir,
            "run_dir": run_dir,
            "enable_forecast_cache": enable_forecast_cache,
            "vary_seed": vary_seed,
            "total_episodes": episodes,
        }
        for i in range(episodes)
    ]

    results = []
    print(f"Spawning {workers} worker process(es) for {episodes} episodes...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_parallel_episode_worker, task): task["episode_idx"] for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results.append(res)
            print(f"  [Worker Completed] Ep {res['episode_idx'] + 1:2d}/{episodes}: "
                  f"wall={res['wall_seconds']:.2f}s, "
                  f"decisions={res['decisions_count']}, "
                  f"useful_spot_gpu_hours={res['useful_spot_gpu_hours']:.4f}, "
                  f"violations={res['hp_gang_violations']}")

    total_wall = time.perf_counter() - start_wall
    results.sort(key=lambda r: r["episode_idx"])

    ep_walls = [r["wall_seconds"] for r in results]
    mean_ep_wall = float(np.mean(ep_walls))
    median_ep_wall = float(np.median(ep_walls))
    decisions_total = sum(r["decisions_count"] for r in results)
    throughput_ep_per_min = (episodes / total_wall) * 60.0
    throughput_sim_hours_per_sec = (episodes * 24.0 * 3600.0) / total_wall / 3600.0
    decisions_per_sec = decisions_total / total_wall

    print("-----------------------------------------------------------------")
    print("PARALLEL PROFILE BENCHMARK RESULTS:")
    print(f"  Total Wall Time:              {total_wall:.2f}s ({total_wall / 60.0:.2f} minutes)")
    print(f"  Episodes Completed:           {len(results)} / {episodes}")
    print(f"  Workers:                      {workers}")
    print(f"  Mean Episode Wall Time:       {mean_ep_wall:.2f}s")
    print(f"  Median Episode Wall Time:     {median_ep_wall:.2f}s")
    print(f"  Total Decisions:              {decisions_total} ({decisions_total / episodes:.1f} per ep)")
    print(f"  Decisions / Wall Second:      {decisions_per_sec:.2f}")
    print(f"  Throughput:                   {throughput_ep_per_min:.2f} episodes/min")
    print(f"  Extrapolated 100-Ep Wall Time: {(total_wall / episodes) * 100.0 / 60.0:.2f} minutes ({(total_wall / episodes) * 100.0 / 3600.0:.2f} hours)")
    print("-----------------------------------------------------------------")

    artifacts = {
        "episodes": episodes,
        "workers": workers,
        "policy": policy,
        "use_warmup_snapshot": use_warmup_snapshot,
        "enable_forecast_cache": enable_forecast_cache,
        "total_wall_seconds": round(total_wall, 3),
        "mean_episode_wall_seconds": round(mean_ep_wall, 3),
        "median_episode_wall_seconds": round(median_ep_wall, 3),
        "total_decisions": decisions_total,
        "decisions_per_episode": round(decisions_total / max(1, episodes), 1),
        "decisions_per_second": round(decisions_per_sec, 2),
        "throughput_episodes_per_min": round(throughput_ep_per_min, 2),
        "extrapolated_100_episodes_minutes": round((total_wall / episodes) * 100.0 / 60.0, 2),
        "results": results
    }
    with open(os.path.join(run_dir, "parallel_profile.json"), "w") as f:
        json.dump(artifacts, f, indent=2)

    manifest = get_system_manifest()
    manifest.update({
        "command": "parallel-profile",
        "episodes": episodes,
        "workers": workers,
        "policy": policy,
        "use_warmup_snapshot": use_warmup_snapshot,
        "enable_forecast_cache": enable_forecast_cache,
        "config_file": config_path
    })
    write_run_artifacts(
        run_dir=run_dir,
        spec=spec,
        manifest=manifest,
        timing={"total_wall_seconds": round(total_wall, 3), "mean_ep_seconds": round(mean_ep_wall, 3)},
        metrics={"episodes_completed": len(results), "mean_spot_hours": float(np.mean([r["useful_spot_gpu_hours"] for r in results]))}
    )
    print(f"Run directory: {run_dir}")
    print("=================================================================")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Research Simulator CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # audit-trace
    p_audit = subparsers.add_parser("audit-trace", help="Audit trace and verify data integrity")
    p_audit.add_argument("--config", required=True, help="Path to config YAML")

    # replay
    p_replay = subparsers.add_parser("replay", help="Replay simulation on configured window")
    p_replay.add_argument("--config", required=True, help="Path to config YAML")
    p_replay.add_argument("--policy", required=True, choices=["native_gfs", "deterministic_no_check"], help="Decision policy")
    p_replay.add_argument("--use-warmup-snapshot", action="store_true", default=False, help="Use cached warmup state snapshot at score_start to bypass repeated 9.5-minute warmup")
    p_replay.add_argument("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR, help="Directory to store and load warmup snapshots")

    # compare-parity
    p_parity = subparsers.add_parser("compare-parity", help="Compare differential parity against upstream GFS")
    p_parity.add_argument("--config", required=True, help="Path to config YAML")

    # profile
    p_profile = subparsers.add_parser("profile", help="Profile simulation performance and extrapolate")
    p_profile.add_argument("--config", required=True, help="Path to config YAML")
    p_profile.add_argument("--window-hours", type=int, default=24, help="Window hours for profiling")
    p_profile.add_argument("--trials", type=int, default=3, help="Number of benchmark trials")
    p_profile.add_argument("--use-warmup-snapshot", action="store_true", default=False, help="Use cached warmup state snapshot at score_start to bypass repeated 9.5-minute warmup")
    p_profile.add_argument("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR, help="Directory to store and load warmup snapshots")

    # parallel-profile
    p_par = subparsers.add_parser("parallel-profile", help="Benchmark multi-worker parallel RL rollouts")
    p_par.add_argument("--config", required=True, help="Path to config YAML")
    p_par.add_argument("--episodes", type=int, default=10, help="Number of episodes to execute")
    p_par.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes")
    p_par.add_argument("--policy", default="deterministic_no_check", choices=["native_gfs", "deterministic_no_check"], help="Decision policy")
    p_par.add_argument("--use-warmup-snapshot", action="store_true", default=False, help="Use cached warmup state snapshot")
    p_par.add_argument("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR, help="Directory to store and load warmup snapshots")
    p_par.add_argument("--enable-forecast-cache", action="store_true", default=False, help="Enable precomputed forecast cache (experimental)")
    p_par.add_argument("--vary-seed", action="store_true", default=False, help="Vary policy seed per episode")

    # audit-run
    p_audit_run = subparsers.add_parser("audit-run", help="Audit serialized run artifacts and reconcile offline metrics")
    p_audit_run.add_argument("--run-dir", required=True, help="Path to run directory")

    args = parser.parse_args()

    if args.subcommand == "audit-trace":
        run_audit_trace(args.config)
    elif args.subcommand == "replay":
        run_replay(args.config, args.policy, args.use_warmup_snapshot, args.snapshot_dir)
    elif args.subcommand == "compare-parity":
        run_compare_parity(args.config)
    elif args.subcommand == "profile":
        run_profile(args.config, args.window_hours, args.trials, args.use_warmup_snapshot, args.snapshot_dir)
    elif args.subcommand == "parallel-profile":
        run_parallel_profile(
            args.config,
            episodes=args.episodes,
            workers=args.workers,
            policy=args.policy,
            use_warmup_snapshot=args.use_warmup_snapshot,
            snapshot_dir=args.snapshot_dir,
            enable_forecast_cache=args.enable_forecast_cache,
            vary_seed=args.vary_seed
        )
    elif args.subcommand == "audit-run":
        run_audit_run(args.run_dir)


def run_audit_run(run_dir: str) -> Dict[str, Any]:
    """Strictly audits a closed episode directory without loading simulator state."""
    run_path = Path(run_dir)
    try:
        if not run_path.is_dir():
            raise ValueError(f"Run directory not found: {run_dir}")
        manifest_file = run_path / "manifest.json"
        config_file = run_path / "config.yaml"
        if not config_file.exists():
            config_file = run_path / "config.json"
        events_file = run_path / "events.jsonl"
        metrics_file = run_path / "metrics.json"
        for required in (manifest_file, config_file, events_file, metrics_file, run_path / "timing.json", run_path / "decision_timing.jsonl"):
            if not required.exists():
                raise ValueError(f"Required artifact missing: {required.name}")

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        if manifest_data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Manifest schema version must be {SCHEMA_VERSION}.")
        if manifest_data.get("status") not in ("AUDITING", "COMPLETED", "EXTERNALLY_TRUNCATED"):
            raise ValueError(f"Run status is not auditable: {manifest_data.get('status')!r}.")
        run_id = manifest_data.get("run_id")
        episode_id = manifest_data.get("episode_id")
        if not run_id or not episode_id:
            raise ValueError("Manifest must contain nonempty run_id and episode_id.")

        with open(config_file, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) if config_file.suffix == ".yaml" else json.load(f)
        field_names = {field.name for field in dataclasses.fields(EpisodeSpec)}
        spec = EpisodeSpec(**{key: value for key, value in config_data.items() if key in field_names})

        prefix_cache_path = run_path.parent / ".audit_prefix_cache.json"
        prefix_cache_key = _audit_prefix_cache_key(spec, manifest_data)
        prefix_cache = _load_audit_prefix_cache(prefix_cache_path)
        cached_prefix = prefix_cache.get("entries", {}).get(prefix_cache_key)
        prefix_cache_candidate = isinstance(cached_prefix, dict)
        prefix_hash = hashlib.sha256()
        prefix_count = 0
        prefix_last_event_seq = None
        prefix_last_time = None
        prefix_complete = False
        cache_lightweight_ok = True

        events_hasher = hashlib.sha256()
        events_bytes = 0
        events_count = 0
        event_call_count = 0
        event_calls: Dict[Any, Dict[str, Any]] = {}
        action_ids = set()
        prev_event = None
        with events_file.open("rb") as stream:
            for line_no, raw_line in enumerate(stream, start=1):
                events_hasher.update(raw_line)
                events_bytes += len(raw_line)
                payload = raw_line.rstrip(b"\r\n")
                if not payload.strip():
                    raise ValueError(f"Blank event line at line {line_no}.")
                ev = json.loads(payload)
                events_count += 1
                in_prefix = (
                    not prefix_complete
                    and isinstance(ev.get("time"), (int, float))
                    and ev.get("time") < spec.score_start
                )
                if in_prefix:
                    prefix_count += 1
                    prefix_last_event_seq = ev.get("event_seq")
                    prefix_last_time = ev.get("time")
                    prefix_hash.update(_canonical_audit_prefix_line(payload))
                    if prefix_cache_candidate:
                        try:
                            _validate_cached_prefix_identity(ev, run_id, episode_id, prev_event)
                        except ValueError:
                            cache_lightweight_ok = False
                    else:
                        validate_event_dict(ev, prev_event=prev_event, spec=spec, expected_run_id=run_id, expected_episode_id=episode_id)
                else:
                    prefix_complete = True
                    validate_event_dict(ev, prev_event=prev_event, spec=spec, expected_run_id=run_id, expected_episode_id=episode_id)
                if ev.get("event_type") == "solver_invocation":
                    event_call_count += 1
                    event_calls[ev["details"]["call_id"]] = ev["details"]
                elif ev.get("event_type") == "action":
                    action_ids.add(ev["details"]["decision_id"])
                prev_event = ev

        actual_sha256 = events_hasher.hexdigest()
        if manifest_data.get("events_sha256") != actual_sha256:
            raise ValueError(f"events.jsonl SHA256 mismatch: expected {manifest_data.get('events_sha256')}, got {actual_sha256}.")
        if manifest_data.get("events_bytes") != events_bytes:
            raise ValueError(f"events.jsonl byte length mismatch: expected {manifest_data.get('events_bytes')}, got {events_bytes}.")
        if events_count != manifest_data.get("events_count"):
            raise ValueError(f"Event count mismatch: manifest={manifest_data.get('events_count')}, file={events_count}.")

        current_prefix = {
            "prefix_event_count": prefix_count,
            "prefix_sha256": prefix_hash.hexdigest(),
            "prefix_last_event_seq": prefix_last_event_seq,
            "prefix_last_time": prefix_last_time,
        }
        cache_hit = bool(
            prefix_cache_candidate
            and cache_lightweight_ok
            and all(cached_prefix.get(key) == value for key, value in current_prefix.items())
        )
        if prefix_cache_candidate and not cache_hit:
            # A cache mismatch never passes silently. Revalidate the complete
            # prefix before continuing with the already validated suffix.
            prev_prefix_event = None
            for prefix_event in _iter_jsonl_events(events_file):
                if prefix_event.get("time", spec.score_start) >= spec.score_start:
                    break
                validate_event_dict(
                    prefix_event,
                    prev_event=prev_prefix_event,
                    spec=spec,
                    expected_run_id=run_id,
                    expected_episode_id=episode_id,
                )
                prev_prefix_event = prefix_event

        invocation_paths = [
            run_path / "solver_invocations.jsonl",
            run_path / "logs" / "solver_invocations.jsonl",
        ]
        started_paths = [
            run_path / "solver_invocations.started.jsonl",
            run_path / "logs" / "solver_invocations.started.jsonl",
        ]
        invocation_file = next((path for path in invocation_paths if path.is_file()), None)
        started_file = next((path for path in started_paths if path.is_file()), None)
        checks_enabled = bool(manifest_data.get("checks_enabled", manifest_data.get("enable_checks", False)))
        if (checks_enabled or event_call_count) and (invocation_file is None or started_file is None):
            raise ValueError("Check enabled run requires final and write ahead solver invocation files.")
        solver_invocations = None
        if invocation_file is not None:
            solver_invocations = [json.loads(line) for line in invocation_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if started_file is not None:
            started_rows = [json.loads(line) for line in started_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            started_ids = [row.get("call_id") for row in started_rows]
            final_ids = [row.get("call_id") for row in (solver_invocations or [])]
            if len(started_ids) != len(set(started_ids)) or len(final_ids) != len(set(final_ids)):
                raise ValueError("Duplicate call_id in solver invocation records.")
            if set(started_ids) != set(final_ids):
                raise ValueError(f"Started and final solver IDs differ: started={set(started_ids)}, final={set(final_ids)}.")
            if any(row.get("status") == "STARTED" for row in (solver_invocations or [])):
                raise ValueError("Completed run has an unfinished solver invocation.")
        if solver_invocations is not None:
            calls_by_window: Dict[int, int] = {}
            calls_by_decision: Dict[int, int] = {}
            for row in solver_invocations:
                call_id = row.get("call_id")
                if row.get("run_id") != run_id or row.get("episode_id") != episode_id:
                    raise ValueError(f"Solver record {call_id} has wrong run or episode identity.")
                event_call = event_calls.get(call_id)
                if event_call is None:
                    raise ValueError(f"Solver record {call_id} has no matching solver_invocation event.")
                for field in ("decision_id", "quota_window", "scenario_id", "scenario_seed", "side", "T_max", "worker_count"):
                    if row.get(field) != event_call.get(field):
                        raise ValueError(f"Solver call {call_id} field {field} differs between event and record.")
                decision_id = int(row["decision_id"])
                window_id = int(row["quota_window"])
                if decision_id not in action_ids:
                    raise ValueError(f"Solver call {call_id} refers to unknown action decision {decision_id}.")
                calls_by_window[window_id] = calls_by_window.get(window_id, 0) + 1
                calls_by_decision[decision_id] = calls_by_decision.get(decision_id, 0) + 1
            if any(count > spec.B_call for count in calls_by_window.values()):
                raise ValueError(f"Solver call quota exceeded: {calls_by_window} with B_call={spec.B_call}.")
            if any(count > 2 * spec.K for count in calls_by_decision.values()):
                raise ValueError(f"A decision exceeded 2*K calls: {calls_by_decision} with K={spec.K}.")

        reduced_metrics = reduce_replay_events(
            _iter_jsonl_events(events_file), spec, solver_invocations=solver_invocations,
            expected_run_id=run_id, expected_episode_id=episode_id,
            events_validated=True,
        )
        online_metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        if online_metrics.get("run_id") != run_id or online_metrics.get("episode_id") != episode_id:
            raise ValueError("Metrics run_id or episode_id does not match manifest.")
        if online_metrics.get("status") != reduced_metrics.get("status"):
            raise ValueError(f"Metrics status {online_metrics.get('status')!r} disagrees with terminal status {reduced_metrics.get('status')!r}.")
        ok, mismatch = reconcile_metrics(
            online_metrics, reduced_metrics,
            check_rewards=manifest_data.get("command") in ("train", "evaluate", "rl_train", "rl_eval"),
        )
        if not ok:
            raise ValueError(f"Reconciliation mismatch: {mismatch}")

        prefix_cache.setdefault("entries", {})[prefix_cache_key] = current_prefix
        _store_audit_prefix_cache(prefix_cache_path, prefix_cache)

        diagnostic = {
            "status": "PASSED", "run_dir": str(run_path.resolve()),
            "schema_version": SCHEMA_VERSION, "run_id": run_id, "episode_id": episode_id,
            "events_count": events_count, "events_sha256": actual_sha256,
            "prefix_audit_cache": "hit" if cache_hit else "miss",
            "prefix_event_count": prefix_count,
            "useful_spot_gpu_seconds_score": reduced_metrics["useful_spot_gpu_seconds_score"],
            "hp_gang_cohort_count": reduced_metrics["hp_gang_cohort_count"],
            "hp_gang_violation_count": reduced_metrics["hp_gang_violation_count"],
            "spot_eviction_events_score": reduced_metrics["spot_eviction_events_score"],
            "solver_calls_score": reduced_metrics["solver_calls_score"],
        }
        print("=================================================================")
        print("AUDIT-RUN: PASSED")
        print("=================================================================")
        print(json.dumps(diagnostic, indent=2, allow_nan=False))
        return diagnostic
    except Exception as exc:
        diagnostic = {"status": "FAILED", "run_dir": str(run_path.resolve()), "error": str(exc)}
        try:
            (run_path / "first_mismatch.json").write_text(json.dumps(diagnostic, indent=2, allow_nan=False), encoding="utf-8")
        except OSError:
            pass
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


def finalize_run_artifacts(run_dir: str, status: str = "COMPLETED") -> Dict[str, Any]:
    """Set an auditing state, require a passing disk audit, then commit final status."""
    manifest_path = Path(run_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "AUDITING"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    try:
        result = run_audit_run(run_dir)
    except SystemExit:
        manifest["status"] = "ABORTED"
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        raise
    manifest["status"] = status
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    main()

