"""
warmup_snapshot.py: High-fidelity warmup state snapshot manager for ResearchSimulator.
Caches neutral cluster allocations, queues, recorder history, estimator neural weights,
quotas, ledger boundary, and random generators at score_start (e.g. May 1 00:00:00).
Restoring a snapshot reproduces bit-for-bit parity with replay from trace time zero
while eliminating repeated 9.5-minute warmup during RL training episodes.
"""

import hashlib
import os
import pickle
import logging
from pathlib import Path
from typing import Optional, Union, Any, Dict
import pandas as pd

from .types import EpisodeSpec, ResearchSimError, SimulationEvent
from .verifier import VerifierProtocol

DEFAULT_SNAPSHOT_DIR = "runs/warmup_snapshots"
SNAPSHOT_SCHEMA_VERSION = "3.1"
CREDIT_ACCOUNTING_VERSION = "completion_credit_cap_v2"
RESOURCE_ACCOUNTING_VERSION = "declared_gpu_units_v1"
MAX_SNAPSHOT_FILENAME_BYTES = 240


def _snapshot_stress_suffix(spec: EpisodeSpec) -> str:
    suffix = (
        f"_spot{int(spec.spot_arrival_multiplier)}_seed{int(spec.spot_arrival_seed)}"
        if int(spec.spot_arrival_multiplier) != 1 or int(spec.spot_arrival_seed) != 0
        else ""
    )
    if spec.spot_stress_start is not None:
        suffix += f"_from{int(spec.spot_stress_start)}"
    if getattr(spec, "candidate_geometry_version", "legacy_v1") != "legacy_v1":
        suffix += f"_cand{spec.candidate_geometry_version}"
    return suffix


def get_snapshot_key(spec: EpisodeSpec) -> str:
    """Generates a deterministic snapshot key based on warmup configuration."""
    clean_filter = spec.node_filter.replace("-", "_")
    clean_split = str(getattr(spec, "split_id", "train")).replace("-", "_")
    stress_suffix = _snapshot_stress_suffix(spec)
    provider = str(spec.forecast_provider_id).replace("-", "_")
    provider_version = str(spec.forecast_provider_version).replace("-", "_")
    return (
        f"warmup_{clean_filter}_{spec.trace_mode}_"
        f"s{spec.score_start}_e{spec.score_end}_f{spec.followup_end}_"
        f"split{clean_split}_envseed{spec.environment_seed}_"
        f"forecast{provider}_{provider_version}{stress_suffix}_"
        f"acct{CREDIT_ACCOUNTING_VERSION}_resource{RESOURCE_ACCOUNTING_VERSION}_"
        f"gpu{spec.gpu_accounting_mode}_v{spec.rule_version}.pkl"
    )


def get_snapshot_path(spec: EpisodeSpec, snapshot_dir: str = DEFAULT_SNAPSHOT_DIR) -> Path:
    """Return current accounting snapshot path, never an older credit state."""
    key = get_snapshot_key(spec)
    if len(key.encode("utf-8")) > MAX_SNAPSHOT_FILENAME_BYTES:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
        clean_filter = spec.node_filter.replace("-", "_")
        clean_split = str(getattr(spec, "split_id", "train")).replace("-", "_")[:32]
        key = (
            f"warmup_{clean_filter}_s{spec.score_start}_split{clean_split}_"
            f"{digest}_v{spec.rule_version}.pkl"
        )
    return Path(snapshot_dir) / key


def has_warmup_snapshot(spec: EpisodeSpec, snapshot_dir: str = DEFAULT_SNAPSHOT_DIR) -> bool:
    """Checks whether a valid snapshot exists for the given specification."""
    return get_snapshot_path(spec, snapshot_dir).is_file()


def _snapshot_split_matches(
    saved_split_id: Optional[str], requested_split_id: str, allow_history_prefix_reuse: bool
) -> bool:
    if saved_split_id == requested_split_id:
        return True
    return bool(
        allow_history_prefix_reuse
        and saved_split_id in {"train", "val", "test"}
        and str(requested_split_id).startswith("history_")
    )


def save_warmup_snapshot(sim: Any, filepath: Union[str, Path], extra_metadata: Optional[Dict[str, Any]] = None) -> Path:
    """
    Serializes simulator state at score_start to the specified filepath.
    Ensures atomic write via temporary file and embeds provenance metadata.
    Schema version: 3.1.
    """
    target = Path(filepath)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(".tmp")

    metadata = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "node_filter": sim.spec.node_filter,
        "calendar_anchor": sim.spec.calendar_anchor,
        "warmup_start": sim.spec.warmup_start,
        "score_start": sim.spec.score_start,
        "score_end": sim.spec.score_end,
        "followup_end": sim.spec.followup_end,
        "retry_mode": sim.spec.retry_mode,
        "trace_mode": sim.spec.trace_mode,
        "forecast_mode": sim.spec.forecast_mode,
        "rule_version": sim.spec.rule_version,
        "split_id": getattr(sim.spec, "split_id", "train"),
        "environment_seed": sim.spec.environment_seed,
        "policy_seed": sim.spec.policy_seed,
        "forecast_provider_id": sim.spec.forecast_provider_id,
        "forecast_provider_version": sim.spec.forecast_provider_version,
        "spot_arrival_multiplier": int(sim.spec.spot_arrival_multiplier),
        "spot_arrival_seed": int(sim.spec.spot_arrival_seed),
        "spot_stress_start": sim.spec.spot_stress_start,
        "candidate_geometry_version": getattr(sim.spec, "candidate_geometry_version", "legacy_v1"),
        "gpu_accounting_mode": sim.spec.gpu_accounting_mode,
        "prefix_workload_sha256": getattr(sim.view, "prefix_workload_sha256", ""),
        "warmup_policy": getattr(sim, "policy_type", "native_gfs"),
        "node_sha256": getattr(sim.view, "hashes", {}).get("node_sha256", ""),
        "job_sha256": getattr(sim.view, "hashes", {}).get("job_sha256", ""),
        "sim_time": sim.time,
        "phase": sim.phase.value if hasattr(sim.phase, "value") else str(sim.phase),
        "code_version": "3.0",
        "credit_accounting_version": CREDIT_ACCOUNTING_VERSION,
        "resource_accounting_version": RESOURCE_ACCOUNTING_VERSION,
        "has_estimator": hasattr(sim, "estimator") and sim.estimator is not None,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    payload = {
        "version": SNAPSHOT_SCHEMA_VERSION,
        "metadata": metadata,
        "sim": sim,
    }

    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    tmp_path.replace(target)
    return target


class WarmupSnapshotDone(Exception):
    """Raised to stop simulation immediately once the warmup snapshot is saved."""
    pass


def build_forecast_cache(sim: Any) -> Dict[int, Any]:
    """
    Precomputes Informer neural forecast outputs (pred_mu_dict, pred_sigma_dict)
    for every quota evaluation timestamp across the scored window.
    High-priority demand history is exogenous and deterministic, guaranteeing
    exact parity while eliminating redundant PyTorch forward passes and pandas churn.
    """
    org_request_df = sim.recorder.agg_seq_recorder()
    forecast_cache = {}
    t_start = sim.spec.score_start
    t_end = sim.spec.score_end
    for t in range(t_start, t_end + 3601, 300):
        now_timestamp = pd.Timestamp(sim.spec.calendar_anchor) + pd.Timedelta(seconds=t)
        pred_mu, pred_sigma = sim.estimator.test(org_request_df, now_timestamp)
        pred_mu_dict, pred_sigma_dict = {}, {}
        for i in range(len(org_request_df.columns)):
            org_name = org_request_df.columns[i]
            vc_name = org_name.split("&&", 1)[0]
            org_mu, org_sigma = pred_mu[0, :, i], pred_sigma[0, :, i]
            if vc_name not in pred_mu_dict:
                pred_mu_dict[vc_name] = org_mu
                pred_sigma_dict[vc_name] = org_sigma * org_sigma
            else:
                pred_mu_dict[vc_name] += org_mu
                pred_sigma_dict[vc_name] += org_sigma * org_sigma
        forecast_cache[t] = (pred_mu_dict, pred_sigma_dict)
    return forecast_cache


def load_warmup_snapshot(
    filepath: Union[str, Path],
    policy_type: Optional[str] = None,
    verifier: Optional[VerifierProtocol] = None,
    log_dir: Optional[str] = None,
    spec: Optional[EpisodeSpec] = None,
    view: Optional[Any] = None,
    duration_estimator: Optional[Any] = None,
    quota_provider: Optional[Any] = None,
    risk_model: Optional[Any] = None,
    action_filter: Optional[Any] = None,
    filter_mode: str = "passthrough",
    enable_forecast_cache: bool = False,
    allow_legacy: bool = False,
    run_id: Optional[str] = None,
    episode_id: Optional[str] = None,
    allow_history_prefix_reuse: bool = False,
) -> Any:
    """
    Loads and rebinds a fresh simulator instance from a warmup snapshot.
    Validates snapshot provenance metadata and data hashes against requested spec and view.
    Always creates a fresh deep-copied simulator and reinitializes dependent objects
    (ledger, budget, candidate generator, action filter, RNG state).
    Preserves all historical counters: completed_jobs, useful_spot_gpu_seconds,
    spot_lost_gpu_seconds, spot_evictions, total_solver_calls, and hp_gang_violations.
    Forecast cache is DISABLED to enforce true online causal forecasting.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Warmup snapshot file not found: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    snapshot_split_known = False
    if isinstance(data, dict) and "sim" in data and "metadata" in data:
        version = data.get("version", "1.0")
        if version != SNAPSHOT_SCHEMA_VERSION and not allow_legacy:
            raise ResearchSimError(
                f"Warmup snapshot schema version '{version}' rejected; required '{SNAPSHOT_SCHEMA_VERSION}'."
            )
        import copy
        source_sim = data["sim"]
        # Snapshot events and the trace view are immutable provenance data
        # during restore. Reusing them through deepcopy's memo avoids copying
        # hundreds of thousands of event objects for every RL episode. The
        # mutable ledger is replaced below, and the view is rebound explicitly.
        deepcopy_memo = {}
        source_ledger = getattr(source_sim, "ledger", None)
        source_events = getattr(source_ledger, "events", None)
        if source_events is not None:
            deepcopy_memo[id(source_events)] = source_events
            for event in source_events:
                deepcopy_memo[id(event)] = event
        source_trace = getattr(source_sim, "trace", None)
        if source_trace is not None:
            # Jobs already completed before score_start are immutable for the
            # remaining replay. Sharing these dictionaries avoids copying the
            # large historical prefix. Active and future jobs are still copied
            # so later arrivals, starts, and preemptions remain isolated.
            for job in getattr(source_trace, "job_list", ()):
                if isinstance(job, dict) and job.get("status") == "end":
                    deepcopy_memo[id(job)] = job
        if view is not None and hasattr(source_sim, "view"):
            deepcopy_memo[id(source_sim.view)] = view
        sim = copy.deepcopy(source_sim, deepcopy_memo)
        meta = data["metadata"]
        if meta.get("credit_accounting_version") != CREDIT_ACCOUNTING_VERSION:
            raise ResearchSimError("Warmup snapshot uses obsolete Spot completion credit accounting.")
        if meta.get("resource_accounting_version") != RESOURCE_ACCOUNTING_VERSION:
            raise ResearchSimError("Warmup snapshot uses obsolete GPU demand accounting.")
        snapshot_split_known = meta.get("split_id") is not None
        if spec is not None:
            if (
                meta.get("node_filter") != spec.node_filter or
                meta.get("calendar_anchor") != spec.calendar_anchor or
                meta.get("warmup_start") != spec.warmup_start or
                meta.get("score_start") != spec.score_start or
                meta.get("sim_time") is not None and int(meta.get("sim_time")) != int(spec.score_start) or
                meta.get("trace_mode") != spec.trace_mode or
                meta.get("forecast_mode") != spec.forecast_mode or
                int(meta.get("environment_seed", -1)) != int(spec.environment_seed) or
                meta.get("forecast_provider_id") != spec.forecast_provider_id or
                meta.get("forecast_provider_version") != spec.forecast_provider_version or
                meta.get("retry_mode", spec.retry_mode) != spec.retry_mode or
                int(meta.get("spot_arrival_multiplier", 1)) != int(spec.spot_arrival_multiplier) or
                int(meta.get("spot_arrival_seed", 0)) != int(spec.spot_arrival_seed) or
                meta.get("spot_stress_start") != spec.spot_stress_start or
                meta.get("candidate_geometry_version", "legacy_v1") != spec.candidate_geometry_version or
                meta.get("gpu_accounting_mode") != spec.gpu_accounting_mode or
                meta.get("rule_version") != spec.rule_version
                or (
                    meta.get("split_id") is not None
                    and not _snapshot_split_matches(
                        str(meta.get("split_id")),
                        str(getattr(spec, "split_id", "train")),
                        allow_history_prefix_reuse,
                    )
                )
            ):
                raise ResearchSimError(f"Warmup snapshot metadata mismatch: {meta} vs requested spec: {spec}")
            if view is not None and hasattr(view, "hashes"):
                saved_node_hash = meta.get("node_sha256")
                saved_job_hash = meta.get("job_sha256")
                curr_node_hash = view.hashes.get("node_sha256")
                curr_job_hash = view.hashes.get("job_sha256")
                if saved_node_hash and curr_node_hash and saved_node_hash != curr_node_hash:
                    raise ResearchSimError(f"Snapshot node hash mismatch: {saved_node_hash} vs current: {curr_node_hash}")
                if saved_job_hash and curr_job_hash and saved_job_hash != curr_job_hash:
                    raise ResearchSimError(f"Snapshot job hash mismatch: {saved_job_hash} vs current: {curr_job_hash}")
                saved_prefix_hash = meta.get("prefix_workload_sha256")
                current_prefix_hash = getattr(view, "prefix_workload_sha256", None)
                if saved_prefix_hash and current_prefix_hash and saved_prefix_hash != current_prefix_hash:
                    raise ResearchSimError(
                        f"Snapshot historical workload mismatch: {saved_prefix_hash} vs current: {current_prefix_hash}"
                    )
    else:
        if not allow_legacy:
            raise ResearchSimError(
                f"Legacy warmup snapshot format rejected; required '{SNAPSHOT_SCHEMA_VERSION}'."
            )
        import copy
        sim = copy.deepcopy(data)

    # The v4 controller is interval state, not physical warmup state. Recreate
    # it after every restore so v3 snapshots remain valid causal prefixes.
    from .control import WeightedAdmissionRanker
    sim.control_paused_context = None
    sim.admission_ranker = WeightedAdmissionRanker()
    sim.spot_queue_gpu_seconds_delta = 0.0

    from .metrics import EventLedger
    from .candidates import CandidateGenerator
    from .verifier import BudgetManager
    from .risk import ActionFilter
    import random
    import numpy as np

    effective_spec = spec if spec is not None else sim.spec
    if spec is not None:
        if (
            sim.spec.warmup_start != spec.warmup_start or
            sim.spec.score_start != spec.score_start or
            sim.spec.node_filter != spec.node_filter or
            sim.spec.trace_mode != spec.trace_mode or
            sim.spec.forecast_mode != spec.forecast_mode or
            int(getattr(sim.spec, "environment_seed", -1)) != int(spec.environment_seed) or
            getattr(sim.spec, "forecast_provider_id", None) != spec.forecast_provider_id or
            getattr(sim.spec, "forecast_provider_version", None) != spec.forecast_provider_version or
            sim.spec.calendar_anchor != spec.calendar_anchor or
            sim.spec.retry_mode != spec.retry_mode or
            getattr(sim.spec, "candidate_geometry_version", "legacy_v1") != spec.candidate_geometry_version or
            getattr(sim.spec, "gpu_accounting_mode", None) != spec.gpu_accounting_mode or
            sim.spec.rule_version != spec.rule_version
            or (
                snapshot_split_known
                and getattr(sim.spec, "split_id", None) is not None
                and not _snapshot_split_matches(
                    str(getattr(sim.spec, "split_id")),
                    str(getattr(spec, "split_id", "train")),
                    allow_history_prefix_reuse,
                )
            )
        ):
            raise ResearchSimError("Supplied spec warmup attributes do not match snapshot attributes.")
        sim.spec = spec

    # Preserve all historical counters and events from snapshot ledger
    saved_ledger = getattr(sim, "ledger", None)
    saved_events = list(getattr(saved_ledger, "events", [])) if saved_ledger else []
    saved_hp_jobs = dict(getattr(saved_ledger, "hp_gang_jobs", {})) if saved_ledger else {}
    saved_hp_queue = list(getattr(saved_ledger, "hp_deadline_queue", [])) if saved_ledger else []
    saved_seq = getattr(saved_ledger, "_seq", 0)
    saved_prefix_viols = getattr(saved_ledger, "uncontrollable_prefix_violations", 0)
    saved_useful_spot = getattr(saved_ledger, "useful_spot_gpu_seconds", 0.0)
    saved_useful_spot_all_phase = getattr(saved_ledger, "useful_spot_gpu_seconds_all_phase", 0.0)
    saved_spot_lost = getattr(saved_ledger, "spot_lost_gpu_seconds", 0.0)
    saved_spot_lost_all_phase = getattr(saved_ledger, "spot_lost_gpu_seconds_all_phase", 0.0)
    saved_spot_evictions = getattr(saved_ledger, "spot_evictions", 0)
    saved_spot_evictions_all_phase = getattr(saved_ledger, "spot_evictions_all_phase", 0)
    saved_solver_calls = getattr(saved_ledger, "total_solver_calls", 0)
    saved_solver_calls_all_phase = getattr(saved_ledger, "solver_calls_all_phase", 0)
    saved_hp_viols = getattr(saved_ledger, "hp_gang_violations", 0)
    saved_completed = dict(getattr(saved_ledger, "completed_jobs", {})) if saved_ledger else {}

    # If completed_jobs was reset to zero or missing in snapshot, derive from event history
    if (saved_completed.get("HP", 0) == 0 and saved_completed.get("Spot", 0) == 0) and saved_events:
        for ev in saved_events:
            if getattr(ev, "event_type", None) == "completion":
                jtype = getattr(ev, "details", {}).get("job_type", "HP")
                saved_completed[jtype] = saved_completed.get(jtype, 0) + 1

    if saved_useful_spot_all_phase == 0.0 and saved_events:
        saved_useful_spot_all_phase = sum(
            float(getattr(ev, "details", {}).get("credited_gpu_seconds", getattr(ev, "details", {}).get("gpu_seconds", 0.0)))
            for ev in saved_events if getattr(ev, "event_type", None) == "credit"
        )
    if saved_spot_lost_all_phase == 0.0 and saved_events:
        saved_spot_lost_all_phase = sum(
            float(getattr(ev, "details", {}).get("lost_gpu_seconds", 0.0))
            for ev in saved_events if getattr(ev, "event_type", None) == "preemption"
        )
    if saved_spot_evictions_all_phase == 0 and saved_events:
        saved_spot_evictions_all_phase = sum(
            1 for ev in saved_events if getattr(ev, "event_type", None) == "preemption"
        )

    run_id_val = run_id or (
        os.path.basename(os.path.dirname(log_dir.rstrip("/\\")))
        if log_dir and os.path.basename(log_dir.rstrip("/\\")) == "logs"
        else os.path.basename(log_dir.rstrip("/\\")) if log_dir else f"run_{sim.episode_id}"
    )
    episode_id_val = episode_id or sim.episode_id
    sim.episode_id = episode_id_val
    sim.run_id = run_id_val
    sim.ledger = EventLedger(sim.episode_id, effective_spec, run_id=run_id_val)

    # A pre-window snapshot may have been created with a different score end
    # and therefore a narrower EpisodeTraceView. Its state at score_start is
    # reusable only when the current view is rebound before future replay.
    if view is not None:
        sim.view = view

    jobs_by_id = {}
    if hasattr(sim, "view") and hasattr(sim.view, "_jobs_df"):
        jobs_by_id = {row["job_name"]: row for row in sim.view._jobs_df.to_dict("records")}

    job_credit_seq = {}
    job_attempt_indices = {}
    migrated_events = []
    for seq_idx, ev in enumerate(saved_events):
        time_val = getattr(ev, "time", 0)
        etype = getattr(ev, "event_type", "unknown")
        jid = getattr(ev, "job_id", None)
        details = dict(getattr(ev, "details", {}))
        vc = getattr(ev, "vc_name", None)
        j = jobs_by_id.get(jid)
        jtype = j["job_type"] if j else details.get("job_type", "HP")

        needs_enrichment = (
            not hasattr(ev, "schema_version")
            or not hasattr(ev, "run_id")
            or not hasattr(ev, "phase")
            or getattr(ev, "schema_version", "") != "2.0"
            or (etype == "arrival" and "worker_num" not in details)
        )

        if needs_enrichment:
            raise ResearchSimError(
                f"Warmup snapshot contains legacy or incomplete event schema at seq {seq_idx} "
                f"({etype}). Regenerate the snapshot from trace time zero; do not infer historical measurements."
            )

        if etype == "start" and details.get("job_type") == "Spot":
            att = getattr(ev, "attempt_index", None)
            if att is None:
                att = details.get("attempt_index", 0)
            job_attempt_indices[jid] = max(job_attempt_indices.get(jid, -1), att)
        elif etype == "credit":
            c_seq = details.get("credit_sequence", 0)
            job_credit_seq[jid] = max(job_credit_seq.get(jid, 0), c_seq + 1)
        migrated_events.append(SimulationEvent(
            schema_version=ev.schema_version,
            run_id=sim.run_id,
            episode_id=sim.episode_id,
            event_seq=ev.event_seq,
            time=time_val,
            phase=ev.phase,
            event_type=etype,
            job_id=jid,
            vc_name=vc,
            attempt_index=ev.attempt_index,
            gpu_count=ev.gpu_count,
            details=details,
        ))

    sim.ledger.events = migrated_events
    sim.ledger.hp_gang_jobs = saved_hp_jobs
    sim.ledger.hp_deadline_queue = saved_hp_queue
    sim.ledger._seq = saved_seq
    sim.ledger.uncontrollable_prefix_violations = saved_prefix_viols
    sim.ledger.useful_spot_gpu_seconds_all_phase = saved_useful_spot_all_phase
    sim.ledger.spot_lost_gpu_seconds_all_phase = saved_spot_lost_all_phase
    sim.ledger.spot_evictions_all_phase = saved_spot_evictions_all_phase
    sim.ledger.solver_calls_all_phase = saved_solver_calls_all_phase
    sim.ledger.hp_gang_violations = saved_hp_viols
    sim.ledger.completed_jobs = saved_completed

    # Preserve legacy / cumulative attributes on ledger for Task 5.1 regression tests
    sim.ledger.useful_spot_gpu_seconds = saved_useful_spot
    sim.ledger.spot_lost_gpu_seconds = saved_spot_lost
    sim.ledger.spot_evictions = saved_spot_evictions
    sim.ledger.total_solver_calls = saved_solver_calls

    # Scored phase accumulators start clean at score_start
    sim.ledger.scored_useful_spot_gpu_seconds = 0.0
    sim.ledger.scored_spot_lost_gpu_seconds = 0.0
    sim.ledger.scored_spot_evictions = 0
    sim.ledger.scored_solver_calls = 0
    sim.ledger.unique_spot_jobs_evicted_score = set()
    sim.ledger.decision_count = 0
    sim.ledger.scored_spot_starts = 0

    # All phase tracking
    sim.ledger.unique_spot_jobs_evicted_all_phase = {
        getattr(ev, "job_id", None) for ev in migrated_events if getattr(ev, "event_type", None) == "preemption"
    } - {None}
    sim.ledger.all_spot_starts = sum(
        1 for ev in migrated_events
        if getattr(ev, "event_type", None) == "start" and (
            getattr(ev, "details", {}).get("job_type") == "Spot" or jobs_by_id.get(getattr(ev, "job_id", None), {}).get("job_type") == "Spot"
        )
    )

    if not hasattr(sim, "_spot_attempt_indices") or not sim._spot_attempt_indices:
        sim._spot_attempt_indices = job_attempt_indices
    if not hasattr(sim, "_spot_credit_sequences") or not sim._spot_credit_sequences:
        sim._spot_credit_sequences = job_credit_seq
    if not hasattr(sim, "prefix_reward_gpu_hours"):
        sim.prefix_reward_gpu_hours = 0.0
    if not hasattr(sim, "prefix_solver_calls"):
        sim.prefix_solver_calls = 0
    if not hasattr(sim, "prefix_deadline_events"):
        sim.prefix_deadline_events = 0
    if not hasattr(sim, "prefix_evictions"):
        sim.prefix_evictions = 0
    if not hasattr(sim, "prefix_lost_gpu_seconds"):
        sim.prefix_lost_gpu_seconds = 0.0
    if not hasattr(sim, "_first_decision_returned"):
        sim._first_decision_returned = False
    if not hasattr(sim, "step_reward_gpu_hours_sum"):
        sim.step_reward_gpu_hours_sum = 0.0
    if not hasattr(sim, "admission_queue_scans"):
        sim.admission_queue_scans = 0
    if not hasattr(sim, "admission_reordered_scans"):
        sim.admission_reordered_scans = 0
    if not hasattr(sim, "admission_duration_nonconstant_scans"):
        sim.admission_duration_nonconstant_scans = 0
    if not hasattr(sim, "decision_timing_records"):
        sim.decision_timing_records = []
    if not hasattr(sim, "timer_func"):
        import time as _sys_time
        sim.timer_func = _sys_time.perf_counter_ns
    if not hasattr(sim, "_last_candidate_start_ns"):
        sim._last_candidate_start_ns = None
    if not hasattr(sim, "_last_candidate_end_ns"):
        sim._last_candidate_end_ns = None
    if not hasattr(sim, "_last_candidate_gen_ms"):
        sim._last_candidate_gen_ms = 0.0
    if not hasattr(sim, "_decision_paused_time_ns"):
        sim._decision_paused_time_ns = None

    if log_dir:
        budget_log_dir = log_dir
        sim.run_id = run_id_val
    else:
        budget_log_dir = None
        sim.run_id = run_id_val

    sim.budget = BudgetManager(
        B_call=effective_spec.B_call,
        K=effective_spec.K,
        log_dir=budget_log_dir,
        ledger=sim.ledger,
        run_id=sim.run_id,
        episode_id=sim.episode_id
    )
    sim.budget.on_second(sim.time)

    sim.candidate_generator = CandidateGenerator(
        M=effective_spec.M,
        geometry_version=effective_spec.candidate_geometry_version,
    )
    if hasattr(sim, "args") and sim.args is not None:
        sim.args.log_range = (effective_spec.score_start, effective_spec.score_end)

    if policy_type is not None:
        sim.policy_type = policy_type
    if verifier is not None:
        sim.verifier = verifier

    sim.duration_estimator = duration_estimator
    sim.quota_provider = quota_provider
    if effective_spec.forecast_mode == "external_causal":
        if quota_provider is None:
            raise ResearchSimError("external_causal snapshot restore requires quota_provider.")
        if str(getattr(quota_provider, "provider_id", "")) != effective_spec.forecast_provider_id:
            raise ResearchSimError("Snapshot quota_provider id does not match requested spec.")
        if str(getattr(quota_provider, "provider_version", "")) != effective_spec.forecast_provider_version:
            raise ResearchSimError("Snapshot quota_provider version does not match requested spec.")
    sim.risk_model = risk_model
    sim.filter_mode = filter_mode
    if action_filter is not None:
        sim.action_filter = action_filter
    else:
        sim.action_filter = ActionFilter(mode=filter_mode, risk_model=risk_model, verifier=getattr(sim, "verifier", None))

    if log_dir is not None and hasattr(sim, "args") and sim.args is not None:
        sim.args.log_dir = log_dir
        sim.args.checkpoints = os.path.join(log_dir, "checkpoints")

    if spec is not None:
        random.seed(spec.environment_seed)
        np.random.seed(spec.environment_seed)
        try:
            import torch
            torch.manual_seed(spec.environment_seed)
        except ImportError:
            pass

    if enable_forecast_cache and sim.spec.forecast_mode == "native_online":
        fcache_path = path.with_suffix(".fcache.pkl")
        if fcache_path.is_file():
            with open(fcache_path, "rb") as f:
                sim.forecast_cache = pickle.load(f)
        else:
            sim.forecast_cache = build_forecast_cache(sim)
            try:
                with open(fcache_path, "wb") as f:
                    pickle.dump(sim.forecast_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                pass
    else:
        sim.forecast_cache = None

    return sim


def generate_warmup_snapshot(
    spec: EpisodeSpec,
    view: Any,
    snapshot_dir: str = DEFAULT_SNAPSHOT_DIR,
    log_dir: Optional[str] = None,
    progress_hook: Optional[Any] = None,
    quota_provider: Optional[Any] = None,
) -> Path:
    """
    Simulates warmup from spec.warmup_start up to spec.score_start,
    saves snapshot at score_start, and halts simulation immediately.
    """
    snapshot_path = get_snapshot_path(spec, snapshot_dir)
    if snapshot_path.is_file():
        return snapshot_path

    from .core import ResearchSimulator

    def hook(sim):
        raise WarmupSnapshotDone()

    sim = ResearchSimulator(
        spec=spec,
        view=view,
        policy_type="native_gfs",
        quota_provider=quota_provider,
        save_warmup_snapshot_to=str(snapshot_path),
        log_dir=log_dir or "/tmp/warmup_gen_logs"
    )
    sim._warmup_snapshot_hook = hook
    if progress_hook:
        sim.progress_hook = progress_hook

    try:
        while True:
            context, delta = sim.advance_until_decision()
            if delta.terminated or delta.truncated:
                break
    except WarmupSnapshotDone:
        pass

    if not snapshot_path.is_file():
        raise ResearchSimError(f"Failed to generate warmup snapshot at {snapshot_path}")

    return snapshot_path

