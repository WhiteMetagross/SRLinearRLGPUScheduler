"""Protocol helpers used by the final scheduling benchmark.

This module isolates replay preparation from the discarded placement RL CLI.
Only the causal duration fit and frozen protocol checks required by the paper
experiments are retained.
"""

from __future__ import annotations

import dataclasses
import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from research_sim.experiment_protocol import build_training_blocks, validate_paper_split
from research_sim.freeze import validate_test_access_grant, verify_frozen_bundle
from research_sim.parallel import plan_parallelism
from research_sim.scenarios import (
    PastFittedScenarioGenerator,
    build_observed_completion_history,
    build_right_censored_service_history,
)
from research_sim.trace import EpisodeTraceView, TraceStore
from research_sim.types import EpisodeSpec, NATIVE_GFS_MIN_HISTORY_SECONDS
from research_sim.verifier import NoCheckVerifier
from research_sim.warmup_snapshot import (
    generate_warmup_snapshot,
    get_snapshot_path,
    has_warmup_snapshot,
    load_warmup_snapshot,
)


METHOD_OVERRIDE_KEYS = {
    "policy_seed",
    "forecast_mode",
    "forecast_provider_id",
    "forecast_provider_version",
    "spot_arrival_seed",
    "device",
    "lr",
    "gamma",
    "gae_lambda",
    "ppo_epochs",
    "mini_batch_size",
    "delta_H",
    "delta_V",
    "eta_H",
    "eta_V",
    "ensemble_size",
    "risk_epochs",
    "risk_lr",
    "kappa",
    "tau_safe",
    "method_id",
    "method_version",
}


def load_yaml(config_path: str) -> dict[str, Any]:
    """Load a simulator config and resolve declared protocol paths."""
    path = Path(config_path).resolve()
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    parent_value = loaded.pop("environment_config", None)
    if parent_value is None:
        for key in ("experiment_manifest", "environment_lock", "test_access_grant"):
            value = loaded.get(key)
            if value:
                candidate = Path(value)
                if not candidate.is_absolute():
                    loaded[key] = str((path.parent / candidate).resolve())
        return loaded
    parent_path = Path(parent_value)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    parent = load_yaml(str(parent_path))
    forbidden = sorted(set(loaded) - METHOD_OVERRIDE_KEYS)
    if forbidden:
        raise ValueError(
            "Method config may override only declared method fields. "
            f"forbidden fields={forbidden}."
        )
    return {**parent, **loaded}


def build_spec_from_dict(cfg: dict[str, Any]) -> EpisodeSpec:
    """Build the immutable episode contract used by all final methods."""
    spec = EpisodeSpec(
        node_filter=cfg.get("node_filter", "A100-SXM4-80GB"),
        calendar_anchor=cfg.get("calendar_anchor", "2024-03-01 00:00:00"),
        warmup_start=cfg.get("warmup_start", 0),
        score_start=cfg.get("score_start", 5270400),
        score_end=cfg.get("score_end", 5356800),
        followup_end=cfg.get("followup_end", 5360400),
        retry_mode=cfg.get("retry_mode", "next_event"),
        trace_mode=cfg.get("trace_mode", "research_full_history"),
        forecast_mode=cfg.get("forecast_mode", "native_online"),
        rule_version=str(cfg.get("rule_version", "1.0")),
        split_id=str(cfg.get("split", "train")),
        environment_seed=int(cfg.get("environment_seed", 20261002)),
        policy_seed=int(cfg.get("policy_seed", 42)),
        forecast_provider_id=str(cfg.get("forecast_provider_id", "builtin")),
        forecast_provider_version=str(cfg.get("forecast_provider_version", "1.0")),
        spot_arrival_multiplier=int(cfg.get("spot_arrival_multiplier", 1)),
        spot_arrival_seed=int(cfg.get("spot_arrival_seed", 0)),
        spot_stress_start=(
            int(cfg["spot_stress_start"])
            if cfg.get("spot_stress_start") is not None
            else None
        ),
        L_H=int(cfg.get("L_H", 3600)),
        B_call=int(cfg.get("B_call", 2000)),
        K=int(cfg.get("K", 4)),
        M=int(cfg.get("M", 8)),
        candidate_geometry_version=str(cfg.get("candidate_geometry_version", "legacy_v1")),
        gpu_accounting_mode=str(cfg.get("gpu_accounting_mode", "auto")),
        control_interval_seconds=int(cfg.get("control_interval_seconds", 300)),
        control_queue_size=int(cfg.get("control_queue_size", 64)),
        max_quota_delta_fraction=float(cfg.get("max_quota_delta_fraction", 1.0)),
    )
    if spec.forecast_mode == "native_online" and spec.score_start < NATIVE_GFS_MIN_HISTORY_SECONDS:
        raise ValueError(
            "Native forecasting requires at least "
            f"{NATIVE_GFS_MIN_HISTORY_SECONDS} seconds of causal recorder history. "
            f"score_start={spec.score_start}."
        )
    return spec


def _paper_manifest_for(
    config_path: str,
    cfg: dict[str, Any],
    spec: EpisodeSpec,
) -> dict[str, Any] | None:
    """Verify the frozen environment before a paper replay starts."""
    mode = str(cfg.get("experiment_mode", "smoke")).lower()
    if mode == "smoke":
        return None
    if mode != "paper":
        raise ValueError("experiment_mode must be smoke or paper.")
    manifest_value = cfg.get("experiment_manifest")
    if not manifest_value:
        raise ValueError("Paper configs must declare experiment_manifest.")
    manifest_path = Path(manifest_value)
    if not manifest_path.is_absolute():
        manifest_path = Path(config_path).resolve().parent / manifest_path
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Experiment manifest must be a YAML mapping: {manifest_path}")
    validate_paper_split(cfg, spec, manifest)
    lock_value = cfg.get("environment_lock")
    if not lock_value:
        raise ValueError("Paper configs must declare environment_lock.")
    lock_path = Path(lock_value)
    if not lock_path.is_absolute():
        lock_path = Path(config_path).resolve().parent / lock_path
    verify_frozen_bundle(lock_path)
    if str(cfg.get("split", "")).lower() == "test":
        grant_value = cfg.get("test_access_grant")
        if not grant_value:
            raise ValueError("The final test split remains locked.")
        grant_path = Path(grant_value)
        if not grant_path.is_absolute():
            grant_path = Path(config_path).resolve().parent / grant_path
        validate_test_access_grant(grant_path, protocol_id=str(manifest["protocol_id"]))
    return manifest


def _validate_trace_hashes(
    manifest: dict[str, Any] | None,
    view: EpisodeTraceView,
) -> None:
    """Reject a trace that does not match the frozen source hashes."""
    if manifest is None:
        return
    expected = manifest.get("trace_hashes")
    actual = getattr(view, "hashes", {})
    if not isinstance(expected, dict) or expected != actual:
        raise ValueError(
            "Trace hashes do not match the frozen manifest. "
            f"Expected {expected!r}, observed {actual!r}."
        )


def run_preflight(config_path: str) -> dict[str, Any]:
    """Validate protocol, trace hashes, snapshots, and CPU resources."""
    cfg = load_yaml(config_path)
    spec = build_spec_from_dict(cfg)
    manifest = _paper_manifest_for(config_path, cfg, spec)
    blocks = (
        build_training_blocks(cfg, spec, manifest=manifest)
        if cfg.get("split") == "train"
        else ()
    )
    if manifest is not None:
        store = TraceStore.get_instance(cfg.get("trace_dir", "data/raw/spot_gpu_v2026"))
        view = store.create_episode_view(spec)
        _validate_trace_hashes(manifest, view)
    snapshot_requested = bool(cfg.get("use_warmup_snapshot", False))
    snapshot_dir = cfg.get("snapshot_dir", "runs/warmup_snapshots")
    snapshot_specs = tuple(block.spec for block in blocks) if blocks else (spec,)
    availability = (
        [has_warmup_snapshot(item, snapshot_dir) for item in snapshot_specs]
        if snapshot_requested
        else None
    )
    result = {
        "status": "PAPER_PREFLIGHT_PASSED" if manifest is not None else "SMOKE_CONFIG_ONLY",
        "split": cfg.get("split"),
        "score_start": spec.score_start,
        "score_end": spec.score_end,
        "followup_end": spec.followup_end,
        "manifest_id": manifest.get("protocol_id") if manifest else None,
        "warmup_snapshot_requested": snapshot_requested,
        "warmup_snapshot_available": all(availability) if availability is not None else None,
        "parallelism_plan": plan_parallelism(
            cfg.get("cpu_workers", 1),
            threads_per_worker=int(cfg.get("threads_per_worker", 1)),
        ).to_dict(),
    }
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


def validate_fit_report(
    scenario_generator: PastFittedScenarioGenerator,
    require_spot: bool = False,
) -> None:
    """Check that causal duration fitting retained usable evidence."""
    report = scenario_generator.fit_report
    required = {
        "retained_hp_gang_rows": report.get("retained_hp_gang_rows"),
        "retained_blocks": report.get("retained_blocks"),
        "sample_count": report.get("sample_count", 0),
    }
    invalid = {key: value for key, value in required.items() if not value or value <= 0}
    if invalid:
        raise ValueError(f"Causal fit preflight failed: {invalid}.")
    if require_spot and int(report.get("spot_sample_count", 0) or 0) <= 0:
        raise ValueError("Causal fit contains no completed Spot jobs.")


def fit_scenarios_from_executed_history(
    cfg: dict[str, Any],
    spec: EpisodeSpec,
    fit_end: int,
    run_dir: str | None = None,
) -> PastFittedScenarioGenerator:
    """Fit duration evidence only from events executed before the split."""
    if fit_end <= spec.warmup_start:
        raise ValueError(f"Invalid causal fit cutoff: {fit_end}.")
    fit_spec = dataclasses.replace(
        spec,
        score_start=int(fit_end),
        score_end=int(fit_end + 300),
        followup_end=int(fit_end + 300 + spec.L_H),
        split_id=f"history_{fit_end}",
    )
    store = TraceStore.get_instance(cfg.get("trace_dir", "data/raw/spot_gpu_v2026"))
    fit_view = store.create_episode_view(fit_spec)
    snapshot_dir = cfg.get("snapshot_dir", "runs/warmup_snapshots")
    fit_log_dir = os.path.join(run_dir, "causal_fit_logs") if run_dir else None
    reuse_score_prefix = int(fit_end) == int(spec.score_start) and has_warmup_snapshot(
        spec, snapshot_dir
    )
    if reuse_score_prefix:
        snapshot_path = get_snapshot_path(spec, snapshot_dir)
    else:
        if not has_warmup_snapshot(fit_spec, snapshot_dir):
            generate_warmup_snapshot(
                spec=fit_spec,
                view=fit_view,
                snapshot_dir=snapshot_dir,
                log_dir=fit_log_dir,
            )
        snapshot_path = get_snapshot_path(fit_spec, snapshot_dir)
    prefix_simulator = load_warmup_snapshot(
        filepath=snapshot_path,
        policy_type="native_gfs",
        verifier=NoCheckVerifier(),
        log_dir=fit_log_dir,
        spec=fit_spec,
        view=fit_view,
        duration_estimator=None,
        run_id=f"causal_fit_{fit_end}",
        episode_id=f"causal_fit_{fit_end}",
        allow_history_prefix_reuse=reuse_score_prefix,
    )
    completion_events = [
        event.to_dict()
        for event in prefix_simulator.ledger.events
        if event.event_type == "completion"
    ]
    observed = build_observed_completion_history(
        completion_events,
        fit_view.job_df,
        fit_cutoff=fit_end,
    )
    running_jobs = [
        job
        for jobs in prefix_simulator.run_list.values()
        for job in jobs
    ]
    right_censored = build_right_censored_service_history(
        running_jobs=running_jobs,
        preemption_events=prefix_simulator.ledger.events,
        fit_cutoff=fit_end,
    )
    scenario_generator = PastFittedScenarioGenerator(
        horizon_seconds=cfg.get("horizon_seconds", 1800),
        L_H=spec.L_H,
        random_seed=int(cfg.get("scenario_seed", spec.policy_seed)),
        hp_arrival_cohort=str(cfg.get("hp_arrival_cohort", "gang_only")),
    )
    scenario_generator.fit(
        fit_view.job_df,
        fit_start=fit_spec.warmup_start,
        fit_end=fit_end,
        gpu_model_filter=spec.node_filter,
        observed_completions=observed,
        require_observed_completions=True,
        right_censored_service=right_censored,
    )
    report = scenario_generator.fit_report
    report["observed_completion_count"] = int(len(observed))
    report["right_censored_service_count"] = int(len(right_censored))
    report["right_censored_service_hash"] = (
        hashlib.sha256(
            right_censored.sort_values("job_id")
            .to_json(orient="records", double_precision=15)
            .encode("utf-8")
        ).hexdigest()
        if not right_censored.empty
        else hashlib.sha256(b"[]").hexdigest()
    )
    report["causal_prefix_snapshot"] = str(snapshot_path)
    return scenario_generator


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a final simulator config.")
    parser.add_argument("command", choices=("preflight",))
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_preflight(args.config)


if __name__ == "__main__":
    main()
