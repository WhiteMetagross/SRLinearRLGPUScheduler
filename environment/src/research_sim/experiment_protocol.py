"""Typed helpers for paper split validation and chronological train blocks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .types import EpisodeSpec, InvalidConfigError


@dataclass(frozen=True)
class TrainingBlock:
    block_id: str
    spec: EpisodeSpec


def build_training_blocks(
    cfg: Mapping[str, Any],
    base_spec: EpisodeSpec,
    manifest: Mapping[str, Any] | None = None,
) -> tuple[TrainingBlock, ...]:
    """Build continuous training passes or legacy chronological blocks.

    Frozen paper protocol 3.1 uses one full split pass. Rollout update boundaries
    do not change the EpisodeSpec and never reset simulator state.
    """
    train_execution = manifest.get("train_execution", {}) if manifest is not None else {}
    frozen_mode = str(train_execution.get("mode", ""))
    config_mode = str(cfg.get("training_mode", ""))
    if frozen_mode == "continuous" or config_mode == "continuous":
        if frozen_mode and config_mode != frozen_mode:
            raise InvalidConfigError(
                f"training_mode={config_mode!r} differs from frozen mode {frozen_mode!r}."
            )
        forbidden = [
            key for key in ("training_blocks", "training_block_seconds", "require_chronological_blocks")
            if key in cfg
        ]
        if forbidden:
            raise InvalidConfigError(
                "Continuous training cannot declare reset based training fields: "
                f"{forbidden}."
            )
        rollout_decisions = int(cfg.get("rollout_decisions", 0))
        expected_rollout = int(train_execution.get("rollout_decisions", rollout_decisions))
        if rollout_decisions <= 1:
            raise InvalidConfigError("Continuous training requires rollout_decisions > 1.")
        if expected_rollout and rollout_decisions != expected_rollout:
            raise InvalidConfigError(
                f"rollout_decisions={rollout_decisions} differs from frozen value {expected_rollout}."
            )
        return (TrainingBlock("train_continuous", replace(base_spec, split_id="train")),)

    raw_blocks = cfg.get("training_blocks")
    block_seconds = cfg.get("training_block_seconds")
    if block_seconds is None and manifest is not None:
        block_seconds = manifest.get("train_block_seconds")
    if raw_blocks is not None and block_seconds is not None:
        raise InvalidConfigError("Set training_blocks or training_block_seconds, not both.")

    entries: list[dict[str, Any]]
    if raw_blocks is not None:
        if not isinstance(raw_blocks, Sequence) or isinstance(raw_blocks, (str, bytes)) or not raw_blocks:
            raise InvalidConfigError("training_blocks must be a nonempty sequence of block mappings.")
        entries = []
        for index, block in enumerate(raw_blocks):
            if not isinstance(block, Mapping):
                raise InvalidConfigError(f"training_blocks[{index}] must be a mapping.")
            entries.append(dict(block))
    elif block_seconds is not None:
        width = int(block_seconds)
        if width <= 0:
            raise InvalidConfigError("training_block_seconds must be positive.")
        entries = []
        start = base_spec.score_start
        index = 0
        while start < base_spec.score_end:
            end = min(start + width, base_spec.score_end)
            entries.append({"id": f"train_{index:04d}", "score_start": start, "score_end": end})
            start = end
            index += 1
    else:
        if bool(cfg.get("require_chronological_blocks", False)):
            raise InvalidConfigError(
                "Paper training requires training_blocks or training_block_seconds. "
                "The episodes setting alone repeats one score window."
            )
        count = int(cfg.get("episodes", 1))
        if count <= 0:
            raise InvalidConfigError("episodes must be positive.")
        return tuple(
            TrainingBlock(f"episode_{i:04d}", replace(base_spec, policy_seed=base_spec.policy_seed + i))
            for i in range(count)
        )

    blocks: list[TrainingBlock] = []
    seen_ids: set[str] = set()
    previous_end: int | None = None
    for index, entry in enumerate(entries):
        block_id = str(entry.get("id", f"train_{index:04d}")).strip()
        if not block_id or block_id in seen_ids:
            raise InvalidConfigError(f"Training block ID must be nonempty and unique: {block_id!r}.")
        seen_ids.add(block_id)
        try:
            start = int(entry["score_start"])
            end = int(entry["score_end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidConfigError(
                f"Training block {block_id!r} requires integer score_start and score_end."
            ) from exc
        followup = int(entry.get("followup_end", end + base_spec.L_H))
        if not (base_spec.score_start <= start < end <= base_spec.score_end):
            raise InvalidConfigError(
                f"Training block {block_id!r} [{start}, {end}) falls outside "
                f"the configured train interval [{base_spec.score_start}, {base_spec.score_end})."
            )
        if followup < end + base_spec.L_H:
            raise InvalidConfigError(
                f"Training block {block_id!r} followup_end must be at least score_end + L_H."
            )
        if previous_end is not None and start < previous_end:
            raise InvalidConfigError("Training score blocks must be ordered and nonoverlapping.")
        block_spec = replace(
            base_spec,
            score_start=start,
            score_end=end,
            followup_end=followup,
            split_id="train",
            policy_seed=base_spec.policy_seed + index,
        )
        blocks.append(TrainingBlock(block_id, block_spec))
        previous_end = end

    if not blocks:
        raise InvalidConfigError("Training schedule produced no blocks.")
    if "episodes" in cfg and int(cfg["episodes"]) != len(blocks):
        raise InvalidConfigError(
            f"episodes={cfg['episodes']} conflicts with {len(blocks)} scheduled training blocks."
        )
    return tuple(blocks)


def validate_paper_split(
    cfg: Mapping[str, Any],
    spec: EpisodeSpec,
    manifest: Mapping[str, Any],
) -> None:
    """Reject any paper config that changes the frozen environment contract."""
    if manifest.get("status") != "FROZEN":
        raise InvalidConfigError("Paper runs require an experiment manifest with status FROZEN.")
    required = (
        "protocol_id", "protocol_version", "trace_hashes", "trace_source", "split_bounds",
        "embargoes", "shared_protocol", "workload_contract", "execution_contract", "study",
    )
    missing = [key for key in required if manifest.get(key) in (None, "", {})]
    if missing:
        raise InvalidConfigError(f"Frozen experiment manifest is missing required fields: {missing}.")
    study = manifest["study"]
    if not isinstance(study, Mapping):
        raise InvalidConfigError("Frozen study contract must be a mapping.")
    study_required = (
        "claim", "primary_metric", "service_metric", "service_violation_limit",
        "actor_seeds",
    )
    study_missing = [key for key in study_required if study.get(key) is None]
    if study_missing:
        raise InvalidConfigError(f"Frozen study contract is missing required fields: {study_missing}.")
    if len(study.get("actor_seeds", [])) < 3:
        raise InvalidConfigError("Frozen study contract requires at least three actor seeds.")

    required_config = (
        "node_filter", "trace_mode", "forecast_mode", "forecast_provider_id",
        "forecast_provider_version", "rule_version", "retry_mode", "L_H", "M",
        "candidate_geometry_version", "gpu_accounting_mode", "environment_seed", "spot_arrival_multiplier",
        "spot_arrival_seed", "B_call", "K", "invalid_action_mode",
        "control_interval_seconds", "control_queue_size", "max_quota_delta_fraction",
        "service_violation_limit", "snapshot_missing_mode", "independent_task_backend",
        "process_start_method", "threads_per_worker", "deterministic_result_order",
    )
    config_missing = [key for key in required_config if key not in cfg]
    if config_missing:
        raise InvalidConfigError(f"Paper config must set protocol fields explicitly: {config_missing}.")
    if float(cfg["service_violation_limit"]) != float(study["service_violation_limit"]):
        raise InvalidConfigError("Paper config service_violation_limit does not match the frozen study contract.")
    if str(cfg["invalid_action_mode"]) != "raise":
        raise InvalidConfigError("Paper configs require invalid_action_mode='raise'.")
    execution = manifest["execution_contract"]
    if not isinstance(execution, Mapping):
        raise InvalidConfigError("Frozen execution contract must be a mapping.")
    if not isinstance(cfg["deterministic_result_order"], bool):
        raise InvalidConfigError("deterministic_result_order must be a YAML boolean.")
    if int(cfg["threads_per_worker"]) < 1:
        raise InvalidConfigError("threads_per_worker must be positive.")
    execution_fields = {
        "snapshot_missing_mode": str(cfg["snapshot_missing_mode"]),
        "independent_task_backend": str(cfg["independent_task_backend"]),
        "process_start_method": str(cfg["process_start_method"]),
        "threads_per_worker": int(cfg["threads_per_worker"]),
        "deterministic_result_order": bool(cfg["deterministic_result_order"]),
    }
    for key, actual in execution_fields.items():
        if actual != execution.get(key):
            raise InvalidConfigError(
                f"Paper config {key}={actual!r} differs from the frozen execution contract."
            )
    config_spec_fields = {
        "node_filter": spec.node_filter,
        "trace_mode": spec.trace_mode,
        "forecast_mode": spec.forecast_mode,
        "forecast_provider_id": spec.forecast_provider_id,
        "forecast_provider_version": spec.forecast_provider_version,
        "rule_version": spec.rule_version,
        "retry_mode": spec.retry_mode,
        "L_H": spec.L_H,
        "M": spec.M,
        "candidate_geometry_version": spec.candidate_geometry_version,
        "gpu_accounting_mode": spec.gpu_accounting_mode,
        "environment_seed": spec.environment_seed,
        "spot_arrival_multiplier": spec.spot_arrival_multiplier,
        "spot_arrival_seed": spec.spot_arrival_seed,
        "control_interval_seconds": spec.control_interval_seconds,
        "control_queue_size": spec.control_queue_size,
        "max_quota_delta_fraction": spec.max_quota_delta_fraction,
    }
    for key, actual in config_spec_fields.items():
        if cfg[key] != actual:
            raise InvalidConfigError(
                f"Paper config {key}={cfg[key]!r} differs from EpisodeSpec value {actual!r}."
            )
    split_name = str(cfg.get("split", "")).lower()
    aliases = {"validation": "val"}
    split_name = aliases.get(split_name, split_name)
    split_bounds = manifest.get("split_bounds")
    if not isinstance(split_bounds, Mapping) or any(k not in split_bounds for k in ("train", "val", "test")):
        raise InvalidConfigError("Frozen manifest must define train, val, and test split bounds.")
    ranges = []
    for name in ("train", "val", "test"):
        bounds_item = split_bounds[name]
        start_item = int(bounds_item["score_start"])
        end_item = int(bounds_item["score_end"])
        if start_item >= end_item:
            raise InvalidConfigError(f"Frozen {name} bounds must satisfy score_start < score_end.")
        ranges.append((start_item, end_item, name))
    embargoes = manifest.get("embargoes", [])
    for left, right in zip(ranges, ranges[1:]):
        if right[0] < left[1]:
            raise InvalidConfigError(f"Frozen {left[2]} and {right[2]} score intervals overlap.")
        if right[0] > left[1] and not any(
            int(gap["start"]) <= left[1] and int(gap["end"]) >= right[0]
            for gap in embargoes
        ):
            raise InvalidConfigError(f"No declared embargo covers the gap between {left[2]} and {right[2]}.")
    if not isinstance(split_bounds, Mapping) or split_name not in split_bounds:
        raise InvalidConfigError(f"Manifest has no split bounds for {split_name!r}.")
    bounds = split_bounds[split_name]
    if spec.score_start != int(bounds["score_start"]) or spec.score_end != int(bounds["score_end"]):
        raise InvalidConfigError(
            f"Config score bounds [{spec.score_start}, {spec.score_end}) do not match "
            f"frozen {split_name} bounds [{bounds['score_start']}, {bounds['score_end']})."
        )
    if spec.followup_end < spec.score_end + spec.L_H:
        raise InvalidConfigError("Paper split followup does not cover every eligible deadline.")
    shared = manifest.get("shared_protocol")
    if not isinstance(shared, Mapping):
        raise InvalidConfigError("Manifest shared_protocol must be a mapping.")
    fields = {
        "node_filter": spec.node_filter,
        "trace_mode": spec.trace_mode,
        "rule_version": spec.rule_version,
        "retry_mode": spec.retry_mode,
        "L_H": spec.L_H,
        "M": spec.M,
        "candidate_geometry_version": spec.candidate_geometry_version,
        "gpu_accounting_mode": spec.gpu_accounting_mode,
        "environment_seed": spec.environment_seed,
        "control_interval_seconds": spec.control_interval_seconds,
        "control_queue_size": spec.control_queue_size,
        "max_quota_delta_fraction": spec.max_quota_delta_fraction,
    }
    for key, actual in fields.items():
        expected = shared.get(key)
        if expected is None or actual != expected:
            raise InvalidConfigError(
                f"Paper config {key}={actual!r} does not match frozen manifest value {expected!r}."
            )
    allowed_forecast_modes = tuple(shared.get("allowed_forecast_modes", ()))
    if spec.forecast_mode not in allowed_forecast_modes:
        raise InvalidConfigError(
            f"forecast_mode={spec.forecast_mode!r} is not allowed by the frozen environment."
        )
    if spec.forecast_mode == "native_online":
        native = shared.get("native_forecast", {})
        if (
            spec.forecast_provider_id != native.get("provider_id")
            or spec.forecast_provider_version != native.get("provider_version")
        ):
            raise InvalidConfigError("Native forecast identity differs from the frozen contract.")

    verification = manifest.get("verification", {})
    if not isinstance(verification, Mapping):
        raise InvalidConfigError("Frozen verification contract must be a mapping.")
    if int(cfg["B_call"]) != int(verification.get("B_call", -1)):
        raise InvalidConfigError("Paper config B_call differs from the frozen verification contract.")
    if int(cfg["K"]) != int(verification.get("K", -1)):
        raise InvalidConfigError("Paper config K differs from the frozen verification contract.")
    if int(spec.B_call) != int(cfg["B_call"]) or int(spec.K) != int(cfg["K"]):
        raise InvalidConfigError("EpisodeSpec verification settings differ from the paper config.")

    workload = manifest["workload_contract"]
    if not isinstance(workload, Mapping):
        raise InvalidConfigError("Frozen workload contract must be a mapping.")
    multiplier = int(spec.spot_arrival_multiplier)
    allowed_multipliers = tuple(int(v) for v in workload.get("allowed_spot_arrival_multipliers", ()))
    if multiplier not in allowed_multipliers:
        raise InvalidConfigError(
            f"spot_arrival_multiplier={multiplier} is not allowed by the frozen contract."
        )
    if multiplier == 1 and spec.spot_stress_start is not None:
        raise InvalidConfigError("Natural workload must not set spot_stress_start.")
    if multiplier > 1:
        lead_in = int(workload.get("stress_lead_in_seconds", -1))
        if spec.spot_stress_start != spec.score_start - lead_in:
            raise InvalidConfigError("Synthetic stress start differs from the frozen lead in.")
        allowed_seeds = {int(v) for v in workload.get("stress_seeds", ())}
        if spec.spot_arrival_seed not in allowed_seeds:
            raise InvalidConfigError("Synthetic workload seed is outside the frozen seed set.")
    if split_name == "train":
        train_execution = manifest.get("train_execution")
        if not isinstance(train_execution, Mapping):
            raise InvalidConfigError("Frozen manifest must define train_execution.")
        blocks = build_training_blocks(cfg, spec, manifest=manifest)
        train_bounds = split_bounds["train"]
        if blocks[0].spec.score_start != int(train_bounds["score_start"]):
            raise InvalidConfigError("Training blocks do not start at the frozen train boundary.")
        if blocks[-1].spec.score_end != int(train_bounds["score_end"]):
            raise InvalidConfigError("Training blocks do not end at the frozen train boundary.")
