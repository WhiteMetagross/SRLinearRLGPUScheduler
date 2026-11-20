from pathlib import Path

import pytest
import yaml

from research_sim.experiment_protocol import build_training_blocks, validate_paper_split
from research_sim.types import EpisodeSpec, InvalidConfigError
from benchmark.protocol_helpers import build_spec_from_dict, load_yaml


def _frozen_fixture(spec, split="val", service_limit=0.1):
    cfg = {
        "split": split,
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
        "B_call": spec.B_call,
        "K": spec.K,
        "control_interval_seconds": spec.control_interval_seconds,
        "control_queue_size": spec.control_queue_size,
        "max_quota_delta_fraction": spec.max_quota_delta_fraction,
        "invalid_action_mode": "raise",
        "service_violation_limit": service_limit,
        "snapshot_missing_mode": "raise",
        "independent_task_backend": "process",
        "process_start_method": "spawn",
        "threads_per_worker": 1,
        "deterministic_result_order": True,
    }
    manifest = {
        "protocol_id": "fixture_v3_1",
        "protocol_version": "3.1",
        "status": "FROZEN",
        "trace_hashes": {"fixture.csv": "abc"},
        "trace_source": {"source_commit": "fixture"},
        "study": {
            "claim": "fixture",
            "primary_metric": "fixture",
            "service_metric": "fixture",
            "service_violation_limit": service_limit,
            "actor_seeds": [1, 2, 3],
        },
        "split_bounds": {
            "train": {"score_start": 100, "score_end": 200},
            "val": {"score_start": 400, "score_end": 500},
            "test": {"score_start": 700, "score_end": 800},
        },
        "embargoes": [{"start": 200, "end": 400}, {"start": 500, "end": 700}],
        "shared_protocol": {
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
            "allowed_forecast_modes": [spec.forecast_mode],
            "native_forecast": {
                "provider_id": spec.forecast_provider_id,
                "provider_version": spec.forecast_provider_version,
            },
        },
        "verification": {"B_call": spec.B_call, "K": spec.K},
        "execution_contract": {
            "snapshot_missing_mode": "raise",
            "independent_task_backend": "process",
            "process_start_method": "spawn",
            "threads_per_worker": 1,
            "deterministic_result_order": True,
        },
        "workload_contract": {
            "allowed_spot_arrival_multipliers": [1, 2, 4],
            "stress_lead_in_seconds": 100,
            "stress_seeds": [4201, 4202, 4203],
        },
        "train_execution": {
            "mode": "continuous",
            "rollout_decisions": 16,
        },
    }
    if split == "train":
        cfg["training_mode"] = "continuous"
        cfg["rollout_decisions"] = 16
    return cfg, manifest


def test_cli_rejects_native_forecast_without_required_history():
    with pytest.raises(ValueError, match="requires at least"):
        build_spec_from_dict({
            "score_start": 300,
            "score_end": 900,
            "followup_end": 4500,
            "forecast_mode": "native_online",
        })

    spec = build_spec_from_dict({
        "score_start": 300,
        "score_end": 900,
        "followup_end": 4500,
        "forecast_mode": "causal_capacity",
    })
    assert spec.score_start == 300


def test_fixed_width_training_schedule_is_chronological_and_has_followup():
    base = EpisodeSpec(
        score_start=100,
        score_end=7300,
        followup_end=10900,
        L_H=3600,
    )
    blocks = build_training_blocks({"training_block_seconds": 3600}, base)

    assert len(blocks) == 2
    assert [(b.spec.score_start, b.spec.score_end) for b in blocks] == [(100, 3700), (3700, 7300)]
    assert [b.spec.followup_end for b in blocks] == [7300, 10900]
    assert blocks[0].block_id != blocks[1].block_id
    assert [b.spec.policy_seed for b in blocks] == [42, 43]


def test_paper_training_rejects_episode_count_without_calendar_schedule():
    base = EpisodeSpec(score_start=100, score_end=200, followup_end=3800)

    with pytest.raises(InvalidConfigError, match="training_blocks or training_block_seconds"):
        build_training_blocks({"episodes": 3, "require_chronological_blocks": True}, base)


def test_legacy_smoke_episodes_keep_window_but_get_distinct_seeds():
    base = EpisodeSpec(score_start=100, score_end=200, followup_end=3800)
    blocks = build_training_blocks({"episodes": 2}, base)

    assert [(b.spec.score_start, b.spec.score_end) for b in blocks] == [(100, 200), (100, 200)]
    assert [b.spec.policy_seed for b in blocks] == [42, 43]


@pytest.mark.parametrize(
    "entries, message",
    [
        ([{"id": "a", "score_start": 100, "score_end": 200},
          {"id": "b", "score_start": 150, "score_end": 250}], "ordered and nonoverlapping"),
        ([{"id": "a", "score_start": 100, "score_end": 200, "followup_end": 3799}], "followup_end"),
        ([{"id": "a", "score_start": 0, "score_end": 200}], "falls outside"),
        ([{"id": "a", "score_start": 100, "score_end": 200},
          {"id": "a", "score_start": 200, "score_end": 300}], "unique"),
    ],
)
def test_invalid_training_schedules_fail_closed(entries, message):
    base = EpisodeSpec(score_start=100, score_end=300, followup_end=3900)

    with pytest.raises(InvalidConfigError, match=message):
        build_training_blocks({"training_blocks": entries}, base)


def test_frozen_paper_split_requires_exact_bounds_and_matching_shared_fields():
    spec = EpisodeSpec(
        split_id="val",
        score_start=400,
        score_end=500,
        followup_end=4100,
        K=0,
        M=8,
        B_call=0,
    )
    cfg, manifest = _frozen_fixture(spec)

    validate_paper_split(cfg, spec, manifest)

    wrong_geometry = dict(cfg, candidate_geometry_version="publication_v1")
    with pytest.raises(InvalidConfigError, match="candidate_geometry_version"):
        validate_paper_split(wrong_geometry, spec, manifest)

    unsafe_snapshot = dict(cfg, snapshot_missing_mode="generate")
    with pytest.raises(InvalidConfigError, match="snapshot_missing_mode"):
        validate_paper_split(unsafe_snapshot, spec, manifest)

    wrong_bounds = dict(manifest)
    wrong_bounds["split_bounds"] = dict(manifest["split_bounds"])
    wrong_bounds["split_bounds"]["val"] = {"score_start": 401, "score_end": 500}
    wrong_bounds["embargoes"] = [{"start": 200, "end": 401}, {"start": 500, "end": 700}]
    with pytest.raises(InvalidConfigError, match="do not match frozen"):
        validate_paper_split(cfg, spec, wrong_bounds)


def test_frozen_manifest_requires_one_continuous_train_pass():
    spec = EpisodeSpec(
        split_id="train",
        score_start=100,
        score_end=200,
        followup_end=3800,
        B_call=0,
        K=0,
    )
    cfg, manifest = _frozen_fixture(spec, split="train")
    manifest["split_bounds"] = {
        "train": {"score_start": 100, "score_end": 200},
        "val": {"score_start": 300, "score_end": 400},
        "test": {"score_start": 500, "score_end": 600},
    }
    manifest["embargoes"] = [
        {"start": 200, "end": 300},
        {"start": 400, "end": 500},
    ]

    validate_paper_split(cfg, spec, manifest)
    blocks = build_training_blocks(cfg, spec, manifest=manifest)
    assert [(b.block_id, b.spec.score_start, b.spec.score_end) for b in blocks] == [
        ("train_continuous", 100, 200)
    ]

    unsafe = dict(cfg, training_block_seconds=50)
    with pytest.raises(InvalidConfigError, match="reset based training fields"):
        validate_paper_split(unsafe, spec, manifest)


def test_draft_paper_manifest_is_not_runnable():
    spec = EpisodeSpec(split_id="val", score_start=400, score_end=500, followup_end=4100)
    with pytest.raises(InvalidConfigError, match="status FROZEN"):
        validate_paper_split({"split": "val"}, spec, {"status": "DRAFT"})


@pytest.mark.parametrize(
    "name",
    [
        "natural_train.yaml",
        "natural_val.yaml",
        "natural_test.yaml",
    ],
)
def test_frozen_environment_configs_match_protocol(name):
    root = Path(__file__).resolve().parents[1]
    config_path = root / "environment" / "configs" / name
    cfg = load_yaml(str(config_path))
    manifest = yaml.safe_load(
        (root / "environment" / "protocol.yaml").read_text(encoding="utf-8")
    )
    spec = build_spec_from_dict(cfg)

    validate_paper_split(cfg, spec, manifest)


def test_method_config_cannot_override_frozen_environment_field(tmp_path):
    root = Path(__file__).resolve().parents[1]
    child = tmp_path / "bad_method.yaml"
    child.write_text(
        "environment_config: "
        f"'{(root / 'environment' / 'configs' / 'natural_val.yaml').as_posix()}'\n"
        "score_start: 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forbidden fields"):
        load_yaml(str(child))
