import json
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark.fit_contextual_control import fit_contextual_control
from benchmark.fit_srgfs import build_candidate
from benchmark.fit_srlinear_quota_calibration import fit as fit_srlinear_calibration
from benchmark.policies import LinearResidualPolicy, SRGFSContextualMixPolicy, build_policy


def _context(time=0, hp=40.0, base=50.0, hp_queue=0.0):
    return SimpleNamespace(
        time=time,
        capacity_by_pool={"A100": 100.0},
        hp_used_by_pool={"A100": hp},
        spot_used_by_pool={"A100": 20.0},
        base_quota_by_pool={"A100": {1: base}},
        hp_queue_gpu_demand=hp_queue,
        admission_jobs=(),
        eviction_failure_rate_by_pool={"A100": 0.0},
    )


class _Provider:
    safe_quota_by_pool = {"A100": {1: 30.0}}
    srlinear_quota_by_pool = {"A100": {1: 50.0}}


def test_contextual_control_fit_is_train_only_and_finite(tmp_path):
    count = 48
    capacity = np.full(count, 100.0)
    base = np.full(count, 50.0)
    target = np.linspace(-0.05, 0.10, count)
    report = {
        "test_outcomes_accessed": False,
        "reports": [{
            "split": "train",
            "series": {
                "time": (np.arange(count) * 300).tolist(),
                "base_quota": base.tolist(),
                "capacity": capacity.tolist(),
                "hp_used": np.linspace(20.0, 40.0, count).tolist(),
                "spot_used": np.linspace(10.0, 30.0, count).tolist(),
                "hp_queue_gpu_demand": np.zeros(count).tolist(),
                "oracle_safe_quota": (base + target * capacity).tolist(),
            },
        }],
    }
    source = tmp_path / "train.json"
    output = tmp_path / "contextual.json"
    source.write_text(json.dumps(report))
    artifact = fit_contextual_control(
        source,
        output,
        risk_limit=0.10,
        minimum_residual=-0.15,
        maximum_residual=0.15,
        qualification="publication",
    )
    assert output.is_file()
    assert artifact["fit_split"] == "train"
    assert artifact["test_outcomes_accessed"] is False
    assert len(artifact["coefficients"]) == 9
    assert np.isfinite(artifact["native_safe_margin_fraction"])


def test_srlinear_calibration_uses_relative_provenance_paths(tmp_path):
    batch = tmp_path / "forecast_batch"
    batch.mkdir()
    (batch / "manifest.json").write_text(json.dumps({"origin_cadence_hours": 1}))
    for seed in (1, 2, 3):
        run = batch / "srlinear" / f"seed{seed}"
        run.mkdir(parents=True)
        (run / "status.json").write_text(json.dumps({
            "state": "COMPLETED",
            "total_origins": 2,
            "seed": seed,
        }))
        rows = [
            {
                "target_sum_organization_max": [10.0, 12.0],
                "pool_point": [9.0, 10.0],
            },
            {
                "target_sum_organization_max": [11.0, 13.0],
                "pool_point": [10.0, 12.0],
            },
        ]
        (run / "predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    artifact = fit_srlinear_calibration(
        batch, tmp_path / "calibration.json", quota_quantile=0.98
    )
    assert artifact["origin_cadence_hours"] == 1
    assert all(not path.startswith("/") for path in artifact["source_prediction_sha256"])


def test_contextual_mix_projects_to_the_safe_srlinear_segment():
    linear = LinearResidualPolicy(
        coefficients=(0.0,) * 9,
        intercept=-0.10,
        feature_mean=(0.0,) * 9,
        feature_scale=(1.0,) * 9,
        safety_offset=0.0,
    )
    policy = SRGFSContextualMixPolicy(
        linear=linear,
        hp_rise_trigger_fraction=0.01,
        eviction_failure_trigger=0.10,
        cooldown_seconds=600,
        trigger_lost_gpu_hours=0.0,
        priority_weights=(0.0,) * 6,
    )
    policy.bind_forecast_provider(_Provider())
    assert policy.action(_context())[0] == np.float32(-0.10)
    assert policy.decision_diagnostics()["executed_lambda"] == 0.5
    policy.observe_transition({
        "time": 300,
        "reward_components": {"lost_gpu_hours": 0.25},
    })
    assert policy.action(_context(time=300))[0] == np.float32(-0.20)
    assert policy.runtime_diagnostics()["rollback_trigger_count"] == 1


def test_candidate_builder_preserves_train_provenance(tmp_path):
    source_hash = "a" * 64
    contextual = {
        "schema_version": "1.0",
        "policy_type": "contextual_linear_residual",
        "qualification": "publication",
        "fit_split": "train",
        "test_outcomes_accessed": False,
        "source_report": "results/training/train.json",
        "source_report_sha256": source_hash,
        "coefficients": [0.0] * 9,
        "intercept": 0.0,
        "feature_mean": [0.0] * 9,
        "feature_scale": [1.0] * 9,
        "safety_offset": 0.0,
        "minimum_residual": -0.15,
        "maximum_residual": 0.15,
        "feature_names": [f"f{index}" for index in range(9)],
        "train_risk_tick_rate": 0.02,
        "native_safe_margin_fraction": 0.07,
    }
    contextual_path = tmp_path / "contextual.json"
    contextual_path.write_text(json.dumps(contextual))
    output = build_candidate(
        contextual_path=contextual_path,
        output_dir=tmp_path / "fitted",
        cooldown_seconds=600,
        hp_rise_trigger_fraction=0.01,
        eviction_failure_trigger=0.10,
    )
    artifact = json.loads(output.read_text())
    assert artifact["fit_split"] == "train"
    assert artifact["test_outcomes_accessed"] is False


def test_missing_generated_artifact_fails_closed():
    from pathlib import Path

    base = Path(__file__).parents[1] / "benchmark"
    declaration = {
        "policy_type": "contextual_linear_residual",
        "action_artifact": "fitted/non_rl_contextual_linear.json",
        "priority_weights": [0.0] * 6,
    }
    with pytest.raises(ValueError, match="Required fitted artifact is missing"):
        build_policy(declaration, suite_base=base, allow_smoke_artifact=False)
