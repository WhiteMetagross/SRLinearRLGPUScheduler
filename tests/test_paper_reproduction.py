import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from benchmark.fit_paper_controls import fit_paper_controls
from benchmark.policies import build_policy


ROOT = Path(__file__).resolve().parents[1]


def _train_report(path: Path) -> None:
    steps = 96
    time = np.arange(steps, dtype=np.float64) * 3600.0
    capacity = np.full(steps, 100.0)
    hp_used = 35.0 + 8.0 * np.sin(2.0 * np.pi * time / 86400.0)
    base = np.full(steps, 55.0)
    safe = capacity - hp_used - 8.0
    payload = {
        "schema_version": "1.0",
        "test_outcomes_accessed": False,
        "reports": [{
            "split": "train",
            "evaluated_hours": 96.0,
            "horizon_seconds": 3600,
            "series": {
                "time": time.tolist(),
                "capacity": capacity.tolist(),
                "hp_used": hp_used.tolist(),
                "base_quota": base.tolist(),
                "spot_used": np.full(steps, 40.0).tolist(),
                "hp_queue_gpu_demand": np.zeros(steps).tolist(),
                "oracle_safe_quota": safe.tolist(),
            },
        }],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _context() -> SimpleNamespace:
    return SimpleNamespace(
        time=0,
        capacity_by_pool={"A100": 100.0},
        hp_used_by_pool={"A100": 40.0},
        spot_used_by_pool={"A100": 30.0},
        base_quota_by_pool={"A100": {1: 50.0}},
        hp_queue_gpu_demand=0.0,
        admission_jobs=(),
    )


def test_all_reported_control_artifacts_fit_from_train_only(tmp_path: Path):
    source = tmp_path / "train.json"
    output = tmp_path / "fitted"
    _train_report(source)
    paths = fit_paper_controls(
        source,
        output,
        risk_limit=0.02,
        minimum_residual=-0.15,
        maximum_residual=0.15,
        qualification="smoke_only",
    )
    assert set(paths) == {
        "fixed_residual",
        "diurnal",
        "online_quantile",
        "contextual_linear",
        "headroom",
        "guarded_base",
    }
    for path in paths.values():
        artifact = json.loads(path.read_text(encoding="utf-8"))
        assert artifact["fit_split"] == "train"
        assert artifact["test_outcomes_accessed"] is False


def test_each_fitted_policy_builds_and_returns_valid_action(tmp_path: Path):
    source = tmp_path / "train.json"
    _train_report(source)
    paths = fit_paper_controls(
        source,
        tmp_path,
        risk_limit=0.02,
        minimum_residual=-0.15,
        maximum_residual=0.15,
        qualification="smoke_only",
    )
    declarations = {
        "fixed_residual": "fixed_residual",
        "diurnal": "hour_of_day_residual",
        "online_quantile": "expanding_quantile_residual",
        "contextual_linear": "contextual_linear_residual",
        "headroom": "current_free_headroom",
        "guarded_base": "guarded_base_quota",
    }
    for name, policy_type in declarations.items():
        policy, _ = build_policy(
            {
                "policy_type": policy_type,
                "action_artifact": str(paths[name]),
                "priority_weights": [0.0] * 6,
            },
            suite_base=tmp_path,
            allow_smoke_artifact=True,
        )
        action = policy.action(_context())
        assert action.shape == (7,)
        assert np.all(np.isfinite(action))


def test_paper_manifest_references_declared_methods():
    manifest = yaml.safe_load(
        (ROOT / "benchmark" / "paper_results_manifest.yaml").read_text(encoding="utf-8")
    )
    for section in manifest["tables"].values():
        if "suite" in section:
            suite = yaml.safe_load(
                (ROOT / "benchmark" / section["suite"]).read_text(encoding="utf-8")
            )
            declared = suite.get("baselines", suite.get("models", {}))
            for method_id in section["methods"].values():
                assert method_id in declared
        else:
            for suite_name, methods in section["suites"].items():
                suite = yaml.safe_load(
                    (ROOT / "benchmark" / suite_name).read_text(encoding="utf-8")
                )
                declared = suite.get("baselines", suite.get("methods", {}))
                for method_id in methods.values():
                    assert method_id in declared
