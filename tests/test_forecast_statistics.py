import json
from pathlib import Path

import numpy as np
import pytest

from benchmark.forecast_statistics import (
    circular_block_bootstrap,
    holm_adjust,
    load_prediction_losses,
    validate_matched_series,
)


def _write_predictions(
    path: Path,
    *,
    model_id: str,
    seed: int,
    predictions: list[list[float]],
    targets: list[list[float]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for day, (prediction, target) in enumerate(zip(predictions, targets), start=1):
        rows.append(
            json.dumps(
                {
                    "model_id": model_id,
                    "seed": seed,
                    "origin": f"2024-05-{day:02d}T20:00:00",
                    "point_scored": True,
                    "pool_point": prediction,
                    "target_sum_organization_max": target,
                }
            )
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_prediction_losses_and_seed_aggregation_are_paired(tmp_path: Path):
    targets = [[10.0, 20.0], [12.0, 22.0], [14.0, 24.0]]
    candidate = load_prediction_losses(
        _write_predictions(
            tmp_path / "candidate.jsonl",
            model_id="candidate",
            seed=17,
            predictions=[[9.0, 19.0], [11.0, 21.0], [13.0, 23.0]],
            targets=targets,
        )
    )
    baselines = [
        load_prediction_losses(
            _write_predictions(
                tmp_path / f"baseline_{seed}.jsonl",
                model_id="baseline",
                seed=seed,
                predictions=predictions,
                targets=targets,
            )
        )
        for seed, predictions in (
            (1, [[8.0, 18.0], [10.0, 20.0], [12.0, 22.0]]),
            (2, [[6.0, 16.0], [8.0, 18.0], [10.0, 20.0]]),
        )
    ]
    candidate_losses, baseline_losses = validate_matched_series(
        candidate, baselines
    )
    np.testing.assert_array_equal(candidate_losses, np.asarray([1.0, 1.0, 1.0]))
    np.testing.assert_array_equal(baseline_losses, np.asarray([3.0, 3.0, 3.0]))


def test_prediction_pairing_rejects_target_mismatch(tmp_path: Path):
    candidate = load_prediction_losses(
        _write_predictions(
            tmp_path / "candidate.jsonl",
            model_id="candidate",
            seed=17,
            predictions=[[1.0], [2.0]],
            targets=[[1.0], [2.0]],
        )
    )
    baseline = load_prediction_losses(
        _write_predictions(
            tmp_path / "baseline.jsonl",
            model_id="baseline",
            seed=1,
            predictions=[[1.0], [2.0]],
            targets=[[1.0], [3.0]],
        )
    )
    with pytest.raises(ValueError, match="Target mismatch"):
        validate_matched_series(candidate, [baseline])


def test_circular_block_bootstrap_is_reproducible_and_detects_uniform_gain():
    differences = np.full(61, -2.0)
    first = circular_block_bootstrap(
        differences,
        block_length=7,
        replicates=1_000,
        seed=17,
        simultaneous_comparisons=2,
    )
    second = circular_block_bootstrap(
        differences,
        block_length=7,
        replicates=1_000,
        seed=17,
        simultaneous_comparisons=2,
    )
    assert first == second
    assert first["percentile_ci_gpu"] == [-2.0, -2.0]
    assert first["familywise_percentile_ci_gpu"] == [-2.0, -2.0]
    assert first["bootstrap_probability_candidate_better"] == 1.0


def test_holm_adjustment_preserves_monotonicity():
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
