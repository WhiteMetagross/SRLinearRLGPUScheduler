import json
from pathlib import Path

import pytest

from benchmark.aggregate_paper_results import aggregate


def _write(path: Path, value: float, complete: bool = True) -> None:
    path.write_text(json.dumps({
        "publication_metrics": {
            "observation_complete": complete,
            "useful_spot_gpu_hours": value,
            "lost_spot_gpu_hours": 2.0,
            "spot_queued_gpu_hours": 3.0,
            "hp_queued_gpu_hours": 4.0,
            "hp_gang_violation_rate": 0.01,
            "spot_evictions": 5,
        }
    }), encoding="utf-8")


def test_aggregation_reports_sample_statistics(tmp_path: Path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write(first, 10.0)
    _write(second, 14.0)
    row = aggregate([f"method={first}", f"method={second}"])[0]
    assert row["runs"] == 2
    assert row["useful_spot_gpu_hours_mean"] == 12.0
    assert row["useful_spot_gpu_hours_sd"] == pytest.approx(2.8284271247461903)


def test_aggregation_rejects_incomplete_runs(tmp_path: Path):
    path = tmp_path / "incomplete.json"
    _write(path, 10.0, complete=False)
    with pytest.raises(ValueError, match="Incomplete"):
        aggregate([f"method={path}"])
