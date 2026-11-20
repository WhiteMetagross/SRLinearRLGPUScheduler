import json
from pathlib import Path

from benchmark.experiments import (
    DEFAULT_PLAN,
    _bar,
    _execute_job,
    _full_run_blockers,
    _load_plan,
    _prepare_payload,
    _render_status,
    _validation_baselines,
)


def test_scheduling_plan_is_complete_and_test_is_locked():
    plan, plan_path, suite = _load_plan(DEFAULT_PLAN)
    expected = (
        "gfs_ref",
        "fifo_admission",
        "shortest_causal_estimate_first",
        "fixed_headroom_quota",
        "yarn_cs_source_compatible",
        "chronus_source_compatible",
        "lyra_source_compatible",
        "fgd_source_compatible",
        "capacity_open_fcfs_best_fit_probe",
        "staggered_lease_best_fit_probe",
        "whole_card_cost_best_fit_probe",
        "whole_card_fgd_probe",
    )
    assert _validation_baselines(plan) == expected
    assert plan["final_test"]["status"] == "locked"
    blockers = _full_run_blockers(plan, suite, expected)
    assert len(blockers) == 1
    assert blockers[0]["baseline_id"] == "fixed_headroom_quota"


def test_paper_scheduler_plan_selects_six_deterministic_baselines():
    plan_path = DEFAULT_PLAN.parent / "paper_scheduler_plan.yaml"
    plan, resolved, suite = _load_plan(plan_path)
    expected = (
        "fixed_headroom_quota",
        "lyra_source_compatible",
        "yarn_cs_source_compatible",
        "fgd_source_compatible",
        "fifo_admission",
        "chronus_source_compatible",
    )
    assert resolved == plan_path.resolve()
    assert _validation_baselines(plan) == expected
    assert plan["final_test"]["status"] == "locked"
    assert plan["comparison"]["reference_baseline"] == "gfs_ref"
    blockers = _full_run_blockers(plan, suite, expected)
    assert len(blockers) == 1
    assert blockers[0]["baseline_id"] == "fixed_headroom_quota"


def test_progress_bar_is_bounded():
    assert _bar(-1.0, width=4) == "[....]"
    assert _bar(0.5, width=4) == "[##..]"
    assert _bar(2.0, width=4) == "[####]"


def test_job_launcher_applies_declared_threads(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    captured = {}

    def fake_run(command, **kwargs):
        captured["environment"] = kwargs["env"]
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr("benchmark.experiments.subprocess.run", fake_run)
    result = _execute_job(
        {
            "baseline_id": "gfs_ref",
            "split": "val",
            "run_seed": 17,
            "progress_file": str(job_dir / "progress.json"),
        },
        "suite.yaml",
        12,
        4,
    )
    assert result["state"] == "COMPLETED"
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert captured["environment"][name] == "4"


def test_status_renders_each_job(tmp_path: Path):
    jobs = []
    for baseline_id, fraction in (("gfs_ref", 1.0), ("fifo_admission", 0.5)):
        job_dir = tmp_path / "jobs" / baseline_id
        job_dir.mkdir(parents=True)
        progress_file = job_dir / "progress.json"
        progress_file.write_text(json.dumps({
            "state": "COMPLETED" if fraction == 1.0 else "RUNNING",
            "progress_fraction": fraction,
            "completed_steps": int(100 * fraction),
            "expected_steps": 100,
        }))
        jobs.append({
            "baseline_id": baseline_id,
            "split": "val",
            "run_seed": 17,
            "expected_steps": 100,
            "progress_file": str(progress_file),
        })
    (tmp_path / "batch.json").write_text(json.dumps({
        "batch_id": "test_batch",
        "jobs": jobs,
    }))
    (tmp_path / "status.json").write_text(json.dumps({"state": "RUNNING"}))
    rendered, terminal = _render_status(tmp_path)
    assert "gfs_ref" in rendered
    assert "fifo_admission" in rendered
    assert "75.00%" in rendered
    assert terminal is False
