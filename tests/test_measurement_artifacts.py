import os
import json
import hashlib
import subprocess
import pytest
from pathlib import Path

from research_sim.types import EpisodeSpec, SimulationEvent
from research_sim.metrics import EventLedger, EpisodeMetrics
from research_sim.measurement import SCHEMA_VERSION, reduce_replay_events
from research_sim.cli import write_run_artifacts, run_audit_run


def _close_ledger(ledger, time, reason="all_jobs_completed", status="completed"):
    ledger.record_event(time=time, event_type="terminal", details={
        "time": time, "terminal_reason": reason, "remaining_queued": 0,
        "remaining_running": 0, "last_event_seq": ledger._seq, "status": status,
    })


def test_write_run_artifacts_and_audit_clean_pass(tmp_path):
    run_dir = tmp_path / "test_run_artifacts"
    spec = EpisodeSpec(
        score_start=100,
        score_end=300,
        followup_end=3700,
        L_H=100,
        B_call=200,
        K=2
    )
    ledger = EventLedger("ep1", spec, run_id="test_run_artifacts")
    ledger.record_job_start(
        time=100, job_id=1, job_type="Spot", requested_worker_count=1,
        actual_worker_count=1, complete_start=True,
        allocations=[{"node_name": "n1", "card_ids": [0]}], attempt_index=0,
    )

    # Add valid sequence of events
    ledger.record_spot_credit(
        time=150,
        job_id=1,
        gpu_seconds=3600.0,
        reason="checkpoint",
        credited_run_seconds=3600.0,
        gpu_equivalents=1.0
    )
    ledger.record_event(200, "action", job_id=1, details={
        "decision_id": 1, "time": 200, "state_version": 0,
        "candidate_keys": ["candidate"], "action_mask": [True],
        "proposed_slot": 1, "verify_requested": True, "executed_slot": 1,
        "executed_deferral": False, "quota_window": 0, "calls_used": 1,
        "budget_after": 196,
    })
    ledger.record_solver_invocation(
        time=200,
        call_id=0,
        decision_id=1,
        quota_window=0,
        scenario_id=10,
        scenario_seed=42,
        side="proposed",
        T_max=2.0,
        worker_count=4
    )
    ledger.finalize_hp_deadlines(3700)
    _close_ledger(ledger, 3700)

    # Compute online metrics
    online_metrics = ledger.compute_summary_metrics(
        current_time=3700,
        remaining_queued=0,
        remaining_running=0,
        terminated=True,
        truncated=False,
        terminal_reason="all_jobs_completed"
    )

    solver_rows = [{
        "schema_version": SCHEMA_VERSION,
        "run_id": "test_run_artifacts",
        "episode_id": "ep1",
        "call_id": 0,
        "invocation_id": 0,
        "decision_id": 1,
        "quota_window": 0,
        "scenario_id": 10,
        "scenario_seed": 42,
        "side": "proposed",
        "status": "OPTIMAL",
        "objective_gpu_seconds": 3600,
        "objective_gpu_hours": 1.0,
        "T_max": 2.0,
        "worker_count": 4,
        "model_build_time_seconds": 0.01,
        "solver_search_time_seconds": 0.05,
        "wall_time_seconds": 0.06,
        "exception": None
    }]

    manifest = {"command": "test", "checks_enabled": True}
    timing = {"elapsed_wall_seconds": 0.5}

    write_run_artifacts(
        run_dir=str(run_dir),
        spec=spec,
        manifest=manifest,
        timing=timing,
        metrics=online_metrics.to_dict(),
        events=ledger.events,
        solver_invocations=solver_rows
    )
    (run_dir / "solver_invocations.started.jsonl").write_text(
        json.dumps({**solver_rows[0], "status": "STARTED"}) + "\n",
        encoding="utf-8",
    )

    # Assert expected files exist
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "timing.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "events.jsonl").exists()
    assert (run_dir / "solver_invocations.jsonl").exists()

    # Verify manifest has SHA256 of events.jsonl
    man = json.loads((run_dir / "manifest.json").read_text())
    ev_bytes = (run_dir / "events.jsonl").read_bytes()
    assert man["events_sha256"] == hashlib.sha256(ev_bytes).hexdigest()
    assert man["events_count"] == len(ledger.events)
    assert man["schema_version"] == SCHEMA_VERSION

    # Audit run must succeed without raising SystemExit(1)
    run_audit_run(str(run_dir))


def test_audit_streams_events_without_reading_the_whole_file(tmp_path, monkeypatch):
    run_dir = tmp_path / "streamed_audit"
    spec = EpisodeSpec(score_start=100, score_end=300, followup_end=3700, L_H=100)
    ledger = EventLedger("ep1", spec, run_id="streamed_audit")
    ledger.record_job_start(
        time=100,
        job_id=1,
        job_type="Spot",
        requested_worker_count=1,
        actual_worker_count=1,
        complete_start=True,
        allocations=[{"node_name": "n1", "card_ids": [0]}],
        attempt_index=0,
    )
    ledger.record_spot_credit(
        150,
        1,
        3600.0,
        "completion",
        credited_run_seconds=3600.0,
    )
    _close_ledger(ledger, 3700, "done")
    metrics = ledger.compute_summary_metrics(3700, 0, 0, True, False, "done").to_dict()
    write_run_artifacts(
        str(run_dir),
        spec,
        {"command": "test"},
        {},
        metrics=metrics,
        events=ledger.events,
    )

    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path):
        if path.name == "events.jsonl":
            raise AssertionError("audit must stream events.jsonl")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    result = run_audit_run(str(run_dir))
    assert result["status"] == "PASSED"


def test_audit_fails_on_hash_tampering(tmp_path):
    run_dir = tmp_path / "tampered_run"
    spec = EpisodeSpec(score_start=100, score_end=300, followup_end=3700, L_H=100)
    ledger = EventLedger("ep1", spec, run_id="tampered_run")
    ledger.record_spot_credit(150, 1, 3600.0, "completion", credited_run_seconds=3600.0)
    _close_ledger(ledger, 3700, "done")
    online_metrics = ledger.compute_summary_metrics(3700, 0, 0, True, False, "done")

    write_run_artifacts(
        run_dir=str(run_dir),
        spec=spec,
        manifest={"command": "test"},
        timing={"elapsed_wall_seconds": 0.5},
        metrics=online_metrics.to_dict(),
        events=ledger.events
    )

    # Tamper with events.jsonl
    ev_file = run_dir / "events.jsonl"
    ev_file.write_text(ev_file.read_text() + "\n")

    with pytest.raises(SystemExit) as exc:
        run_audit_run(str(run_dir))
    assert exc.value.code == 1


def test_audit_fails_on_online_metric_discrepancy(tmp_path):
    run_dir = tmp_path / "divergent_run"
    spec = EpisodeSpec(score_start=100, score_end=300, followup_end=3700, L_H=100)
    ledger = EventLedger("ep1", spec, run_id="divergent_run")
    ledger.record_spot_credit(150, 1, 3600.0, "completion", credited_run_seconds=3600.0)
    _close_ledger(ledger, 3700, "done")
    online_metrics = ledger.compute_summary_metrics(3700, 0, 0, True, False, "done")

    metrics_dict = online_metrics.to_dict()
    # Artificially modify metric
    metrics_dict["useful_spot_gpu_seconds_score"] = 9999.0

    write_run_artifacts(
        run_dir=str(run_dir),
        spec=spec,
        manifest={"command": "test"},
        timing={"elapsed_wall_seconds": 0.5},
        metrics=metrics_dict,
        events=ledger.events
    )

    with pytest.raises(SystemExit) as exc:
        run_audit_run(str(run_dir))
    assert exc.value.code == 1


def test_audit_rejects_missing_terminal_after_valid_hash_rewrite(tmp_path):
    run_dir = tmp_path / "missing_terminal"
    spec = EpisodeSpec(score_start=100, score_end=300, followup_end=3700, L_H=100)
    ledger = EventLedger("ep1", spec, run_id="missing_terminal")
    ledger.record_job_start(100, 1, "Spot", requested_worker_count=1,
                            actual_worker_count=1, complete_start=True,
                            allocations=[{"node_name": "n1", "card_ids": [0]}], attempt_index=0)
    ledger.record_spot_credit(150, 1, 3600.0, "completion", credited_run_seconds=3600.0)
    _close_ledger(ledger, 3700, "done")
    metrics = ledger.compute_summary_metrics(3700, 0, 0, True, False, "done").to_dict()
    write_run_artifacts(str(run_dir), spec, {"command": "test"}, {}, metrics=metrics, events=ledger.events)

    event_path = run_dir / "events.jsonl"
    lines = event_path.read_text(encoding="utf-8").splitlines()
    event_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    event_bytes = event_path.read_bytes()
    manifest["events_sha256"] = hashlib.sha256(event_bytes).hexdigest()
    manifest["events_count"] = len(lines) - 1
    manifest["events_bytes"] = len(event_bytes)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        run_audit_run(str(run_dir))
    assert exc.value.code == 1
    diagnostic = json.loads((run_dir / "first_mismatch.json").read_text(encoding="utf-8"))
    assert "terminal" in diagnostic["error"]


def test_check_enabled_audit_requires_invocation_files(tmp_path):
    run_dir = tmp_path / "missing_solver_records"
    spec = EpisodeSpec(score_start=100, score_end=300, followup_end=3700, L_H=100, K=1)
    ledger = EventLedger("ep1", spec, run_id="missing_solver_records")
    ledger.record_event(150, "action", job_id=1, details={
        "decision_id": 0, "time": 150, "state_version": 0,
        "candidate_keys": ["c"], "action_mask": [True],
        "proposed_slot": 1, "verify_requested": True, "executed_slot": 1,
        "executed_deferral": False, "quota_window": 0, "calls_used": 1,
        "budget_after": 198,
    })
    ledger.record_solver_invocation(150, 0, 0, 0, 1, 2, "proposed", 1.0, 1)
    _close_ledger(ledger, 3700, "done")
    metrics = ledger.compute_summary_metrics(3700, 0, 0, True, False, "done").to_dict()
    write_run_artifacts(
        str(run_dir), spec, {"command": "evaluate", "checks_enabled": True}, {},
        metrics=metrics, events=ledger.events,
    )

    with pytest.raises(SystemExit) as exc:
        run_audit_run(str(run_dir))
    assert exc.value.code == 1
    diagnostic = json.loads((run_dir / "first_mismatch.json").read_text(encoding="utf-8"))
    assert "solver invocation files" in diagnostic["error"]
