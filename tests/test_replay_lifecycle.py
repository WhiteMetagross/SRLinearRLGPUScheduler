import json

import pytest

from research_sim import cli
from research_sim.types import EpisodeSpec, ResearchSimError


def test_snapshot_restore_failure_marks_replay_aborted(tmp_path, monkeypatch):
    run_dir = tmp_path / "failed_snapshot_replay"
    spec = EpisodeSpec(score_start=100, score_end=200, followup_end=3800)
    view = type("TraceViewStub", (), {"total_jobs": 0, "total_nodes": 0})()
    store = type("TraceStoreStub", (), {"create_episode_view": lambda self, _spec: view})()

    monkeypatch.setattr(cli, "load_spec_from_config", lambda _path: spec)
    monkeypatch.setattr(cli, "create_run_dir", lambda *_args: str(run_dir))
    monkeypatch.setattr(cli.TraceStore, "get_instance", lambda _path: store)
    monkeypatch.setattr(cli, "get_snapshot_path", lambda *_args: tmp_path / "snapshot.pkl")
    monkeypatch.setattr(cli, "has_warmup_snapshot", lambda *_args: True)

    def reject_snapshot(**_kwargs):
        raise ResearchSimError("snapshot event schema is incomplete")

    monkeypatch.setattr(cli, "load_warmup_snapshot", reject_snapshot)

    with pytest.raises(ResearchSimError, match="schema is incomplete"):
        cli.run_replay(
            config_path="unused.yaml",
            policy="deterministic_no_check",
            use_warmup_snapshot=True,
            snapshot_dir=str(tmp_path),
        )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    timing = json.loads((run_dir / "timing.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ABORTED"
    assert manifest["abort_error_type"] == "ResearchSimError"
    assert "schema is incomplete" in manifest["abort_error_message"]
    assert timing["status"] == "ABORTED"
