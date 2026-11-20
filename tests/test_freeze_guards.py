import hashlib
import json

import pytest

from research_sim.freeze import validate_test_access_grant, verify_frozen_bundle
from research_sim.types import InvalidConfigError


def test_frozen_bundle_rejects_drift_and_unlisted_files(tmp_path):
    payload = tmp_path / "payload.txt"
    payload.write_text("stable\n", encoding="utf-8")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    checksums = tmp_path / "FROZEN_SHA256SUMS"
    checksums.write_text(f"{digest}  payload.txt\n", encoding="utf-8")

    report = verify_frozen_bundle(checksums)
    assert report["file_count"] == 1

    payload.write_text("changed\n", encoding="utf-8")
    with pytest.raises(InvalidConfigError, match="mismatched"):
        verify_frozen_bundle(checksums)

    payload.write_text("stable\n", encoding="utf-8")
    (tmp_path / "extra.txt").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(InvalidConfigError, match="unexpected"):
        verify_frozen_bundle(checksums)


def test_final_test_grant_requires_locked_method_hash(tmp_path):
    path = tmp_path / "grant.json"
    with pytest.raises(InvalidConfigError, match="Final test is locked"):
        validate_test_access_grant(path, protocol_id="p1")

    path.write_text(json.dumps({
        "status": "UNLOCKED",
        "protocol_id": "p1",
        "acknowledgement": "METHOD_SELECTION_COMPLETE",
        "method_lock_sha256": "a" * 64,
        "opened_at": "2026-10-03T00:00:00+05:30",
    }), encoding="utf-8")
    grant = validate_test_access_grant(path, protocol_id="p1")
    assert grant["status"] == "UNLOCKED"
