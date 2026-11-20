"""Integrity and final test access guards for frozen research environments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .types import InvalidConfigError


CHECKSUM_FILENAME = "FROZEN_SHA256SUMS"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tracked_files(root: Path) -> tuple[str, ...]:
    ignored_names = {CHECKSUM_FILENAME}
    paths = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name in ignored_names:
            continue
        if (
            "__pycache__" in path.parts
            or any(part.endswith(".egg-info") for part in path.parts)
            or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        paths.append(path.relative_to(root).as_posix())
    return tuple(sorted(paths))


def load_checksums(checksum_path: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    for line_number, raw in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise InvalidConfigError(
                f"Malformed frozen checksum line {line_number}: {raw!r}."
            )
        digest, relative = parts
        relative = relative.lstrip("* ")
        if relative in expected:
            raise InvalidConfigError(f"Duplicate frozen checksum path: {relative}.")
        expected[relative] = digest.lower()
    if not expected:
        raise InvalidConfigError("Frozen checksum manifest is empty.")
    return expected


def verify_frozen_bundle(checksum_path: str | Path) -> dict[str, Any]:
    checksum_file = Path(checksum_path).resolve()
    if not checksum_file.is_file():
        raise InvalidConfigError(f"Frozen checksum manifest not found: {checksum_file}.")
    root = checksum_file.parent
    expected = load_checksums(checksum_file)
    actual_paths = set(_tracked_files(root))
    expected_paths = set(expected)
    missing = sorted(expected_paths - actual_paths)
    unexpected = sorted(actual_paths - expected_paths)
    mismatched = []
    for relative in sorted(expected_paths & actual_paths):
        actual_digest = _sha256(root / relative)
        if actual_digest != expected[relative]:
            mismatched.append(relative)
    if missing or unexpected or mismatched:
        raise InvalidConfigError(
            "Frozen environment integrity failure: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}."
        )
    manifest_digest = _sha256(checksum_file)
    return {
        "status": "FROZEN_BUNDLE_VERIFIED",
        "root": str(root),
        "file_count": len(expected),
        "checksum_manifest_sha256": manifest_digest,
    }


def validate_test_access_grant(
    grant_path: str | Path,
    *,
    protocol_id: str,
) -> Mapping[str, Any]:
    path = Path(grant_path).resolve()
    if not path.is_file():
        raise InvalidConfigError(
            "Final test is locked. Create the declared grant only after method and "
            "hyperparameter selection is complete."
        )
    try:
        grant = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidConfigError(f"Invalid final test access grant: {path}.") from exc
    required = {
        "status": "UNLOCKED",
        "protocol_id": str(protocol_id),
        "acknowledgement": "METHOD_SELECTION_COMPLETE",
    }
    for key, expected in required.items():
        if grant.get(key) != expected:
            raise InvalidConfigError(
                f"Final test access grant field {key!r} must equal {expected!r}."
            )
    method_hash = str(grant.get("method_lock_sha256", ""))
    if len(method_hash) != 64 or any(ch not in "0123456789abcdef" for ch in method_hash.lower()):
        raise InvalidConfigError("Final test access grant requires method_lock_sha256.")
    if not str(grant.get("opened_at", "")).strip():
        raise InvalidConfigError("Final test access grant requires opened_at.")
    return grant
