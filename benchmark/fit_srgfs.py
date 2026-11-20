#!/usr/bin/env python3
"""Build the train only SRGFS artifact from frozen controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_train_artifact(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("fit_split") != "train":
        raise ValueError(f"SRGFS source is not train fitted: {path}")
    if payload.get("test_outcomes_accessed") is not False:
        raise ValueError(f"SRGFS source does not lock test outcomes: {path}")
    if payload.get("qualification") != "publication":
        raise ValueError(f"SRGFS source is not publication qualified: {path}")
    return payload


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_candidate(
    *,
    contextual_path: Path,
    output_dir: Path,
    cooldown_seconds: int,
    hp_rise_trigger_fraction: float,
    eviction_failure_trigger: float,
) -> Path:
    if cooldown_seconds <= 0 or cooldown_seconds % 300 != 0:
        raise ValueError("cooldown_seconds must be a positive multiple of 300.")
    contextual = _read_train_artifact(contextual_path)
    if "native_safe_margin_fraction" not in contextual:
        raise ValueError("Contextual artifact lacks the native safe margin.")
    common = {
        "schema_version": "1.0",
        "qualification": "publication",
        "fit_split": "train",
        "test_outcomes_accessed": False,
        "source_report": contextual["source_report"],
        "source_report_sha256": contextual["source_report_sha256"],
        "contextual_artifact_sha256": _sha256(contextual_path),
        "native_safe_margin_fraction": float(contextual["native_safe_margin_fraction"]),
        "cooldown_seconds": int(cooldown_seconds),
        "hp_rise_trigger_fraction": float(hp_rise_trigger_fraction),
        "eviction_failure_trigger": float(eviction_failure_trigger),
        "trigger_lost_gpu_hours": 0.0,
        "control_interval_seconds": 300,
        "selection_status": "validation_candidate",
        "test_selection_complete": False,
    }
    contextual_payload = dict(common)
    contextual_payload.update({
        "policy_type": "srgfs_contextual_mix",
        "linear": {
            key: contextual[key]
            for key in (
                "coefficients",
                "intercept",
                "feature_mean",
                "feature_scale",
                "safety_offset",
                "minimum_residual",
                "maximum_residual",
                "feature_names",
                "train_risk_tick_rate",
            )
        },
        "mixing_rule": "bounded_contextual_target_projection_v1",
    })
    output = output_dir / "srgfs_contextual_mix.json"
    _write(output, contextual_payload)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contextual", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cooldown-seconds", type=int, default=600)
    parser.add_argument("--hp-rise-trigger-fraction", type=float, default=0.01)
    parser.add_argument("--eviction-failure-trigger", type=float, default=0.10)
    args = parser.parse_args()
    output = build_candidate(
        contextual_path=args.contextual.resolve(),
        output_dir=args.output_dir.resolve(),
        cooldown_seconds=args.cooldown_seconds,
        hp_rise_trigger_fraction=args.hp_rise_trigger_fraction,
        eviction_failure_trigger=args.eviction_failure_trigger,
    )
    print(json.dumps({"output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
