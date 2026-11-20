#!/usr/bin/env python3
"""Fit SRLinear residual dispersion from completed training predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(batch: Path, output: Path, *, quota_quantile: float) -> dict:
    manifest_path = batch / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Forecast calibration source requires manifest.json.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cadence = int(manifest.get("origin_cadence_hours", 24))
    prediction_files = sorted(batch.glob("srlinear/seed*/predictions.jsonl"))
    if len(prediction_files) < 3:
        raise ValueError("SRLinear quota calibration requires three completed seed files.")
    residual_rows = []
    seeds = []
    source_hashes = {}
    for path in prediction_files:
        status = json.loads((path.parent / "status.json").read_text(encoding="utf-8"))
        if status.get("state") != "COMPLETED" or int(status.get("total_origins", 0)) < 2:
            raise ValueError(f"Incomplete SRLinear source run: {path.parent}.")
        seeds.append(int(status["seed"]))
        source_hashes[str(path.relative_to(batch))] = _sha256(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            target = np.asarray(row["target_sum_organization_max"], dtype=np.float64)
            point = np.asarray(row["pool_point"], dtype=np.float64)
            if target.shape != point.shape:
                raise ValueError(f"Prediction shape mismatch in {path}.")
            residual_rows.append(target - point)
    residual = np.asarray(residual_rows, dtype=np.float64)
    quantiles = (0.90, 0.95, 0.98, 0.99)
    payload = {
        "schema_version": "1.0",
        "artifact_type": "srlinear_quota_calibration",
        "fit_split": "train",
        "qualification": "publication",
        "seeds": sorted(seeds),
        "source_prediction_sha256": source_hashes,
        "source_manifest_sha256": _sha256(manifest_path),
        "origin_cadence_hours": cadence,
        "residual_count": int(len(residual)),
        "residual_sigma_by_horizon": np.std(residual, axis=0, ddof=1).tolist(),
        "conformal_offsets_by_quantile": {
            f"{value:.2f}": np.quantile(
                residual, value, axis=0, method="higher"
            ).tolist()
            for value in quantiles
        },
        "quota_quantile": float(quota_quantile),
        "test_outcomes_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quota-quantile", type=float, default=0.98)
    args = parser.parse_args()
    if not 0.0 < args.quota_quantile < 1.0:
        raise SystemExit("--quota-quantile must lie in (0, 1).")
    payload = fit(
        args.forecast_batch.resolve(), args.output.resolve(),
        quota_quantile=args.quota_quantile,
    )
    print(json.dumps({
        "status": "completed",
        "output": str(args.output.resolve()),
        "residual_count": payload["residual_count"],
        "seeds": payload["seeds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
