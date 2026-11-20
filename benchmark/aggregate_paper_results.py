"""Aggregate completed validation metrics without copying run artifacts into Git."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = (
    "useful_spot_gpu_hours",
    "lost_spot_gpu_hours",
    "spot_queued_gpu_hours",
    "hp_queued_gpu_hours",
    "hp_gang_violation_rate",
    "spot_evictions",
)


def _publication_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("publication_metrics", payload)
    if not isinstance(metrics, dict):
        raise ValueError(f"No publication metrics mapping in {path}.")
    if metrics.get("observation_complete") is not True:
        raise ValueError(f"Incomplete validation result rejected: {path}.")
    missing = [key for key in METRICS if key not in metrics]
    if missing:
        raise ValueError(f"Missing metrics {missing} in {path}.")
    return metrics


def aggregate(inputs: list[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for specification in inputs:
        if "=" not in specification:
            raise ValueError("Each input must use METHOD=PATH syntax.")
        method, raw_path = specification.split("=", 1)
        grouped.setdefault(method, []).append(_publication_metrics(Path(raw_path)))
    rows = []
    for method, records in grouped.items():
        row: dict[str, Any] = {"method": method, "runs": len(records)}
        for metric in METRICS:
            values = np.asarray([float(record[metric]) for record in records])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return sorted(rows, key=lambda row: str(row["method"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Repeat METHOD=PATH for every completed publication_metrics.json or validation.json.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = aggregate(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".json":
        args.output.write_text(
            json.dumps({"schema_version": "1.0", "rows": rows}, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"status": "completed", "rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
