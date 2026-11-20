#!/usr/bin/env python3
"""Paired moving block bootstrap for rolling forecast comparisons."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PredictionLossSeries:
    model_id: str
    seed: int
    origins: tuple[str, ...]
    losses: np.ndarray
    targets: np.ndarray
    path: Path
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_prediction_losses(path: Path) -> PredictionLossSeries:
    """Load one prediction artifact and compute per-origin aggregate MAE."""

    path = path.resolve()
    origins: list[str] = []
    losses: list[float] = []
    targets: list[np.ndarray] = []
    model_id: str | None = None
    seed: int | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("point_scored") is not True:
                raise ValueError(f"{path}:{line_number} is not point scored.")
            current_model = str(record["model_id"])
            current_seed = int(record["seed"])
            if model_id is None:
                model_id = current_model
                seed = current_seed
            if current_model != model_id or current_seed != seed:
                raise ValueError(f"{path} mixes model or seed identities.")
            origin = str(record["origin"])
            prediction = np.asarray(record["pool_point"], dtype=np.float64)
            target = np.asarray(
                record["target_sum_organization_max"], dtype=np.float64
            )
            if prediction.ndim != 1 or prediction.shape != target.shape or not len(target):
                raise ValueError(f"{path}:{line_number} has invalid forecast shape.")
            if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(target)):
                raise ValueError(f"{path}:{line_number} contains nonfinite values.")
            origins.append(origin)
            losses.append(float(np.abs(prediction - target).mean()))
            targets.append(target)
    if model_id is None or seed is None or not origins:
        raise ValueError(f"{path} contains no scored predictions.")
    if len(set(origins)) != len(origins):
        raise ValueError(f"{path} contains duplicate origins.")
    parsed_origins = [datetime.fromisoformat(value) for value in origins]
    if any(right <= left for left, right in zip(parsed_origins, parsed_origins[1:])):
        raise ValueError(f"{path} origins are not strictly increasing.")
    return PredictionLossSeries(
        model_id=model_id,
        seed=seed,
        origins=tuple(origins),
        losses=np.asarray(losses, dtype=np.float64),
        targets=np.stack(targets),
        path=path,
        sha256=_sha256(path),
    )


def validate_matched_series(
    candidate: PredictionLossSeries,
    baseline_runs: Iterable[PredictionLossSeries],
    *,
    expected_cadence_hours: float = 24.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate pairing and return candidate and mean baseline losses."""

    baseline_runs = tuple(baseline_runs)
    if not baseline_runs:
        raise ValueError("At least one baseline prediction run is required.")
    seeds = [run.seed for run in baseline_runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Baseline prediction runs contain duplicate seeds.")
    for run in baseline_runs:
        if run.origins != candidate.origins:
            raise ValueError(
                f"Origin mismatch between {candidate.path} and {run.path}."
            )
        if run.targets.shape != candidate.targets.shape or not np.array_equal(
            run.targets, candidate.targets
        ):
            raise ValueError(
                f"Target mismatch between {candidate.path} and {run.path}."
            )
    parsed = [datetime.fromisoformat(value) for value in candidate.origins]
    cadence_seconds = float(expected_cadence_hours) * 3600.0
    gaps = np.asarray(
        [(right - left).total_seconds() for left, right in zip(parsed, parsed[1:])],
        dtype=np.float64,
    )
    if gaps.size and not np.allclose(gaps, cadence_seconds, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"Origins do not have the required {expected_cadence_hours:g} hour cadence."
        )
    baseline_losses = np.stack([run.losses for run in baseline_runs]).mean(axis=0)
    return candidate.losses.copy(), baseline_losses


def circular_block_bootstrap(
    paired_differences: np.ndarray,
    *,
    block_length: int,
    replicates: int,
    seed: int,
    confidence_level: float = 0.95,
    simultaneous_comparisons: int = 1,
) -> dict[str, float | int | list[float]]:
    """Bootstrap the mean paired difference using circular moving blocks."""

    differences = np.asarray(paired_differences, dtype=np.float64).reshape(-1)
    if not differences.size or not np.all(np.isfinite(differences)):
        raise ValueError("Paired differences must be finite and nonempty.")
    if block_length <= 0 or block_length > len(differences):
        raise ValueError("Block length must lie between one and the series length.")
    if replicates < 100:
        raise ValueError("At least 100 bootstrap replicates are required.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("Confidence level must lie in (0, 1).")
    if simultaneous_comparisons <= 0:
        raise ValueError("Simultaneous comparison count must be positive.")

    rng = np.random.default_rng(int(seed))
    block_count = math.ceil(len(differences) / int(block_length))
    starts = rng.integers(
        0, len(differences), size=(int(replicates), block_count), endpoint=False
    )
    offsets = np.arange(int(block_length), dtype=np.int64)
    indices = (starts[:, :, None] + offsets[None, None, :]) % len(differences)
    indices = indices.reshape(int(replicates), -1)[:, : len(differences)]
    bootstrap_means = differences[indices].mean(axis=1)
    observed = float(differences.mean())
    alpha = 1.0 - float(confidence_level)
    interval = np.quantile(
        bootstrap_means, [alpha / 2.0, 1.0 - alpha / 2.0]
    )
    familywise_alpha = alpha / int(simultaneous_comparisons)
    simultaneous_interval = np.quantile(
        bootstrap_means,
        [familywise_alpha / 2.0, 1.0 - familywise_alpha / 2.0],
    )
    centered = bootstrap_means - observed
    one_sided_p = float(
        (np.count_nonzero(centered <= observed) + 1) / (len(centered) + 1)
    )
    return {
        "block_length_origins": int(block_length),
        "replicates": int(replicates),
        "seed": int(seed),
        "observed_mean_difference_gpu": observed,
        "bootstrap_mean_difference_gpu": float(bootstrap_means.mean()),
        "confidence_level": float(confidence_level),
        "percentile_ci_gpu": [float(interval[0]), float(interval[1])],
        "familywise_confidence_level": float(1.0 - familywise_alpha),
        "familywise_percentile_ci_gpu": [
            float(simultaneous_interval[0]),
            float(simultaneous_interval[1]),
        ],
        "bootstrap_probability_candidate_better": float(
            np.mean(bootstrap_means < 0.0)
        ),
        "centered_bootstrap_one_sided_p": one_sided_p,
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Return Holm adjusted p-values while preserving input labels."""

    if not p_values:
        return {}
    if any(not 0.0 <= value <= 1.0 for value in p_values.values()):
        raise ValueError("P-values must lie in [0, 1].")
    ordered = sorted(p_values, key=lambda name: (p_values[name], name))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, (total - rank) * float(p_values[name]))
        adjusted[name] = min(1.0, running)
    return adjusted


def _require_passing_audit(run_dir: Path) -> dict[str, Any]:
    audit_path = run_dir / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise ValueError(f"Forecast run audit is not PASS: {audit_path}.")
    return audit


def _prediction_paths(run_dir: Path, model_id: str) -> tuple[Path, ...]:
    paths = tuple(sorted((run_dir / model_id).glob("seed*/predictions.jsonl")))
    if not paths:
        raise ValueError(f"No prediction artifacts found for {model_id} in {run_dir}.")
    return paths


def _verify_audit_hashes(
    audit: dict[str, Any], series: Iterable[PredictionLossSeries]
) -> None:
    expected = {
        (str(record["model_id"]), int(record["seed"])): str(
            record["prediction_sha256"]
        )
        for record in audit.get("runs", [])
    }
    for run in series:
        key = (run.model_id, run.seed)
        if expected.get(key) != run.sha256:
            raise ValueError(f"Prediction hash is not covered by passing audit: {run.path}.")


def _git_state() -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    return {"commit": commit, "dirty": bool(status)}


def _derived_seed(master_seed: int, label: str, block_length: int) -> int:
    label_word = int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "big")
    sequence = np.random.SeedSequence([int(master_seed), label_word, int(block_length)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def run_comparison(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    candidate_run_dir = args.candidate_run_dir.resolve()
    baseline_run_dir = args.baseline_run_dir.resolve()
    candidate_audit = _require_passing_audit(candidate_run_dir)
    baseline_audit = _require_passing_audit(baseline_run_dir)
    candidate_paths = _prediction_paths(candidate_run_dir, args.candidate_model)
    if len(candidate_paths) != 1:
        raise ValueError("Deterministic candidate comparison requires exactly one run.")
    candidate = load_prediction_losses(candidate_paths[0])
    _verify_audit_hashes(candidate_audit, (candidate,))

    comparisons: dict[str, Any] = {}
    primary_p_values: dict[str, float] = {}
    for baseline_name in args.baselines:
        baseline_paths = _prediction_paths(baseline_run_dir, baseline_name)
        baseline_runs = tuple(load_prediction_losses(path) for path in baseline_paths)
        if len(baseline_runs) < args.minimum_baseline_seeds:
            raise ValueError(
                f"{baseline_name} has {len(baseline_runs)} runs, fewer than required "
                f"{args.minimum_baseline_seeds}."
            )
        if any(run.model_id != baseline_name for run in baseline_runs):
            raise ValueError(f"Prediction model identity does not match {baseline_name}.")
        _verify_audit_hashes(baseline_audit, baseline_runs)
        candidate_losses, baseline_losses = validate_matched_series(
            candidate,
            baseline_runs,
            expected_cadence_hours=args.expected_cadence_hours,
        )
        differences = candidate_losses - baseline_losses
        blocks: dict[str, Any] = {}
        for block_length in args.block_lengths:
            result = circular_block_bootstrap(
                differences,
                block_length=block_length,
                replicates=args.replicates,
                seed=_derived_seed(args.seed, baseline_name, block_length),
                confidence_level=args.confidence_level,
                simultaneous_comparisons=len(args.baselines),
            )
            blocks[str(block_length)] = result
        primary = blocks[str(args.primary_block_length)]
        primary_p_values[baseline_name] = float(
            primary["centered_bootstrap_one_sided_p"]
        )
        baseline_mae = float(baseline_losses.mean())
        candidate_mae = float(candidate_losses.mean())
        comparisons[baseline_name] = {
            "baseline_model": baseline_name,
            "baseline_seeds": [run.seed for run in baseline_runs],
            "baseline_prediction_artifacts": [
                {"path": str(run.path), "sha256": run.sha256}
                for run in baseline_runs
            ],
            "candidate_mae_gpu": candidate_mae,
            "baseline_mean_seed_mae_gpu": baseline_mae,
            "mean_difference_gpu": candidate_mae - baseline_mae,
            "relative_mae_improvement": (
                (baseline_mae - candidate_mae) / baseline_mae
            ),
            "candidate_daily_wins": int(np.count_nonzero(differences < 0.0)),
            "candidate_daily_ties": int(np.count_nonzero(differences == 0.0)),
            "origin_count": int(len(differences)),
            "block_results": blocks,
        }

    adjusted = holm_adjust(primary_p_values)
    for baseline_name, adjusted_p in adjusted.items():
        comparisons[baseline_name]["primary_holm_adjusted_one_sided_p"] = adjusted_p
        comparisons[baseline_name]["primary_familywise_significant"] = bool(
            adjusted_p < (1.0 - args.confidence_level)
            and comparisons[baseline_name]["block_results"][
                str(args.primary_block_length)
            ]["familywise_percentile_ci_gpu"][1]
            < 0.0
        )

    timestamp = datetime.now(timezone.utc)
    output_dir = (
        args.output_root.resolve()
        / f"{args.candidate_model}_paired_block_bootstrap_{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    result = {
        "schema_version": "1.0",
        "status": "PASS",
        "created_at": timestamp.isoformat(),
        "analysis": "paired_circular_moving_block_bootstrap",
        "interpretation": "Negative differences favor the candidate.",
        "candidate_model": args.candidate_model,
        "candidate_seed": candidate.seed,
        "candidate_prediction_artifact": {
            "path": str(candidate.path),
            "sha256": candidate.sha256,
        },
        "candidate_run_audit": str(candidate_run_dir / "audit.json"),
        "baseline_run_audit": str(baseline_run_dir / "audit.json"),
        "origin_count": len(candidate.origins),
        "origin_start": candidate.origins[0],
        "origin_end": candidate.origins[-1],
        "horizons_per_origin": int(candidate.targets.shape[1]),
        "expected_cadence_hours": args.expected_cadence_hours,
        "bootstrap_master_seed": args.seed,
        "replicates": args.replicates,
        "primary_block_length_origins": args.primary_block_length,
        "sensitivity_block_lengths_origins": list(args.block_lengths),
        "confidence_level": args.confidence_level,
        "multiple_comparison_method": "Holm on centered bootstrap one-sided p-values; Bonferroni familywise percentile intervals.",
        "baseline_seed_aggregation": "Mean per-origin absolute error across baseline seeds before paired resampling.",
        "comparisons": comparisons,
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        },
        "git": _git_state(),
        "implementation_sha256": _sha256(Path(__file__)),
    }
    _write_json_atomic(output_dir / "results.json", result)
    _write_json_atomic(
        output_dir / "status.json",
        {
            "schema_version": "1.0",
            "status": "COMPLETED",
            "result_sha256": _sha256(output_dir / "results.json"),
        },
    )
    return output_dir, result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--baselines", nargs="+", required=True)
    parser.add_argument("--block-lengths", nargs="+", type=int, default=[3, 7, 14])
    parser.add_argument("--primary-block-length", type=int, default=7)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--expected-cadence-hours", type=float, default=24.0)
    parser.add_argument("--minimum-baseline-seeds", type=int, default=3)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "runs" / "forecast_analysis",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.primary_block_length not in args.block_lengths:
        raise ValueError("Primary block length must appear in block lengths.")
    if len(set(args.baselines)) != len(args.baselines):
        raise ValueError("Baseline names must be unique.")
    output_dir, result = run_comparison(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    print(f"OUTPUT_DIR={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
