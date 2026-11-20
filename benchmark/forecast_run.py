#!/usr/bin/env python3
"""Run causal rolling forecast baselines on Native GFS history."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_SRC = REPOSITORY_ROOT / "environment" / "src"
for import_root in (REPOSITORY_ROOT, ENVIRONMENT_SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from benchmark.forecast_data import (  # noqa: E402
    align_prediction,
    build_origin_data,
    load_protocol,
    load_snapshot_history,
    sha256_file,
    split_origins,
)
from benchmark.forecast_metrics import aggregate_forecast_records  # noqa: E402
from benchmark.seeds import (  # noqa: E402
    STOCHASTIC_SEED_DERIVATION,
    derive_run_seeds,
)
from methods.forecasters import (  # noqa: E402
    AdaptiveSeasonalResidualLinearForecaster,
    DLinearForecaster,
    NLinearForecaster,
    OrgLinearReleasedForecaster,
    OrgLinearRForecaster,
    PatchTSTForecaster,
    SeasonalNaiveForecaster,
    TemporalMambaForecaster,
)


MODEL_IDS = (
    "seasonal_naive",
    "dlinear",
    "nlinear",
    "srlinear",
    "patchtst",
    "temporal_mamba",
    "orglinear_released",
    "orglinear_r",
)

IMPLEMENTATION_PATHS = tuple(
    [
        REPOSITORY_ROOT / "benchmark" / "forecast_run.py",
        REPOSITORY_ROOT / "benchmark" / "forecast_data.py",
        REPOSITORY_ROOT / "benchmark" / "forecast_metrics.py",
        REPOSITORY_ROOT / "benchmark" / "seeds.py",
        REPOSITORY_ROOT / "methods" / "forecasters" / "__init__.py",
        REPOSITORY_ROOT / "methods" / "forecasters" / "models.py",
        REPOSITORY_ROOT / "methods" / "forecasters" / "advanced.py",
        REPOSITORY_ROOT / "methods" / "forecasters" / "srlinear.py",
        REPOSITORY_ROOT / "methods" / "forecasters" / "srlinear_components.py",
    ]
    + sorted(
        (ENVIRONMENT_SRC / "research_sim" / "gfs" / "estimator").rglob("*.py")
    )
)
ENVIRONMENT_LOCK = REPOSITORY_ROOT / "benchmark" / "forecast_environment.lock.yml"


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_state() -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    return {"commit": commit, "dirty": bool(status)}


def _driver_version() -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()[0].strip()
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _implementation_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(REPOSITORY_ROOT)): sha256_file(path)
        for path in IMPLEMENTATION_PATHS
    }


def _runtime_manifest(selected_devices: dict[str, str]) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": _package_version("scipy"),
        "scikit_learn": _package_version("scikit-learn"),
        "joblib": _package_version("joblib"),
        "threadpoolctl": _package_version("threadpoolctl"),
        "pyyaml": _package_version("PyYAML"),
        "mamba_ssm": _package_version("mamba_ssm"),
        "causal_conv1d": _package_version("causal_conv1d"),
        "triton": _package_version("triton"),
        "chinese_calendar": _package_version("chinese-calendar"),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "selected_devices": dict(selected_devices),
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "driver": _driver_version(),
    }


def load_suite(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != "1.0":
        raise ValueError("Forecast suite requires schema_version '1.0'.")
    if raw.get("test_policy") != "locked":
        raise ValueError("Forecast suite must keep final test locked.")
    if raw.get("development_split") != "train":
        raise ValueError("Forecast model selection must run on train only.")
    if raw.get("randomness", {}).get("stochastic_seed_derivation") != STOCHASTIC_SEED_DERIVATION:
        raise ValueError("Unsupported forecast seed derivation.")
    if (REPOSITORY_ROOT / "test_access_grant.json").exists():
        raise ValueError("Final test access grant exists during forecast development.")
    raw["_suite_path"] = str(path.resolve())
    return raw


def _select_device(
    suite: dict[str, Any], override: str | None, model_id: str
) -> torch.device:
    requested = override or str(
        suite["models"][model_id].get("device", suite["resources"]["device"])
    )
    gpu_index = int(suite["resources"].get("gpu_index", 0))
    if requested == "auto":
        requested = f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA forecast device requested but unavailable.")
    return device


def _model_settings(
    suite: dict[str, Any], model_id: str, *, smoke: bool
) -> dict[str, Any]:
    settings = dict(suite["models"][model_id])
    if smoke and model_id != "seasonal_naive":
        if "epochs" in settings:
            settings["epochs"] = int(
                settings.get("smoke_epochs", suite["smoke"]["epochs"])
            )
        if "batch_size" in settings:
            settings["batch_size"] = int(
                settings.get("smoke_batch_size", suite["smoke"]["batch_size"])
            )
        if "patience" in settings:
            settings["patience"] = 1
        if "max_iter" in settings:
            settings["max_iter"] = int(settings.get("smoke_max_iter", 5))
    return settings


def _build_model(
    suite: dict[str, Any],
    *,
    model_id: str,
    seed: int,
    device: torch.device,
    run_dir: Path,
    smoke: bool,
):
    rolling = suite["rolling"]
    settings = _model_settings(suite, model_id, smoke=smoke)
    lookback = int(rolling["lookback_hours"])
    horizon = int(rolling["horizon_hours"])
    workers = int(
        settings.get(
            "data_loader_workers",
            suite["resources"]["data_loader_workers"],
        )
    )
    if model_id == "seasonal_naive":
        return SeasonalNaiveForecaster(
            horizon=horizon,
            seasonal_lag=int(settings["seasonal_lag_hours"]),
        )
    if model_id == "dlinear":
        return DLinearForecaster(
            lookback=lookback,
            horizon=horizon,
            moving_average=int(settings["moving_average"]),
            epochs=int(settings["epochs"]),
            batch_size=int(settings["batch_size"]),
            learning_rate=float(settings["learning_rate"]),
            weight_decay=float(settings["weight_decay"]),
            validation_hours=int(settings["validation_hours"]),
            sample_stride_hours=int(settings["sample_stride_hours"]),
            patience=int(settings["patience"]),
            device=device,
            seed=seed,
            num_workers=workers,
        )
    if model_id == "srlinear":
        return AdaptiveSeasonalResidualLinearForecaster(
            horizon=horizon,
            periods=tuple(int(value) for value in settings["periods"]),
            base_period=int(settings["base_period"]),
            recent_windows=tuple(
                int(value) for value in settings["recent_windows"]
            ),
            residual_lookback=int(settings["residual_lookback"]),
            validation_hours=int(settings["validation_hours"]),
            validation_cadence_hours=int(settings["validation_cadence_hours"]),
            tuning_validation_cadence_hours=int(
                settings["tuning_validation_cadence_hours"]
            ),
            sample_stride_hours=int(settings["sample_stride_hours"]),
            ridge_alphas=tuple(float(value) for value in settings["ridge_alphas"]),
            reconciliation_grid=tuple(
                float(value) for value in settings["reconciliation_grid"]
            ),
            epochs=int(settings["epochs"]),
            learning_rate=float(settings["learning_rate"]),
            weight_decay=float(settings["weight_decay"]),
            patience=int(settings["patience"]),
            organization_loss_weight=float(settings["organization_loss_weight"]),
            gate_stability_weight=float(settings["gate_stability_weight"]),
            loss_rmse_weight=float(settings["loss_rmse_weight"]),
            loss_tail_weight=float(settings["loss_tail_weight"]),
            loss_tail_fraction=float(settings["loss_tail_fraction"]),
            loss_peak_weight=float(settings["loss_peak_weight"]),
            loss_underprediction_weight=float(
                settings["loss_underprediction_weight"]
            ),
            loss_peak_quantile=float(settings["loss_peak_quantile"]),
            validation_rmse_weight=float(settings["validation_rmse_weight"]),
            base_gate_logit=float(settings["base_gate_logit"]),
            gradient_clip=float(settings["gradient_clip"]),
            device=device,
            seed=seed,
        )
    shared_torch = {
        "lookback": lookback,
        "horizon": horizon,
        "epochs": int(settings.get("epochs", 1)),
        "batch_size": int(settings.get("batch_size", 64)),
        "learning_rate": float(settings.get("learning_rate", 1e-3)),
        "weight_decay": float(settings.get("weight_decay", 0.0)),
        "validation_hours": int(settings.get("validation_hours", 168)),
        "sample_stride_hours": int(settings.get("sample_stride_hours", 1)),
        "patience": int(settings.get("patience", 3)),
        "device": device,
        "seed": seed,
        "num_workers": workers,
        "precision": str(settings.get("precision", "fp32")),
        "fused_optimizer": bool(settings.get("fused_optimizer", False)),
        "gpu_resident": bool(settings.get("gpu_resident", False)),
        "validation_batch_size": int(
            settings.get("validation_batch_size", settings.get("batch_size", 64))
        ),
    }
    if model_id == "nlinear":
        return NLinearForecaster(**shared_torch)
    if model_id == "patchtst":
        return PatchTSTForecaster(
            patch_length=int(settings["patch_length"]),
            patch_stride=int(settings["patch_stride"]),
            model_width=int(settings["model_width"]),
            heads=int(settings["heads"]),
            layers=int(settings["layers"]),
            feedforward_width=int(settings["feedforward_width"]),
            dropout=float(settings["dropout"]),
            **shared_torch,
        )
    if model_id == "temporal_mamba":
        return TemporalMambaForecaster(
            patch_length=int(settings["patch_length"]),
            patch_stride=int(settings["patch_stride"]),
            model_width=int(settings["model_width"]),
            state_width=int(settings["state_width"]),
            convolution_width=int(settings["convolution_width"]),
            expansion=int(settings["expansion"]),
            layers=int(settings["layers"]),
            dropout=float(settings["dropout"]),
            **shared_torch,
        )
    adapter = {
        "orglinear_released": OrgLinearReleasedForecaster,
        "orglinear_r": OrgLinearRForecaster,
    }[model_id]
    return adapter(
        lookback=lookback,
        horizon=horizon,
        epochs=int(settings["epochs"]),
        batch_size=int(settings["batch_size"]),
        learning_rate=float(settings["learning_rate"]),
        device=device,
        seed=seed,
        num_workers=workers,
        run_dir=run_dir,
        quantiles=tuple(float(q) for q in rolling["quantiles"]),
    )


def _model_seeds(suite: dict[str, Any], model_id: str, *, mode: str) -> tuple[int, ...]:
    master = int(suite["randomness"]["master_seed"])
    if not bool(suite["models"][model_id].get("stochastic", True)):
        return (master,)
    count = int(
        suite[mode]["stochastic_seed_count"]
        if mode in {"smoke", "preflight", "pilot"}
        else suite["randomness"]["stochastic_seed_count"]
    )
    return derive_run_seeds(master, count)


def _quantile_string_map(values: dict[float, np.ndarray]) -> dict[str, list[float]]:
    return {
        f"{float(quantile):.2f}": np.asarray(prediction, dtype=np.float64).tolist()
        for quantile, prediction in sorted(values.items())
    }


def _read_prediction_records(
    path: Path, *, repair_trailing_partial: bool
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    records: list[dict[str, Any]] = []
    for line_index, line in enumerate(raw_lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            is_final_nonempty = not any(item.strip() for item in raw_lines[line_index + 1 :])
            if not repair_trailing_partial or not is_final_nonempty:
                raise ValueError(f"Invalid prediction JSON at line {line_index + 1}: {path}.")
            break
        if int(record.get("origin_index", -1)) != len(records):
            raise ValueError(f"Nonconsecutive prediction origin at line {line_index + 1}: {path}.")
        records.append(record)
    if repair_trailing_partial and len(records) != len([line for line in raw_lines if line.strip()]):
        temporary = path.with_name(f".{path.name}.{os.getpid()}.repair")
        temporary.write_text(
            "".join(json.dumps(record, sort_keys=True, allow_nan=False) + "\n" for record in records),
            encoding="utf-8",
        )
        temporary.replace(path)
    return records


def run_one(
    suite: dict[str, Any],
    *,
    model_id: str,
    seed: int,
    origins: tuple[pd.Timestamp, ...],
    source,
    calendar_anchor: pd.Timestamp,
    device: torch.device,
    run_dir: Path,
    smoke_hyperparameters: bool,
    score_from_start: bool,
    resume: bool = False,
) -> dict[str, Any]:
    status_path = run_dir / "status.json"
    prediction_path = run_dir / "predictions.jsonl"
    if resume:
        if not run_dir.is_dir() or not status_path.exists():
            raise FileNotFoundError(f"Cannot resume missing forecast run: {run_dir}.")
        prior_status = json.loads(status_path.read_text(encoding="utf-8"))
        if prior_status.get("model_id") != model_id or int(prior_status.get("seed", -1)) != seed:
            raise ValueError(f"Resume identity mismatch: {run_dir}.")
        started_at = str(prior_status.get("started_at", datetime.now(timezone.utc).isoformat()))
        prior_wall_seconds = float(prior_status.get("wall_seconds", 0.0))
        prior_peak_cuda_memory = int(prior_status.get("peak_cuda_memory_bytes", 0))
        records = _read_prediction_records(prediction_path, repair_trailing_partial=True)
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        started_at = datetime.now(timezone.utc).isoformat()
        prior_wall_seconds = 0.0
        prior_peak_cuda_memory = 0
        records = []
    if len(records) > len(origins):
        raise ValueError(f"Saved predictions exceed configured origins: {run_dir}.")
    for index, record in enumerate(records):
        if record.get("model_id") != model_id or int(record.get("seed", -1)) != seed:
            raise ValueError(f"Saved prediction identity mismatch: {run_dir}.")
        if pd.Timestamp(record["origin"]) != origins[index]:
            raise ValueError(f"Saved prediction origin mismatch at index {index}: {run_dir}.")
    _write_json_atomic(
        status_path,
        {
            "schema_version": "1.0",
            "state": "RUNNING",
            "model_id": model_id,
            "seed": int(seed),
            "started_at": started_at,
            "completed_origins": len(records),
            "total_origins": len(origins),
            "resumed": bool(resume),
            "wall_seconds": prior_wall_seconds,
        },
    )
    model = _build_model(
        suite,
        model_id=model_id,
        seed=seed,
        device=device,
        run_dir=run_dir,
        smoke=smoke_hyperparameters,
    )
    rolling = suite["rolling"]
    horizon = int(rolling["horizon_hours"])
    calibration_origins = 0 if score_from_start else int(rolling["calibration_origins"])
    quantiles = tuple(float(q) for q in rolling["quantiles"])
    residual_history = [
        np.asarray(record["target_sum_organization_max"], dtype=np.float64)
        - np.asarray(record["pool_point"], dtype=np.float64)
        for record in records
    ]
    started_clock = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        with prediction_path.open("a" if resume else "w", encoding="utf-8") as prediction_file:
            for index in range(len(records), len(origins)):
                origin = origins[index]
                calendar_timestamp = calendar_anchor + (origin - source.recorder_anchor)
                origin_data = build_origin_data(
                    source,
                    origin=origin,
                    horizon=horizon,
                    minimum_history_hours=int(rolling["lookback_hours"]),
                )
                if origin_data.causal_history.index.max() >= origin:
                    raise AssertionError("Forecast input crossed the causal origin.")
                output = model.fit_predict(
                    origin_data.causal_history,
                    origin,
                    run_dir,
                    calendar_timestamp=calendar_timestamp,
                )
                aligned = align_prediction(
                    prediction=output.point,
                    prediction_columns=output.columns,
                    target_columns=origin_data.target_columns,
                )
                pool_point = aligned.sum(axis=1)
                conformal: dict[float, np.ndarray] = {}
                if len(residual_history) >= calibration_origins and residual_history:
                    residuals = np.asarray(residual_history, dtype=np.float64)
                    for quantile in quantiles:
                        offset = np.quantile(
                            residuals,
                            quantile,
                            axis=0,
                            method="higher",
                        )
                        conformal[quantile] = np.clip(pool_point + offset, 0.0, None)
                point_scored = True
                quantile_scored = index >= calibration_origins
                record = {
                    "schema_version": "1.0",
                    "model_id": model_id,
                    "seed": int(seed),
                    "origin_index": index,
                    "origin": origin.isoformat(),
                    "calendar_timestamp": calendar_timestamp.isoformat(),
                    "history_end": origin_data.causal_history.index.max().isoformat(),
                    "history_rows": len(origin_data.causal_history),
                    "prediction_columns": list(output.columns),
                    "target_columns": list(origin_data.target_columns),
                    "point_by_organization": aligned.tolist(),
                    "pool_point": pool_point.tolist(),
                    "target_sum_organization_max": (
                        origin_data.target_sum_organization_max.tolist()
                    ),
                    "target_simultaneous_pool_max": (
                        origin_data.target_simultaneous_pool_max.tolist()
                    ),
                    "intrinsic_pool_quantiles": _quantile_string_map(
                        dict(output.intrinsic_pool_quantiles)
                    ),
                    "conformal_pool_quantiles": _quantile_string_map(conformal),
                    "scored": point_scored,
                    "point_scored": point_scored,
                    "quantile_scored": quantile_scored,
                }
                prediction_file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                prediction_file.flush()
                records.append(record)
                residual_history.append(
                    origin_data.target_sum_organization_max - pool_point
                )
                _write_json_atomic(
                    status_path,
                    {
                        "schema_version": "1.0",
                        "state": "RUNNING",
                        "model_id": model_id,
                        "seed": int(seed),
                        "started_at": started_at,
                        "heartbeat_at": datetime.now(timezone.utc).isoformat(),
                        "completed_origins": index + 1,
                        "total_origins": len(origins),
                        "progress_fraction": (index + 1) / len(origins),
                        "latest_origin": origin.isoformat(),
                        "resumed": bool(resume),
                        "wall_seconds": prior_wall_seconds
                        + float(time.perf_counter() - started_clock),
                        "peak_cuda_memory_bytes": max(
                            prior_peak_cuda_memory,
                            int(torch.cuda.max_memory_allocated(device))
                            if device.type == "cuda"
                            else 0,
                        ),
                    },
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        metrics = aggregate_forecast_records(records, capacity=source.capacity)
        metrics.update(
            {
                "schema_version": "1.1",
                "model_id": model_id,
                "seed": int(seed),
                "calibration_origins": calibration_origins,
                "total_origins": len(origins),
                "wall_seconds": prior_wall_seconds
                + float(time.perf_counter() - started_clock),
                "peak_cuda_memory_bytes": max(
                    prior_peak_cuda_memory,
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else 0,
                ),
            }
        )
        _write_json_atomic(run_dir / "metrics.json", metrics)
        _write_json_atomic(
            status_path,
            {
                "schema_version": "1.0",
                "state": "COMPLETED",
                "model_id": model_id,
                "seed": int(seed),
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "completed_origins": len(origins),
                "total_origins": len(origins),
                "progress_fraction": 1.0,
                "resumed": bool(resume),
                "wall_seconds": metrics["wall_seconds"],
                "peak_cuda_memory_bytes": metrics["peak_cuda_memory_bytes"],
            },
        )
        return metrics
    except Exception as exc:
        _write_json_atomic(
            status_path,
            {
                "schema_version": "1.0",
                "state": "FAILED",
                "model_id": model_id,
                "seed": int(seed),
                "started_at": started_at,
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed_origins": len(records),
                "total_origins": len(origins),
                "resumed": bool(resume),
                "wall_seconds": prior_wall_seconds
                + float(time.perf_counter() - started_clock),
                "peak_cuda_memory_bytes": max(
                    prior_peak_cuda_memory,
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else 0,
                ),
            },
        )
        raise


def _aggregate_batch(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(str(result["model_id"]), []).append(result)
    summary: dict[str, Any] = {"schema_version": "1.0", "models": {}}
    for model_id, model_results in sorted(grouped.items()):
        maes = np.asarray([item["point"]["mae_gpu"] for item in model_results])
        rmses = np.asarray([item["point"]["rmse_gpu"] for item in model_results])
        summary["models"][model_id] = {
            "runs": len(model_results),
            "seeds": [int(item["seed"]) for item in model_results],
            "mae_gpu_mean": float(maes.mean()),
            "mae_gpu_std": float(maes.std(ddof=1)) if len(maes) > 1 else 0.0,
            "rmse_gpu_mean": float(rmses.mean()),
            "rmse_gpu_std": float(rmses.std(ddof=1)) if len(rmses) > 1 else 0.0,
        }
    return summary


def audit_batch(run_dir: Path) -> dict[str, Any]:
    """Recompute every completed run metric from durable predictions."""

    batch_dir = run_dir.resolve()
    manifest_path = batch_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing forecast manifest: {manifest_path}.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("test_locked") is not True:
        raise ValueError("Forecast batch does not prove that final test stayed locked.")
    capacity = float(manifest["capacity"])
    run_statuses = sorted(batch_dir.glob("*/seed*/status.json"))
    if not run_statuses:
        raise ValueError("Forecast batch has no model run statuses.")
    audited_runs = []
    for status_path in run_statuses:
        model_dir = status_path.parent
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") != "COMPLETED":
            raise ValueError(f"Forecast run is not complete: {model_dir}.")
        prediction_path = model_dir / "predictions.jsonl"
        records = [
            json.loads(line)
            for line in prediction_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(records) != int(status["total_origins"]):
            raise ValueError(f"Forecast origin count mismatch: {model_dir}.")
        for expected_index, record in enumerate(records):
            if int(record["origin_index"]) != expected_index:
                raise ValueError(f"Forecast origin order mismatch: {model_dir}.")
            if pd.Timestamp(record["history_end"]) >= pd.Timestamp(record["origin"]):
                raise ValueError(f"Forecast causality violation: {model_dir}.")
            point = np.asarray(record["point_by_organization"], dtype=np.float64)
            target = np.asarray(
                record["target_sum_organization_max"], dtype=np.float64
            )
            if point.ndim != 2 or target.ndim != 1 or point.shape[0] != target.shape[0]:
                raise ValueError(f"Forecast prediction shape mismatch: {model_dir}.")
            if not np.all(np.isfinite(point)) or not np.all(np.isfinite(target)):
                raise ValueError(f"Forecast artifact contains nonfinite values: {model_dir}.")
        recomputed = aggregate_forecast_records(records, capacity=capacity)
        stored = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
        missing = set(recomputed) - set(stored)
        if missing - {"point_scored_origins", "quantile_scored_origins"}:
            raise ValueError(f"Forecast metrics are incomplete: {model_dir}.")
        comparable = {key: value for key, value in recomputed.items() if key in stored}
        if comparable != {key: stored[key] for key in comparable}:
            raise ValueError(f"Forecast metrics do not reproduce: {model_dir}.")
        audited_runs.append(
            {
                "model_id": status["model_id"],
                "seed": int(status["seed"]),
                "origin_count": len(records),
                "prediction_sha256": sha256_file(prediction_path),
                "metrics_sha256": sha256_file(model_dir / "metrics.json"),
            }
        )
    audit = {
        "schema_version": "1.0",
        "status": "PASS",
        "batch_dir": str(batch_dir),
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "run_count": len(audited_runs),
        "runs": audited_runs,
    }
    _write_json_atomic(batch_dir / "audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return audit


def show_batch_status(run_dir: Path) -> dict[str, Any]:
    batch_dir = run_dir.resolve()
    root_status_path = batch_dir / "status.json"
    run_paths = sorted(batch_dir.glob("*/seed*/status.json"))
    if root_status_path.exists():
        batch_status = json.loads(root_status_path.read_text(encoding="utf-8"))
    elif run_paths:
        states = [
            json.loads(path.read_text(encoding="utf-8")).get("state")
            for path in run_paths
        ]
        state = (
            "FAILED"
            if "FAILED" in states
            else "COMPLETED" if all(item == "COMPLETED" for item in states) else "RUNNING"
        )
        batch_status = {
            "schema_version": "legacy_synthesized",
            "state": state,
            "completed_runs": sum(item == "COMPLETED" for item in states),
            "total_runs": len(states),
            "root_status_missing": True,
        }
    else:
        raise FileNotFoundError(f"Missing forecast batch statuses under: {batch_dir}.")
    payload = {"batch": batch_status, "runs": []}
    for path in run_paths:
        item = json.loads(path.read_text(encoding="utf-8"))
        item["run_dir"] = str(path.parent)
        payload["runs"].append(item)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _acquire_batch_lock(batch_dir: Path) -> Path:
    lock_path = batch_dir / ".runner.lock"
    if lock_path.exists():
        try:
            prior = json.loads(lock_path.read_text(encoding="utf-8"))
            prior_pid = int(prior["pid"])
            os.kill(prior_pid, 0)
        except (KeyError, ValueError, ProcessLookupError, json.JSONDecodeError):
            lock_path.unlink(missing_ok=True)
        except PermissionError as exc:
            raise RuntimeError(f"Cannot inspect active forecast runner lock: {lock_path}.") from exc
        else:
            raise RuntimeError(
                f"Forecast batch already has live runner PID {prior_pid}: {batch_dir}."
            )
    lock_path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return lock_path


def _resolve_batch_dir(value: str, suite_value: str) -> Path:
    if value != "latest":
        return Path(value).resolve()
    suite_path = Path(suite_value).resolve()
    raw = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    output_root = _resolve(suite_path.parent, str(raw["output_root"]))
    latest_path = output_root / "latest.json"
    if not latest_path.exists():
        raise FileNotFoundError(f"Missing forecast latest pointer: {latest_path}.")
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    return Path(latest["batch_dir"]).resolve()


def execute(args: argparse.Namespace) -> Path:
    if args.command in {"audit", "status"}:
        if args.run_dir is None:
            raise ValueError(f"{args.command} requires --run-dir.")
        selected_run_dir = _resolve_batch_dir(args.run_dir, args.suite)
        if args.command == "audit":
            audit_batch(selected_run_dir)
        else:
            show_batch_status(selected_run_dir)
        return selected_run_dir
    resuming = args.command == "resume"
    existing_manifest: dict[str, Any] | None = None
    if resuming:
        if args.run_dir is None:
            raise ValueError("resume requires --run-dir.")
        if (
            args.models != "all"
            or args.device is not None
            or args.origin_cadence_hours is not None
        ):
            raise ValueError("Resume uses the saved model plan and devices; omit overrides.")
        batch_dir = _resolve_batch_dir(args.run_dir, args.suite)
        manifest_path = batch_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing forecast manifest: {manifest_path}.")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("mode") != "run":
            raise ValueError("Only a complete run batch may be resumed.")
        suite_path = (REPOSITORY_ROOT / str(existing_manifest["suite_path"])).resolve()
        mode = "run"
    else:
        suite_path = Path(args.suite).resolve()
        mode = args.command
    suite = load_suite(suite_path)
    if args.origin_cadence_hours is not None:
        if int(args.origin_cadence_hours) < 1:
            raise ValueError("Origin cadence override must be positive.")
        suite["rolling"]["origin_cadence_hours"] = int(args.origin_cadence_hours)
    suite_base = suite_path.parent
    protocol_path = _resolve(suite_base, str(suite["environment_protocol"]))
    source_path = _resolve(suite_base, str(suite["source_snapshot"]))
    protocol = load_protocol(protocol_path)
    source = load_snapshot_history(source_path)
    if source.source_metadata.get("split_id") != "val":
        raise ValueError("Train forecast carrier must be the validation-start snapshot.")
    origins = split_origins(
        protocol,
        split=str(suite["development_split"]),
        recorder_anchor=source.recorder_anchor,
        cadence_hours=int(suite["rolling"]["origin_cadence_hours"]),
    )
    quick_mode = mode in {"smoke", "preflight", "pilot"}
    smoke_hyperparameters = mode in {"smoke", "preflight"}
    if mode in {"smoke", "pilot"}:
        origins = origins[: int(suite[mode]["max_origins"])]
    elif mode == "preflight":
        if suite[mode].get("origin_selection") != "first_and_last":
            raise ValueError("Preflight requires first_and_last origin selection.")
        origins = (origins[0], origins[-1])
    if not origins:
        raise ValueError("Forecast suite selected no rolling origins.")
    max_target_time = origins[-1] + pd.Timedelta(
        hours=int(suite["rolling"]["horizon_hours"])
    )
    if max_target_time > source.minute_history.index.max() + pd.Timedelta(minutes=1):
        raise ValueError("Forecast source does not cover the final target horizon.")

    requested_models = (
        tuple(existing_manifest["models"])
        if resuming and existing_manifest is not None
        else MODEL_IDS if args.models == "all" else tuple(args.models.split(","))
    )
    unknown = set(requested_models) - set(MODEL_IDS)
    if unknown:
        raise ValueError(f"Unknown forecast models: {sorted(unknown)}.")
    models = tuple(
        model_id
        for model_id in requested_models
        if bool(suite["models"][model_id].get("enabled", False))
    )
    torch.set_num_threads(int(suite["resources"]["torch_threads"]))
    output_root = _resolve(suite_base, str(suite["output_root"]))
    selected_devices = {
        model_id: str(_select_device(suite, args.device, model_id))
        for model_id in models
    }
    minimum_free = float(suite["resources"].get("minimum_output_free_gib", 0.0))
    free_gib = shutil.disk_usage(output_root).free / (1024**3)
    if free_gib < minimum_free:
        raise RuntimeError(
            f"Forecast output has {free_gib:.2f} GiB free, below {minimum_free:.2f} GiB."
        )
    cuda_models = [model_id for model_id, device in selected_devices.items() if device.startswith("cuda")]
    gpu_free_gib: float | None = None
    if cuda_models:
        gpu_free_gib = torch.cuda.mem_get_info(int(suite["resources"].get("gpu_index", 0)))[0] / (1024**3)
        minimum_gpu_free = float(suite["resources"].get("minimum_gpu_free_gib", 0.0))
        if gpu_free_gib < minimum_gpu_free:
            raise RuntimeError(
                f"GPU has {gpu_free_gib:.2f} GiB free, below {minimum_gpu_free:.2f} GiB."
            )
    if resuming:
        assert existing_manifest is not None
        if sha256_file(suite_path) != existing_manifest["suite_sha256"]:
            raise ValueError("Forecast suite changed after batch creation; resume refused.")
        if sha256_file(protocol_path) != existing_manifest["protocol_sha256"]:
            raise ValueError("Environment protocol changed after batch creation; resume refused.")
        if source.source_sha256 != existing_manifest["source_snapshot_sha256"]:
            raise ValueError("Forecast source snapshot changed after batch creation; resume refused.")
        expected = (
            len(origins), origins[0].isoformat(), origins[-1].isoformat()
        )
        observed = (
            int(existing_manifest["origin_count"]),
            str(existing_manifest["first_origin"]),
            str(existing_manifest["last_origin"]),
        )
        if expected != observed:
            raise ValueError("Forecast origins changed after batch creation; resume refused.")
        if selected_devices != existing_manifest["runtime"]["selected_devices"]:
            raise ValueError("Forecast device plan changed after batch creation; resume refused.")
        if _implementation_hashes() != existing_manifest["implementation_sha256"]:
            raise ValueError("Forecast implementation changed after batch creation; resume refused.")
        if sha256_file(ENVIRONMENT_LOCK) != existing_manifest["environment_lock_sha256"]:
            raise ValueError("Forecast environment lock changed after batch creation; resume refused.")
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        batch_dir = output_root / str(suite["benchmark_id"]) / mode / timestamp
        batch_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": "1.0",
        "benchmark_id": suite["benchmark_id"],
        "mode": mode,
        "split": str(suite["development_split"]),
        "split_bounds": protocol["split_bounds"][str(suite["development_split"])],
        "models": list(models),
        "origin_count": len(origins),
        "origin_cadence_hours": int(suite["rolling"]["origin_cadence_hours"]),
        "origin_cadence_override": args.origin_cadence_hours,
        "first_origin": origins[0].isoformat(),
        "last_origin": origins[-1].isoformat(),
        "suite_path": str(suite_path.relative_to(REPOSITORY_ROOT)),
        "suite_sha256": sha256_file(suite_path),
        "protocol_path": str(protocol_path.relative_to(REPOSITORY_ROOT)),
        "protocol_sha256": sha256_file(protocol_path),
        "source_snapshot": str(source_path.relative_to(REPOSITORY_ROOT)),
        "source_snapshot_sha256": source.source_sha256,
        "source_metadata": source.source_metadata,
        "capacity": source.capacity,
        "timestamp_contract": {
            "recorder_anchor": source.recorder_anchor.isoformat(),
            "calendar_anchor": str(protocol["calendar_anchor"]),
            "model_calendar_timestamp": "calendar_anchor + simulation_seconds",
            "data_origin_timestamp": "recorder_anchor + simulation_seconds",
        },
        "git": _git_state(),
        "implementation_sha256": _implementation_hashes(),
        "environment_lock_path": str(ENVIRONMENT_LOCK.relative_to(REPOSITORY_ROOT)),
        "environment_lock_sha256": sha256_file(ENVIRONMENT_LOCK),
        "runtime": _runtime_manifest(selected_devices),
        "resource_preflight": {
            "output_free_gib": free_gib,
            "gpu_free_gib": gpu_free_gib,
            "minimum_output_free_gib": minimum_free,
            "minimum_gpu_free_gib": float(suite["resources"].get("minimum_gpu_free_gib", 0.0)),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_locked": True,
    }
    if not resuming:
        _write_json_atomic(batch_dir / "manifest.json", manifest)
        _write_json_atomic(
            output_root / "latest.json",
            {
                "schema_version": "1.0",
                "batch_dir": str(batch_dir),
                "mode": mode,
                "created_at": manifest["created_at"],
            },
        )
        (batch_dir / "config.yaml").write_text(
            suite_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
    results: list[dict[str, Any]] = []
    run_plan = [
        (model_id, seed)
        for model_id in models
        for seed in _model_seeds(suite, model_id, mode=mode)
    ]
    batch_status_path = batch_dir / "status.json"

    def write_batch_status(state: str, completed_runs: int, error: str | None = None) -> None:
        payload = {
            "schema_version": "1.0",
            "state": state,
            "benchmark_id": suite["benchmark_id"],
            "mode": mode,
            "batch_dir": str(batch_dir),
            "heartbeat_at": datetime.now(timezone.utc).isoformat(),
            "completed_runs": int(completed_runs),
            "total_runs": len(run_plan),
            "progress_fraction": completed_runs / len(run_plan),
        }
        if error is not None:
            payload["error"] = error
        _write_json_atomic(batch_status_path, payload)

    completed_before_start = 0
    for model_id, seed in run_plan:
        metrics_path = batch_dir / model_id / f"seed{seed}" / "metrics.json"
        status_path = metrics_path.parent / "status.json"
        if metrics_path.exists() and status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("state") == "COMPLETED":
                results.append(json.loads(metrics_path.read_text(encoding="utf-8")))
                completed_before_start += 1
    write_batch_status("RUNNING", completed_before_start)
    lock_path = _acquire_batch_lock(batch_dir)
    try:
        completed_runs = completed_before_start
        for model_id, seed in run_plan:
            device = _select_device(suite, args.device, model_id)
            run_dir = batch_dir / model_id / f"seed{seed}"
            if (run_dir / "metrics.json").exists() and (run_dir / "status.json").exists():
                status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
                if status.get("state") == "COMPLETED":
                    continue
            result = run_one(
                suite,
                model_id=model_id,
                seed=seed,
                origins=origins,
                source=source,
                calendar_anchor=pd.Timestamp(protocol["calendar_anchor"]),
                device=device,
                run_dir=run_dir,
                smoke_hyperparameters=smoke_hyperparameters,
                score_from_start=quick_mode,
                resume=run_dir.exists(),
            )
            results.append(result)
            completed_runs += 1
            write_batch_status("RUNNING", completed_runs)
        summary = _aggregate_batch(results)
        summary.update(
            {
                "benchmark_id": suite["benchmark_id"],
                "mode": mode,
                "batch_dir": str(batch_dir),
                "test_locked": True,
            }
        )
        _write_json_atomic(batch_dir / "summary.json", summary)
        write_batch_status("AUDITING", len(run_plan))
        audit_batch(batch_dir)
        write_batch_status("COMPLETED", len(run_plan))
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"BATCH_DIR={batch_dir}")
    except Exception as exc:
        write_batch_status("FAILED", len(results), f"{type(exc).__name__}: {exc}")
        raise
    finally:
        lock_path.unlink(missing_ok=True)
    return batch_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        default=str(REPOSITORY_ROOT / "benchmark" / "forecast_suite.yaml"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument(
        "--models",
        default="all",
        help="Comma separated model IDs or all.",
    )
    parser.add_argument("--run-dir", default=None)
    parser.add_argument(
        "--origin-cadence-hours",
        type=int,
        default=None,
        help="Audited rolling origin cadence override for a new batch.",
    )
    parser.add_argument(
        "command", choices=("smoke", "preflight", "pilot", "run", "resume", "audit", "status")
    )
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
