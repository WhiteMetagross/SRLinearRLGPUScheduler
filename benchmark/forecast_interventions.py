"""Score window forecast interventions for non RL GFS experiments."""

from __future__ import annotations

import hashlib
import json
import copy
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm
import torch
import yaml

from research_sim.forecasts import CausalForecastContext, validate_quota_update


def _framework_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_framework_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SRLinearOnlineQuotaProvider:
    """Causal hourly SRLinear refits with five minute quota service."""

    provider_version = "score_window_hourly_v1"

    def __init__(
        self,
        *,
        model: Any,
        mode: str,
        run_dir: Path,
        seed: int,
        calibration: Mapping[str, Any],
        native_estimator: Any | None = None,
        native_simulator: Any | None = None,
        native_safe_margin_fraction: float | None = None,
        cache_dir: Path | None = None,
        cache_namespace: str | None = None,
    ) -> None:
        if mode not in {"mean", "mean_native_sigma", "residual_gaussian", "conformal"}:
            raise ValueError(f"Unsupported SRLinear quota mode {mode!r}.")
        if native_safe_margin_fraction is not None:
            if native_estimator is None:
                raise ValueError("Native safe quota exposure requires the native estimator.")
            if not 0.0 <= float(native_safe_margin_fraction) <= 1.0:
                raise ValueError("native_safe_margin_fraction must lie within [0, 1].")
        self.provider_id = (
            f"srlinear_{mode}_with_native_safe"
            if native_safe_margin_fraction is not None
            else f"srlinear_{mode}"
        )
        self.model = model
        self.mode = mode
        self.run_dir = run_dir
        self.seed = int(seed)
        self.calibration = dict(calibration)
        self.native_estimator = native_estimator
        self.native_simulator = native_simulator
        self.native_safe_margin_fraction = (
            None
            if native_safe_margin_fraction is None
            else float(native_safe_margin_fraction)
        )
        self._bucket: int | None = None
        self._point: np.ndarray | None = None
        self._upper: np.ndarray | None = None
        self._fit_count = 0
        self._refresh_count = 0
        self._quota_count = 0
        self._eta_by_pool: dict[str, float] = {}
        self.safe_quota_by_pool: dict[str, dict[int, int]] = {}
        self.srlinear_quota_by_pool: dict[str, dict[int, int]] = {}
        self.native_quota_by_pool: dict[str, dict[int, int]] = {}
        self.uncertainty_by_pool: dict[str, float] = {}
        self._native_eta_by_pool: dict[str, float] = {}
        self._primed_native_quota: dict[str, dict[str, Any]] | None = None
        self._native_training_count = 0
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.cache_namespace = cache_namespace
        if (self.cache_dir is None) != (self.cache_namespace is None):
            raise ValueError("Forecast cache directory and namespace must be declared together.")
        self._cache_hits = 0
        self._cache_misses = 0

    @staticmethod
    def _history_sha256(history: pd.DataFrame) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps([str(value) for value in history.columns]).encode("utf-8"))
        digest.update(pd.util.hash_pandas_object(history, index=True).to_numpy().tobytes())
        return digest.hexdigest()

    def _cached_point(
        self,
        history: pd.DataFrame,
        timestamp: pd.Timestamp,
    ) -> np.ndarray | None:
        if self.cache_dir is None or self.cache_namespace is None:
            return None
        history_sha256 = self._history_sha256(history)
        path = (
            self.cache_dir
            / self.cache_namespace
            / f"seed{self.seed}"
            / timestamp.strftime("%Y%m%dT%H%M%S")
            / f"{history_sha256}.npz"
        )
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"].item()))
            point = np.asarray(payload["point"], dtype=np.float64)
        expected = {
            "namespace": self.cache_namespace,
            "seed": self.seed,
            "timestamp": timestamp.isoformat(),
            "history_sha256": history_sha256,
        }
        if metadata != expected:
            raise ValueError(f"SRLinear cache identity mismatch: {path}")
        self._cache_hits += 1
        return point

    def _store_cached_point(
        self,
        history: pd.DataFrame,
        timestamp: pd.Timestamp,
        point: np.ndarray,
    ) -> None:
        if self.cache_dir is None or self.cache_namespace is None:
            return
        history_sha256 = self._history_sha256(history)
        path = (
            self.cache_dir
            / self.cache_namespace
            / f"seed{self.seed}"
            / timestamp.strftime("%Y%m%dT%H%M%S")
            / f"{history_sha256}.npz"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "namespace": self.cache_namespace,
            "seed": self.seed,
            "timestamp": timestamp.isoformat(),
            "history_sha256": history_sha256,
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    point=np.asarray(point, dtype=np.float64),
                    metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                )
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _native_pool_sigma(estimator: Any, history: pd.DataFrame, timestamp: pd.Timestamp) -> np.ndarray:
        pred_mu, pred_sigma = estimator.test(history, timestamp)
        del pred_mu
        variance: np.ndarray | None = None
        for index in range(len(history.columns)):
            sigma = np.asarray(pred_sigma[0, :, index], dtype=np.float64)
            variance = sigma * sigma if variance is None else variance + sigma * sigma
        if variance is None:
            raise ValueError("Native estimator returned no organization uncertainty.")
        return np.sqrt(variance)

    @staticmethod
    def _native_pool_forecast(
        estimator: Any,
        history: pd.DataFrame,
        timestamp: pd.Timestamp,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate released OrgLinear means and independent variances."""

        pred_mu, pred_sigma = estimator.test(history, timestamp)
        mean: np.ndarray | None = None
        variance: np.ndarray | None = None
        for index in range(len(history.columns)):
            org_mean = np.asarray(pred_mu[0, :, index], dtype=np.float64)
            org_sigma = np.asarray(pred_sigma[0, :, index], dtype=np.float64)
            mean = org_mean.copy() if mean is None else mean + org_mean
            variance = (
                org_sigma * org_sigma
                if variance is None
                else variance + org_sigma * org_sigma
            )
        if mean is None or variance is None:
            raise ValueError("Native estimator returned no organization forecast.")
        return mean, np.sqrt(variance)

    def _native_safe_bound(self, context: Any, timestamp: pd.Timestamp) -> np.ndarray:
        if self.native_estimator is None or self.native_safe_margin_fraction is None:
            raise RuntimeError("Native safe quota endpoint was not enabled.")
        cache = (
            None
            if self.native_simulator is None
            else getattr(self.native_simulator, "forecast_cache", None)
        )
        if cache is not None and int(context.time) in cache:
            pred_mu_dict, pred_sigma_dict = cache[int(context.time)]
            pools = tuple(context.capacity_by_pool)
            if len(pools) != 1 or pools[0] not in pred_mu_dict:
                raise ValueError("Native forecast cache does not match the SRAC pool.")
            mean = np.asarray(pred_mu_dict[pools[0]], dtype=np.float64)
            sigma = np.sqrt(np.asarray(pred_sigma_dict[pools[0]], dtype=np.float64))
            return np.clip(
                self._gaussian_upper(
                    mean, sigma, float(context.guarantee_rate)
                ),
                0.0,
                None,
            )
        history = context.org_request_history
        if (
            int(context.time) % 86400 == 0
            or self.native_estimator.requires_schema_refresh(history)
        ):
            self.native_estimator.train(history, timestamp)
            self._native_training_count += 1
        point, sigma = self._native_pool_forecast(
            self.native_estimator, history, timestamp
        )
        return np.clip(
            self._gaussian_upper(point, sigma, float(context.guarantee_rate)),
            0.0,
            None,
        )

    @staticmethod
    def _gaussian_upper(point: np.ndarray, sigma: np.ndarray, probability: float) -> np.ndarray:
        point = np.asarray(point, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        if point.shape != sigma.shape:
            raise ValueError("Gaussian point and sigma shapes differ.")
        upper = point.copy()
        positive = sigma > 1e-8
        if np.any(positive):
            upper[positive] = norm.ppf(
                float(probability), loc=point[positive], scale=sigma[positive]
            )
        return upper

    def prime_native_quota(self, current_quota: Mapping[str, Any]) -> None:
        """Preserve the Native GFS quota visible at the first scored decision."""

        if self.native_safe_margin_fraction is None:
            return
        if self._quota_count != 0 or self._primed_native_quota is not None:
            raise ValueError("Native quota endpoint can be primed only once before activation.")
        self._primed_native_quota = copy.deepcopy(dict(current_quota))

    def _refresh(self, context: Any, timestamp: pd.Timestamp) -> None:
        history = context.org_request_history
        point = self._cached_point(history, timestamp)
        if point is None:
            rng_state = _framework_rng_state()
            try:
                output = self.model.fit_predict(
                    history,
                    timestamp,
                    self.run_dir / "fits",
                    calendar_timestamp=timestamp,
                )
            finally:
                _restore_framework_rng_state(rng_state)
            point = np.asarray(output.point, dtype=np.float64).sum(axis=1)
            self._store_cached_point(history, timestamp, point)
            self._cache_misses += 1
            self._fit_count += 1
        if not np.all(np.isfinite(point)) or np.any(point < 0.0):
            raise ValueError("SRLinear produced an invalid aggregate point forecast.")
        if self.mode == "mean":
            upper = point.copy()
        elif self.mode == "mean_native_sigma":
            if self.native_estimator is None:
                raise ValueError("Native sigma mode requires the snapshot estimator.")
            if (
                int(context.time) % 86400 == 0
                or self.native_estimator.requires_schema_refresh(history)
            ):
                self.native_estimator.train(history, timestamp)
            sigma = self._native_pool_sigma(self.native_estimator, history, timestamp)
            upper = self._gaussian_upper(point, sigma, float(context.guarantee_rate))
        elif self.mode == "residual_gaussian":
            sigma = np.asarray(
                self.calibration["residual_sigma_by_horizon"], dtype=np.float64
            )
            if sigma.shape != point.shape:
                raise ValueError("SRLinear residual sigma does not match forecast horizon.")
            upper = self._gaussian_upper(point, sigma, float(context.guarantee_rate))
        else:
            probability = float(self.calibration.get("quota_quantile", 0.98))
            offsets = self.calibration["conformal_offsets_by_quantile"]
            key = min(offsets, key=lambda value: abs(float(value) - probability))
            offset = np.asarray(offsets[key], dtype=np.float64)
            if offset.shape != point.shape:
                raise ValueError("SRLinear conformal offset does not match forecast horizon.")
            upper = point + offset
        self._point = np.clip(point, 0.0, None)
        self._upper = np.clip(np.asarray(upper, dtype=np.float64), 0.0, None)
        self._refresh_count += 1

    def update_quota(self, context: Any) -> dict[str, dict[str, Any]]:
        timestamp = pd.Timestamp(context.calendar_timestamp)
        bucket = int(context.time) // 3600
        if self._bucket != bucket:
            self._refresh(context, timestamp)
            self._bucket = bucket
        if self._upper is None:
            raise RuntimeError("SRLinear provider has no forecast after refresh.")
        minute = int(timestamp.minute)
        horizon_count = min(len(self._upper), 1 if minute == 0 else 2)
        demand_bound = float(np.max(self._upper[:horizon_count]))
        primed_native = self._primed_native_quota
        self._primed_native_quota = None
        native_bound = (
            self._native_safe_bound(context, timestamp)
            if self.native_safe_margin_fraction is not None and primed_native is None
            else None
        )
        residual_sigma = np.asarray(
            self.calibration.get("residual_sigma_by_horizon", ()), dtype=np.float64
        )
        if (native_bound is not None or primed_native is not None) and (
            len(residual_sigma) < horizon_count
            or np.any(~np.isfinite(residual_sigma))
            or np.any(residual_sigma < 0.0)
        ):
            raise ValueError("Native endpoint exposure requires valid SRLinear uncertainty.")
        update: dict[str, dict[str, Any]] = {}
        for pool, capacity_value in context.capacity_by_pool.items():
            capacity = int(capacity_value)
            raw_free = float(np.clip(capacity - np.floor(demand_bound), 0, capacity))
            pool_key = str(pool)
            eta = self._eta_by_pool.get(
                pool_key,
                float(context.current_quota_by_pool[pool].get("eta", 1.0)),
            )
            succeeded, failed = context.eviction_outcomes_by_pool[pool]
            if succeeded + failed > 0:
                eviction_rate = float(failed) / float(succeeded + failed)
                threshold = 1.5 * (1.0 - float(context.guarantee_rate))
                if eviction_rate >= threshold and eviction_rate > 0.0:
                    eta *= (1.0 - float(context.guarantee_rate)) / eviction_rate
            eta = float(np.clip(eta, 0.0, 1.0))
            self._eta_by_pool[pool_key] = eta
            free = int(np.clip(np.floor(raw_free * eta), 0, capacity))
            srlinear_quota = {int(hour): free for hour in context.guarantee_hours}
            self.srlinear_quota_by_pool[pool_key] = srlinear_quota
            if native_bound is not None or primed_native is not None:
                if primed_native is not None:
                    raw_primed = primed_native[pool]
                    native_eta = float(raw_primed.get("eta", 1.0))
                    native_quota = {
                        int(hour): float(raw_primed["quota"][int(hour)])
                        for hour in context.guarantee_hours
                    }
                else:
                    native_eta = self._native_eta_by_pool.get(
                        pool_key,
                        float(context.current_quota_by_pool[pool].get("eta", 1.0)),
                    )
                    if succeeded + failed > 0:
                        eviction_rate = float(failed) / float(succeeded + failed)
                        lower = 0.5 * (1.0 - float(context.guarantee_rate))
                        upper = 1.5 * (1.0 - float(context.guarantee_rate))
                        if eviction_rate >= upper and eviction_rate > 0.0:
                            native_eta *= (
                                1.0 - float(context.guarantee_rate)
                            ) / eviction_rate
                        elif (
                            0.0 < eviction_rate <= lower
                            and self.native_simulator is not None
                            and len(self.native_simulator.spot_que_list[pool]) > 0
                            and int(context.time)
                            - self.native_simulator.spot_que_list[pool][0].get_preempt_time()
                            > 3600
                        ):
                            native_eta *= (
                                1.5
                                - (1.0 - float(context.guarantee_rate)) / eviction_rate
                            )
                    native_eta = float(max(0.0, native_eta))
                    native_free_by_horizon = np.clip(
                        capacity - np.floor(native_bound), 0, capacity
                    )
                    native_quota = {
                        int(hour): float(
                            np.min(native_free_by_horizon[: int(hour)]) * native_eta
                        )
                        for hour in context.guarantee_hours
                    }
                self._native_eta_by_pool[pool_key] = native_eta
                self.native_quota_by_pool[pool_key] = native_quota
                self.uncertainty_by_pool[pool_key] = float(
                    np.max(residual_sigma[:horizon_count])
                )
                native_free = float(min(native_quota.values()))
                hp_headroom_cap = float(np.clip(
                    capacity
                    - float(context.hp_used_by_pool[pool])
                    - self.native_safe_margin_fraction * capacity,
                    0,
                    capacity,
                ))
                safe_free = int(np.clip(
                    np.floor(min(native_free, hp_headroom_cap)), 0, capacity
                ))
                self.safe_quota_by_pool[pool_key] = {
                    int(hour): safe_free for hour in context.guarantee_hours
                }
            update[str(pool)] = {
                "quota": srlinear_quota,
                "eta": eta,
            }
        self._quota_count += 1
        return update

    def manifest(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "mode": self.mode,
            "seed": self.seed,
            "refresh_seconds": 3600,
            "fit_count": self._fit_count,
            "refresh_count": self._refresh_count,
            "quota_count": self._quota_count,
            "calibration_source_sha256": self.calibration.get("source_sha256"),
            "warmup_provider": "native_gfs",
            "activation": "score_start_before_first_control_action",
            "eta_update": "native_overrisk_decrease_only_v1",
            "native_safe_endpoint": self.native_safe_margin_fraction is not None,
            "native_quota_endpoint": self.native_safe_margin_fraction is not None,
            "uncertainty_endpoint": self.native_safe_margin_fraction is not None,
            "native_safe_margin_fraction": self.native_safe_margin_fraction,
            "native_training_count": self._native_training_count,
            "safe_endpoint_rule": (
                "minimum_of_native_orglinear_quota_and_current_hp_headroom_v1"
                if self.native_safe_margin_fraction is not None
                else None
            ),
            "forecast_cache_enabled": self.cache_dir is not None,
            "forecast_cache_namespace": self.cache_namespace,
            "forecast_cache_hits": self._cache_hits,
            "forecast_cache_misses": self._cache_misses,
        }


def build_forecast_intervention(
    declaration: Mapping[str, Any],
    *,
    suite_base: Path,
    simulator: Any,
    run_dir: Path,
    seed: int,
    smoke: bool,
) -> SRLinearOnlineQuotaProvider:
    if declaration.get("type") != "srlinear_online":
        raise ValueError(f"Unsupported forecast intervention {declaration.get('type')!r}.")
    forecast_suite_path = Path(str(declaration["forecast_suite"]))
    if not forecast_suite_path.is_absolute():
        forecast_suite_path = (suite_base / forecast_suite_path).resolve()
    forecast_suite = yaml.safe_load(forecast_suite_path.read_text(encoding="utf-8"))
    calibration_path = Path(str(declaration["calibration_artifact"]))
    if not calibration_path.is_absolute():
        calibration_path = (suite_base / calibration_path).resolve()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("test_outcomes_accessed") is not False:
        raise ValueError("Forecast calibration must prove test_outcomes_accessed=false.")
    if not smoke and int(calibration.get("origin_cadence_hours", 24)) != 1:
        raise ValueError(
            "Complete hourly deployment requires a one hour origin calibration artifact."
        )
    calibration["source_sha256"] = _sha256(calibration_path)
    if "quota_quantile" in declaration:
        calibration["quota_quantile"] = float(declaration["quota_quantile"])
    from benchmark.forecast_run import _build_model

    device_name = str(declaration.get("device", "auto"))
    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = _build_model(
        forecast_suite,
        model_id="srlinear",
        seed=int(seed),
        device=device,
        run_dir=run_dir / "forecast_provider",
        smoke=smoke,
    )
    cache_dir_value = declaration.get("cache_dir")
    cache_dir = None if cache_dir_value is None else Path(str(cache_dir_value)).resolve()
    cache_namespace = None
    if cache_dir is not None:
        model_config = {
            key: value
            for key, value in vars(model).items()
            if isinstance(value, (str, int, float, bool, tuple, list))
        }
        source_root = Path(__file__).parents[1] / "methods" / "forecasters"
        source_digest = hashlib.sha256()
        for source_path in (
            source_root / "srlinear.py",
            source_root / "srlinear_components.py",
        ):
            source_digest.update(source_path.read_bytes())
        namespace_payload = {
            "cache_schema": "causal_history_keyed_v2",
            "model_id": "srlinear",
            "smoke": bool(smoke),
            "forecast_suite_sha256": _sha256(forecast_suite_path),
            "model_source_sha256": source_digest.hexdigest(),
            "model_config": model_config,
        }
        cache_namespace = hashlib.sha256(
            json.dumps(namespace_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:24]
    mode = str(declaration["mode"])
    native_safe_margin = declaration.get("native_safe_margin_fraction")
    native_safe_artifact = declaration.get("native_safe_artifact")
    if native_safe_artifact is not None:
        if native_safe_margin is not None:
            raise ValueError(
                "Declare native_safe_artifact or native_safe_margin_fraction, not both."
            )
        safe_path = Path(str(native_safe_artifact))
        if not safe_path.is_absolute():
            safe_path = (suite_base / safe_path).resolve()
        safe_payload = json.loads(safe_path.read_text(encoding="utf-8"))
        if safe_payload.get("test_outcomes_accessed") is not False:
            raise ValueError("Native safe artifact must prove test_outcomes_accessed=false.")
        if not smoke and safe_payload.get("qualification") != "publication":
            raise ValueError("Complete runs require a publication qualified safe artifact.")
        native_safe_margin = safe_payload["native_safe_margin_fraction"]
    needs_native = mode == "mean_native_sigma" or native_safe_margin is not None
    return SRLinearOnlineQuotaProvider(
        model=model,
        mode=mode,
        run_dir=run_dir / "forecast_provider",
        seed=int(seed),
        calibration=calibration,
        native_estimator=(simulator.estimator if needs_native else None),
        native_simulator=(simulator if needs_native else None),
        native_safe_margin_fraction=(
            None if native_safe_margin is None else float(native_safe_margin)
        ),
        cache_dir=cache_dir,
        cache_namespace=cache_namespace,
    )


def activate_forecast_at_score_boundary(simulator: Any, provider: Any) -> Any:
    """Replace the restored native quota before the first scored control action."""

    if int(simulator.time) != int(simulator.spec.score_start):
        raise ValueError("Forecast intervention must activate exactly at score_start.")
    timestamp = pd.Timestamp(simulator.spec.calendar_anchor) + pd.Timedelta(
        seconds=int(simulator.time)
    )
    history = simulator.recorder.agg_seq_recorder()
    capacity_by_pool = {
        pool: int(simulator.cluster.vc_dict[pool].total_gpus)
        for pool in simulator.cluster.vc_list
    }
    hp_used_by_pool = {
        pool: float(sum(
            job["gpu_request"] * job["worker_num"]
            for job in simulator.run_list[pool] if job["type"] == "HP"
        ))
        for pool in simulator.cluster.vc_list
    }
    spot_used_by_pool = {
        pool: float(simulator.gpu_request_dict[pool]["Spot"])
        for pool in simulator.cluster.vc_list
    }
    eviction_outcomes = {
        pool: tuple(int(value) for value in simulator.cluster.vc_dict[pool].get_eviction_rate(
            simulator.time, simulator.args.pred_len
        ))
        for pool in simulator.cluster.vc_list
    }
    context = CausalForecastContext.build(
        time=simulator.time,
        calendar_timestamp=str(timestamp),
        org_request_history=history,
        capacity_by_pool=capacity_by_pool,
        hp_used_by_pool=hp_used_by_pool,
        spot_used_by_pool=spot_used_by_pool,
        current_quota_by_pool=copy.deepcopy(simulator.spot_quota),
        eviction_outcomes_by_pool=eviction_outcomes,
        guarantee_rate=simulator.args.guarantee_rate,
        guarantee_hours=simulator.args.guarantee_hour,
    )
    prime_native = getattr(provider, "prime_native_quota", None)
    if callable(prime_native):
        prime_native(copy.deepcopy(simulator.spot_quota))
    simulator.spot_quota = validate_quota_update(
        provider.update_quota(context),
        capacity_by_pool=capacity_by_pool,
        guarantee_hours=simulator.args.guarantee_hour,
    )
    simulator.quota_update_count += 1
    simulator.eligibility_epoch += 1
    simulator.ledger.record_event(
        simulator.time,
        "quota_update",
        details={
            "spot_quota": {
                key: value["quota"] for key, value in simulator.spot_quota.items()
            },
            "provider_id": provider.provider_id,
            "provider_version": provider.provider_version,
            "score_boundary_intervention": True,
        },
    )
    simulator.control_paused_context = simulator._build_control_context()
    return simulator.control_paused_context
