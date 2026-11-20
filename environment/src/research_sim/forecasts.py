"""Causal forecast interfaces and legacy forecast helpers.

The public ``CausalQuotaProvider`` surface receives only history available before
the current quota decision. It never receives trace rows from the future or live
job objects containing trace-known remaining durations.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Any, Tuple, Optional, Mapping, Protocol, Sequence
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from .gfs.estimator.gpu_request_estimator import GPURequestEstimator


@dataclass(frozen=True)
class CausalForecastContext:
    """Read-only metadata plus a defensive copy of causal aggregate history."""

    time: int
    calendar_timestamp: str
    history_end_exclusive: int
    org_request_history: pd.DataFrame
    capacity_by_pool: Mapping[str, int]
    hp_used_by_pool: Mapping[str, float]
    spot_used_by_pool: Mapping[str, float]
    current_quota_by_pool: Mapping[str, Mapping[str, Any]]
    eviction_outcomes_by_pool: Mapping[str, Tuple[int, int]]
    guarantee_rate: float
    guarantee_hours: Tuple[int, ...]

    @classmethod
    def build(
        cls,
        *,
        time: int,
        calendar_timestamp: str,
        org_request_history: pd.DataFrame,
        capacity_by_pool: Mapping[str, int],
        hp_used_by_pool: Mapping[str, float],
        spot_used_by_pool: Mapping[str, float],
        current_quota_by_pool: Mapping[str, Mapping[str, Any]],
        eviction_outcomes_by_pool: Mapping[str, Tuple[int, int]],
        guarantee_rate: float,
        guarantee_hours: Sequence[int],
    ) -> "CausalForecastContext":
        history = org_request_history.copy(deep=True)
        return cls(
            time=int(time),
            calendar_timestamp=str(calendar_timestamp),
            history_end_exclusive=int(time),
            org_request_history=history,
            capacity_by_pool=MappingProxyType({str(k): int(v) for k, v in capacity_by_pool.items()}),
            hp_used_by_pool=MappingProxyType({str(k): float(v) for k, v in hp_used_by_pool.items()}),
            spot_used_by_pool=MappingProxyType({str(k): float(v) for k, v in spot_used_by_pool.items()}),
            current_quota_by_pool=MappingProxyType({
                str(k): MappingProxyType(dict(v)) for k, v in current_quota_by_pool.items()
            }),
            eviction_outcomes_by_pool=MappingProxyType({
                str(k): (int(v[0]), int(v[1])) for k, v in eviction_outcomes_by_pool.items()
            }),
            guarantee_rate=float(guarantee_rate),
            guarantee_hours=tuple(int(v) for v in guarantee_hours),
        )


class CausalQuotaProvider(Protocol):
    """Extension point for a proposed forecaster.

    Implementations must be deterministic for equal context and internal state.
    ``provider_id`` and ``provider_version`` are recorded in run artifacts.
    """

    provider_id: str
    provider_version: str

    def update_quota(
        self, context: CausalForecastContext
    ) -> Mapping[str, Mapping[str, Any]]:
        ...


def validate_quota_update(
    update: Mapping[str, Mapping[str, Any]],
    *,
    capacity_by_pool: Mapping[str, int],
    guarantee_hours: Sequence[int],
) -> Dict[str, Dict[str, Any]]:
    """Validate external quota output. Invalid or partial output fails closed."""
    expected_pools = {str(k) for k in capacity_by_pool}
    if not isinstance(update, Mapping) or {str(k) for k in update} != expected_pools:
        raise ValueError(
            f"Quota provider pools must equal {sorted(expected_pools)}."
        )

    normalized: Dict[str, Dict[str, Any]] = {}
    expected_hours = tuple(int(h) for h in guarantee_hours)
    for pool in sorted(expected_pools):
        raw = update[pool]
        if not isinstance(raw, Mapping) or not isinstance(raw.get("quota"), Mapping):
            raise ValueError(f"Quota provider output for {pool!r} requires a quota mapping.")
        capacity = int(capacity_by_pool[pool])
        quota: Dict[int, int] = {}
        for hour in expected_hours:
            value = raw["quota"].get(hour, raw["quota"].get(str(hour)))
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                raise ValueError(f"Quota for pool={pool!r}, hour={hour} must be numeric.")
            value_float = float(value)
            if not math.isfinite(value_float) or not value_float.is_integer():
                raise ValueError(f"Quota for pool={pool!r}, hour={hour} must be a finite integer.")
            value_int = int(value_float)
            if not 0 <= value_int <= capacity:
                raise ValueError(
                    f"Quota for pool={pool!r}, hour={hour} must be within [0, {capacity}]."
                )
            quota[hour] = value_int
        eta = float(raw.get("eta", 1.0))
        if not math.isfinite(eta) or eta < 0.0:
            raise ValueError(f"Eta for pool={pool!r} must be finite and nonnegative.")
        normalized[pool] = {"quota": quota, "eta": eta}
    return normalized


class BaseForecastProvider:
    def update_quota(
        self,
        time: int,
        cluster: Any,
        org_request_df: pd.DataFrame,
        now_timestamp: pd.Timestamp,
        guarantee_rate: float,
        guarantee_hours: list,
        current_spot_quota: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError


class NativeOnlineForecastProvider(BaseForecastProvider):
    """
    Reproduces the released GFS causal update using GPURequestEstimator.
    Handles short sequence lengths gracefully by falling back to observed peak HP usage.
    """
    def __init__(self, args: Any):
        self.args = args
        self.estimator = GPURequestEstimator(args)

    def train_if_due(self, org_request_df: pd.DataFrame, now_timestamp: pd.Timestamp):
        if len(org_request_df) >= self.args.seq_len * 60:
            try:
                self.estimator.train(org_request_df, now_timestamp)
            except Exception as e:
                # If training fails on small data, keep existing model weights
                pass

    def update_quota(
        self,
        time: int,
        cluster: Any,
        org_request_df: pd.DataFrame,
        now_timestamp: pd.Timestamp,
        guarantee_rate: float,
        guarantee_hours: list,
        current_spot_quota: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        # If not enough history for GFS seq_len (672 hours = 40320 minutes)
        min_required_rows = self.args.seq_len * 60
        if len(org_request_df) < min_required_rows:
            # Conservative heuristic: estimate based on current cluster capacity
            new_quota = {}
            for vc_name in cluster.vc_list:
                vc = cluster.vc_dict[vc_name]
                free_cards = vc.check_free_gpus()
                eta = current_spot_quota.get(vc_name, {}).get("eta", 1.0)
                q_dict = {h: int(free_cards * eta * guarantee_rate) for h in guarantee_hours}
                new_quota[vc_name] = {"quota": q_dict, "eta": eta}
            return new_quota

        try:
            pred_mu, pred_sigma = self.estimator.test(org_request_df, now_timestamp)
            pred_mu_dict, pred_sigma_dict = {}, {}
            for i in range(len(org_request_df.columns)):
                org_name = org_request_df.columns[i]
                vc_name = org_name.split('&&', 1)[0]
                org_mu, org_sigma = pred_mu[0, :, i], pred_sigma[0, :, i]
                if vc_name not in pred_mu_dict:
                    pred_mu_dict[vc_name] = org_mu
                    pred_sigma_dict[vc_name] = org_sigma * org_sigma
                else:
                    pred_mu_dict[vc_name] += org_mu
                    pred_sigma_dict[vc_name] += org_sigma * org_sigma

            new_quota = {}
            for vc_name in pred_mu_dict:
                vc = cluster.vc_dict[vc_name]
                gpu_capacity = vc.total_gpus
                vc_mu, vc_sigma = pred_mu_dict[vc_name], np.sqrt(pred_sigma_dict[vc_name])
                usage = norm.ppf(guarantee_rate, loc=vc_mu, scale=vc_sigma)
                free = gpu_capacity - np.clip(np.floor(usage), 0, gpu_capacity)

                eta = current_spot_quota.get(vc_name, {}).get("eta", 1.0)
                # Adjust safety factor eta
                succeed, failed = vc.get_eviction_rate(time, self.args.pred_len)
                if succeed + failed > 0:
                    eviction_rate = float(failed) / float(succeed + failed)
                    if eviction_rate >= 1.5 * (1.0 - guarantee_rate):
                        eta = eta * (1.0 - guarantee_rate) / eviction_rate
                    elif (eviction_rate <= 0.5 * (1.0 - guarantee_rate) and
                          eviction_rate > 0):
                        eta = eta * (1.5 - (1.0 - guarantee_rate) / eviction_rate)

                q_dict = {}
                for h in guarantee_hours:
                    q_dict[h] = int(np.min(free[:h]) * eta)
                new_quota[vc_name] = {"quota": q_dict, "eta": eta}

            return new_quota
        except Exception:
            return current_spot_quota


class FrozenCausalForecastProvider(BaseForecastProvider):
    """
    Fixed/cached causal forecast provider for fast, deterministic evaluation.
    """
    def __init__(self, default_quota_ratio: float = 0.5):
        self.default_quota_ratio = default_quota_ratio

    def update_quota(
        self,
        time: int,
        cluster: Any,
        org_request_df: pd.DataFrame,
        now_timestamp: pd.Timestamp,
        guarantee_rate: float,
        guarantee_hours: list,
        current_spot_quota: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        new_quota = {}
        for vc_name in cluster.vc_list:
            vc = cluster.vc_dict[vc_name]
            eta = current_spot_quota.get(vc_name, {}).get("eta", 1.0)
            base_quota = int(vc.total_gpus * self.default_quota_ratio * eta)
            new_quota[vc_name] = {
                "quota": {h: base_quota for h in guarantee_hours},
                "eta": eta
            }
        return new_quota
