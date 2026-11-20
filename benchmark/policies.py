"""Supported deterministic policies for the unified A100 benchmark."""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np


ACTION_DIM = 7


class BaselinePolicy(Protocol):
    """Minimal causal policy interface used by the benchmark runner."""

    def action(self, context: Any) -> np.ndarray:
        ...

    def manifest(self) -> dict[str, Any]:
        ...


def _validate_action(action: np.ndarray) -> np.ndarray:
    value = np.asarray(action, dtype=np.float32)
    if value.shape != (ACTION_DIM,):
        raise ValueError(f"Baseline action must have shape {(ACTION_DIM,)}, got {value.shape}.")
    if not np.all(np.isfinite(value)) or np.any(np.abs(value) > 1.0):
        raise ValueError("Baseline action must be finite and bounded by [-1, 1].")
    return value


class FixedActionPolicy:
    """Return one immutable control action at every decision."""

    def __init__(self, fixed_action: np.ndarray) -> None:
        self.fixed_action = _validate_action(fixed_action).copy()

    def action(self, context: Any) -> np.ndarray:
        del context
        return self.fixed_action.copy()

    def manifest(self) -> dict[str, Any]:
        return {
            "policy_type": "fixed_action",
            "fixed_action": self.fixed_action.tolist(),
        }


class CurrentFreeHeadroomPolicy:
    """Cap Spot quota using current free capacity and a train fitted margin."""

    def __init__(
        self,
        *,
        margin_fraction: float,
        minimum_residual: float,
        maximum_residual: float,
        priority_weights: tuple[float, ...],
    ) -> None:
        if not 0.0 <= margin_fraction <= 1.0:
            raise ValueError("Headroom margin must lie within [0, 1].")
        self.margin_fraction = float(margin_fraction)
        self.minimum_residual = float(minimum_residual)
        self.maximum_residual = float(maximum_residual)
        self.priority_weights = tuple(float(value) for value in priority_weights)

    def action(self, context: Any) -> np.ndarray:
        _, capacity, hp_used, _, base = _single_pool_values(context)
        residual = (capacity - hp_used - base) / capacity - self.margin_fraction
        residual = float(np.clip(
            residual, self.minimum_residual, self.maximum_residual
        ))
        return _validate_action(np.asarray(
            (residual, *self.priority_weights), dtype=np.float32
        ))

    def manifest(self) -> dict[str, Any]:
        return {
            "policy_type": "current_free_headroom",
            "margin_fraction": self.margin_fraction,
            "minimum_residual": self.minimum_residual,
            "maximum_residual": self.maximum_residual,
            "priority_weights": list(self.priority_weights),
        }


class HourOfDayResidualPolicy:
    """Select one train fitted quota residual for each hour of day."""

    def __init__(
        self,
        *,
        residual_by_hour: tuple[float, ...],
        minimum_residual: float,
        maximum_residual: float,
        priority_weights: tuple[float, ...],
    ) -> None:
        if len(residual_by_hour) != 24:
            raise ValueError("Hour policy requires exactly 24 residuals.")
        values = np.asarray(residual_by_hour, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("Hour policy residuals must be finite.")
        self.residual_by_hour = tuple(float(value) for value in values)
        self.minimum_residual = float(minimum_residual)
        self.maximum_residual = float(maximum_residual)
        self.priority_weights = tuple(float(value) for value in priority_weights)
        _validate_action(np.asarray((0.0, *self.priority_weights), dtype=np.float32))

    def action(self, context: Any) -> np.ndarray:
        hour = (int(context.time) // 3600) % 24
        residual = float(np.clip(
            self.residual_by_hour[hour],
            self.minimum_residual,
            self.maximum_residual,
        ))
        return _validate_action(np.asarray(
            (residual, *self.priority_weights), dtype=np.float32
        ))

    def manifest(self) -> dict[str, Any]:
        return {
            "policy_type": "hour_of_day_residual",
            "residual_by_hour": list(self.residual_by_hour),
            "minimum_residual": self.minimum_residual,
            "maximum_residual": self.maximum_residual,
            "priority_weights": list(self.priority_weights),
        }


def _single_pool_values(context: Any) -> tuple[str, float, float, float, float]:
    pools = tuple(context.capacity_by_pool)
    if len(pools) != 1:
        raise ValueError("Quota controls require the single A100 pool benchmark.")
    pool = pools[0]
    capacity = float(context.capacity_by_pool[pool])
    if capacity <= 0.0:
        raise ValueError("Pool capacity must be positive.")
    base_by_horizon = context.base_quota_by_pool[pool]
    if not base_by_horizon:
        raise ValueError("Quota control requires at least one base quota horizon.")
    return (
        pool,
        capacity,
        float(context.hp_used_by_pool[pool]),
        float(context.spot_used_by_pool[pool]),
        float(min(base_by_horizon.values())),
    )


def _context_features(context: Any, *, base_override: float | None = None) -> np.ndarray:
    """Return the frozen causal feature vector used by linear quota control."""

    _, capacity, hp_used, spot_used, base = _single_pool_values(context)
    if base_override is not None:
        if not math.isfinite(float(base_override)):
            raise ValueError("Contextual quota base override must be finite.")
        base = float(np.clip(base_override, 0.0, capacity))
    phase_day = 2.0 * math.pi * ((int(context.time) % 86400) / 86400.0)
    phase_week = 2.0 * math.pi * ((int(context.time) % (7 * 86400)) / (7 * 86400.0))
    return np.asarray(
        (
            hp_used / capacity,
            base / capacity,
            spot_used / capacity,
            float(context.hp_queue_gpu_demand) / capacity,
            len(context.admission_jobs) / max(1.0, len(context.admission_jobs) + 64.0),
            math.sin(phase_day),
            math.cos(phase_day),
            math.sin(phase_week),
            math.cos(phase_week),
        ),
        dtype=np.float64,
    )


class LinearResidualPolicy:
    """Train fitted linear quota residual with fixed causal features."""

    def __init__(
        self,
        *,
        coefficients: tuple[float, ...],
        intercept: float,
        feature_mean: tuple[float, ...],
        feature_scale: tuple[float, ...],
        safety_offset: float,
        minimum_residual: float = -0.15,
        maximum_residual: float = 0.15,
        priority_weights: tuple[float, ...] = (0.0,) * 6,
    ) -> None:
        lengths = {len(coefficients), len(feature_mean), len(feature_scale)}
        if lengths != {9}:
            raise ValueError("Linear quota control requires nine frozen features.")
        values = np.asarray(
            (*coefficients, intercept, *feature_mean, *feature_scale, safety_offset),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)) or np.any(np.asarray(feature_scale) <= 0.0):
            raise ValueError("Linear quota artifact contains invalid values.")
        if not -1.0 <= minimum_residual <= maximum_residual <= 1.0:
            raise ValueError("Linear quota bounds are invalid.")
        if len(priority_weights) != ACTION_DIM - 1:
            raise ValueError("Priority weights must contain six values.")
        self.coefficients = tuple(float(value) for value in coefficients)
        self.intercept = float(intercept)
        self.feature_mean = tuple(float(value) for value in feature_mean)
        self.feature_scale = tuple(float(value) for value in feature_scale)
        self.safety_offset = float(safety_offset)
        self.minimum_residual = float(minimum_residual)
        self.maximum_residual = float(maximum_residual)
        self.priority_weights = tuple(float(value) for value in priority_weights)

    def residual_for_base(self, context: Any, base_quota: float) -> float:
        """Evaluate the frozen policy against an explicit causal quota base."""

        features = _context_features(context, base_override=base_quota)
        standardized = (
            features - np.asarray(self.feature_mean)
        ) / np.asarray(self.feature_scale)
        prediction = (
            float(np.dot(standardized, np.asarray(self.coefficients)))
            + self.intercept
            + self.safety_offset
        )
        return float(np.clip(prediction, self.minimum_residual, self.maximum_residual))

    def residual(self, context: Any) -> float:
        _, _, _, _, base = _single_pool_values(context)
        return self.residual_for_base(context, base)

    def action(self, context: Any) -> np.ndarray:
        return _validate_action(np.asarray(
            (self.residual(context), *self.priority_weights), dtype=np.float32
        ))

    def manifest(self) -> dict[str, Any]:
        return {
            "policy_type": "contextual_linear_residual",
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "safety_offset": self.safety_offset,
            "minimum_residual": self.minimum_residual,
            "maximum_residual": self.maximum_residual,
            "priority_weights": list(self.priority_weights),
        }


class ExpandingQuantileResidualPolicy:
    """Update a delayed causal residual quantile during evaluation."""

    def __init__(
        self,
        *,
        initial_residuals: tuple[float, ...],
        risk_limit: float,
        horizon_seconds: int,
        minimum_residual: float,
        maximum_residual: float,
        priority_weights: tuple[float, ...],
    ) -> None:
        if not initial_residuals or not 0.0 < risk_limit < 1.0:
            raise ValueError("Expanding quantile policy requires causal history and a valid risk limit.")
        self.initial_residuals = tuple(float(value) for value in initial_residuals)
        self.risk_limit = float(risk_limit)
        self.horizon_seconds = int(horizon_seconds)
        self.minimum_residual = float(minimum_residual)
        self.maximum_residual = float(maximum_residual)
        self.priority_weights = tuple(float(value) for value in priority_weights)
        self._residuals = list(self.initial_residuals)
        self._pending: deque[dict[str, float]] = deque()
        self._hp_history: deque[tuple[int, float]] = deque()
        _validate_action(np.asarray((0.0, *self.priority_weights), dtype=np.float32))

    def action(self, context: Any) -> np.ndarray:
        _, capacity, hp_used, _, base = _single_pool_values(context)
        now = int(context.time)
        self._hp_history.append((now, hp_used))
        while self._pending and self._pending[0]["time"] + self.horizon_seconds <= now:
            row = self._pending.popleft()
            peak = max(
                value
                for time_value, value in self._hp_history
                if row["time"] <= time_value <= row["time"] + self.horizon_seconds
            )
            self._residuals.append(
                (row["capacity"] - peak - row["base"]) / row["capacity"]
            )
        cutoff = now - self.horizon_seconds
        while self._hp_history and self._hp_history[0][0] < cutoff:
            self._hp_history.popleft()
        self._pending.append({"time": now, "capacity": capacity, "base": base})
        residual = float(np.quantile(
            np.asarray(self._residuals), self.risk_limit, method="lower"
        ))
        residual = float(np.clip(
            residual, self.minimum_residual, self.maximum_residual
        ))
        return _validate_action(np.asarray(
            (residual, *self.priority_weights), dtype=np.float32
        ))

    def manifest(self) -> dict[str, Any]:
        return {
            "policy_type": "expanding_quantile_residual",
            "initial_residual_count": len(self.initial_residuals),
            "risk_limit": self.risk_limit,
            "horizon_seconds": self.horizon_seconds,
            "minimum_residual": self.minimum_residual,
            "maximum_residual": self.maximum_residual,
            "priority_weights": list(self.priority_weights),
        }


class GuardedBaseQuotaPolicy:
    """Apply train fitted backlog and HP pressure guards to the active quota base."""

    def __init__(
        self,
        *,
        hp_rise_margin_fraction: float,
        backlog_bonus_fraction: float,
        emergency_hp_queue_fraction: float,
        minimum_residual: float,
        maximum_residual: float,
        priority_weights: tuple[float, ...],
    ) -> None:
        self.hp_rise_margin_fraction = float(hp_rise_margin_fraction)
        self.backlog_bonus_fraction = float(backlog_bonus_fraction)
        self.emergency_hp_queue_fraction = float(emergency_hp_queue_fraction)
        self.minimum_residual = float(minimum_residual)
        self.maximum_residual = float(maximum_residual)
        self.priority_weights = tuple(float(value) for value in priority_weights)

    def action(self, context: Any) -> np.ndarray:
        _, capacity, hp_used, _, base = _single_pool_values(context)
        target = base
        if context.admission_jobs and context.hp_queue_gpu_demand <= 0.0:
            target += self.backlog_bonus_fraction * capacity
        if context.hp_queue_gpu_demand / capacity >= self.emergency_hp_queue_fraction:
            target = min(
                target,
                capacity - hp_used - self.hp_rise_margin_fraction * capacity,
            )
        residual = float(np.clip(
            (np.clip(target, 0.0, capacity) - base) / capacity,
            self.minimum_residual,
            self.maximum_residual,
        ))
        return _validate_action(np.asarray(
            (residual, *self.priority_weights), dtype=np.float32
        ))

    def manifest(self) -> dict[str, Any]:
        return {
            "policy_type": "guarded_base_quota",
            "hp_rise_margin_fraction": self.hp_rise_margin_fraction,
            "backlog_bonus_fraction": self.backlog_bonus_fraction,
            "emergency_hp_queue_fraction": self.emergency_hp_queue_fraction,
            "minimum_residual": self.minimum_residual,
            "maximum_residual": self.maximum_residual,
            "priority_weights": list(self.priority_weights),
        }


def _linear_from_artifact(
    artifact: Mapping[str, Any], priority: tuple[float, ...]
) -> LinearResidualPolicy:
    return LinearResidualPolicy(
        coefficients=tuple(float(value) for value in artifact["coefficients"]),
        intercept=float(artifact["intercept"]),
        feature_mean=tuple(float(value) for value in artifact["feature_mean"]),
        feature_scale=tuple(float(value) for value in artifact["feature_scale"]),
        safety_offset=float(artifact["safety_offset"]),
        minimum_residual=float(artifact["minimum_residual"]),
        maximum_residual=float(artifact["maximum_residual"]),
        priority_weights=priority,
    )


class SRGFSContextualMixPolicy:
    """Mix the native safe quota and SRLinear quota with reactive protection."""

    policy_type = "srgfs_contextual_mix"

    def __init__(
        self,
        *,
        linear: LinearResidualPolicy,
        hp_rise_trigger_fraction: float,
        eviction_failure_trigger: float,
        cooldown_seconds: int,
        trigger_lost_gpu_hours: float,
        priority_weights: tuple[float, ...],
    ) -> None:
        if not 0.0 <= hp_rise_trigger_fraction <= 1.0:
            raise ValueError("HP rise trigger must lie within [0, 1].")
        if not 0.0 <= eviction_failure_trigger <= 1.0:
            raise ValueError("Eviction failure trigger must lie within [0, 1].")
        if cooldown_seconds <= 0 or trigger_lost_gpu_hours < 0.0:
            raise ValueError("Reactive protection parameters are invalid.")
        self.linear = linear
        self.hp_rise_trigger_fraction = float(hp_rise_trigger_fraction)
        self.eviction_failure_trigger = float(eviction_failure_trigger)
        self.cooldown_seconds = int(cooldown_seconds)
        self.trigger_lost_gpu_hours = float(trigger_lost_gpu_hours)
        self.priority_weights = tuple(float(value) for value in priority_weights)
        self._provider: Any | None = None
        self._guard_until = -1
        self._previous_hp_fraction: float | None = None
        self._last_observed_time = -1
        self._rollback_trigger_count = 0
        self._reactive_guard_count = 0
        self._action_count = 0
        self._observed_lost_gpu_hours = 0.0
        self._last_decision: dict[str, Any] = {}

    def bind_forecast_provider(self, provider: Any) -> None:
        if not hasattr(provider, "safe_quota_by_pool"):
            raise ValueError("SRGFS requires a provider with a safe quota endpoint.")
        self._provider = provider

    def action(self, context: Any) -> np.ndarray:
        if self._provider is None:
            raise ValueError("SRGFS must be bound to its forecast provider.")
        pool, capacity, hp_used, _, base = _single_pool_values(context)
        safe_quota = float(min(self._provider.safe_quota_by_pool[pool].values()))
        srlinear_quota = float(min(self._provider.srlinear_quota_by_pool[pool].values()))
        if abs(base - srlinear_quota) > 0.51:
            raise ValueError("SRGFS base quota differs from its SRLinear endpoint.")
        span = srlinear_quota - safe_quota
        contextual_target = srlinear_quota + self.linear.residual(context) * capacity
        proposed_lambda = 0.0 if abs(span) <= 1e-9 else (
            contextual_target - safe_quota
        ) / span
        proposed_lambda = float(np.clip(proposed_lambda, 0.0, 1.0))
        hp_fraction = hp_used / capacity
        hp_rise = 0.0 if self._previous_hp_fraction is None else (
            hp_fraction - self._previous_hp_fraction
        )
        self._previous_hp_fraction = hp_fraction
        eviction_failure = max(
            (float(value) for value in context.eviction_failure_rate_by_pool.values()),
            default=0.0,
        )
        now = int(context.time)
        reasons = []
        if now < self._guard_until:
            reasons.append("observed_rollback")
        if float(context.hp_queue_gpu_demand) > 0.0:
            reasons.append("hp_queue")
        if hp_rise >= self.hp_rise_trigger_fraction > 0.0:
            reasons.append("hp_rise")
        if eviction_failure >= self.eviction_failure_trigger > 0.0:
            reasons.append("eviction_failure")
        executed_lambda = 0.0 if reasons else proposed_lambda
        target = (1.0 - executed_lambda) * safe_quota + executed_lambda * srlinear_quota
        residual = float(np.clip((target - base) / capacity, -1.0, 1.0))
        self._action_count += 1
        self._reactive_guard_count += int(bool(reasons))
        self._last_decision = {
            "time": now,
            "safe_quota": safe_quota,
            "srlinear_quota": srlinear_quota,
            "proposed_lambda": proposed_lambda,
            "executed_lambda": executed_lambda,
            "target_quota": target,
            "executed_residual": residual,
            "guard_reasons": reasons,
        }
        return _validate_action(np.asarray(
            (residual, *self.priority_weights), dtype=np.float32
        ))

    def observe_transition(self, info: Mapping[str, Any]) -> None:
        observed_time = int(info["time"])
        if observed_time < self._last_observed_time:
            raise ValueError("SRGFS received nonmonotonic transition time.")
        components = info.get("reward_components")
        if not isinstance(components, Mapping):
            raise ValueError("SRGFS requires completed interval feedback.")
        lost_gpu_hours = float(components.get("lost_gpu_hours", 0.0))
        if not np.isfinite(lost_gpu_hours) or lost_gpu_hours < 0.0:
            raise ValueError("SRGFS received invalid lost GPU hours.")
        self._last_observed_time = observed_time
        self._observed_lost_gpu_hours += lost_gpu_hours
        if lost_gpu_hours > self.trigger_lost_gpu_hours:
            self._guard_until = max(
                self._guard_until, observed_time + self.cooldown_seconds
            )
            self._rollback_trigger_count += 1

    def decision_diagnostics(self) -> dict[str, Any]:
        return dict(self._last_decision)

    def runtime_diagnostics(self) -> dict[str, Any]:
        return {
            "action_count": self._action_count,
            "reactive_guard_count": self._reactive_guard_count,
            "rollback_trigger_count": self._rollback_trigger_count,
            "observed_lost_gpu_hours": self._observed_lost_gpu_hours,
            "guard_until": self._guard_until,
            "last_observed_time": self._last_observed_time,
        }

    def manifest(self) -> dict[str, Any]:
        return {
            "policy_type": self.policy_type,
            "linear": self.linear.manifest(),
            "hp_rise_trigger_fraction": self.hp_rise_trigger_fraction,
            "eviction_failure_trigger": self.eviction_failure_trigger,
            "cooldown_seconds": self.cooldown_seconds,
            "trigger_lost_gpu_hours": self.trigger_lost_gpu_hours,
            "priority_weights": list(self.priority_weights),
            "mixing_rule": "bounded_contextual_target_projection_v1",
        }


def _load_artifact(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    artifact = json.loads(raw)
    if not isinstance(artifact, dict) or artifact.get("schema_version") != "1.0":
        raise ValueError(f"Invalid fitted baseline artifact: {path}")
    if artifact.get("test_outcomes_accessed") is not False:
        raise ValueError("Fitted artifact must state test_outcomes_accessed=false.")
    return artifact, hashlib.sha256(raw).hexdigest()


def build_policy(
    baseline: Mapping[str, Any],
    *,
    suite_base: Path,
    allow_smoke_artifact: bool,
) -> tuple[BaselinePolicy, dict[str, Any]]:
    """Build one supported policy and fail closed on missing train artifacts."""

    policy_type = str(baseline.get("policy_type", "fixed_action"))
    if policy_type == "fixed_action":
        policy = FixedActionPolicy(np.asarray(baseline.get("action"), dtype=np.float32))
        return policy, policy.manifest()
    artifact_path = Path(str(baseline.get("action_artifact", "")))
    if not artifact_path.is_absolute():
        artifact_path = (suite_base / artifact_path).resolve()
    if not artifact_path.is_file():
        raise ValueError(f"Required fitted artifact is missing: {artifact_path}")
    artifact, artifact_sha256 = _load_artifact(artifact_path)
    qualification = artifact.get("qualification")
    if qualification != "publication" and not allow_smoke_artifact:
        raise ValueError("Complete runs require a publication qualified artifact.")
    if artifact.get("policy_type") != policy_type:
        raise ValueError("Declared policy type does not match its fitted artifact.")
    priority = tuple(float(value) for value in baseline.get("priority_weights", [0.0] * 6))
    if policy_type == "fixed_residual":
        policy: BaselinePolicy = FixedActionPolicy(np.asarray(
            (float(artifact["residual_fraction"]), *priority), dtype=np.float32
        ))
    elif policy_type == "current_free_headroom":
        policy: BaselinePolicy = CurrentFreeHeadroomPolicy(
            margin_fraction=float(artifact["margin_fraction"]),
            minimum_residual=float(artifact["minimum_residual"]),
            maximum_residual=float(artifact["maximum_residual"]),
            priority_weights=priority,
        )
    elif policy_type == "hour_of_day_residual":
        policy = HourOfDayResidualPolicy(
            residual_by_hour=tuple(float(value) for value in artifact["residual_by_hour"]),
            minimum_residual=float(artifact["minimum_residual"]),
            maximum_residual=float(artifact["maximum_residual"]),
            priority_weights=priority,
        )
    elif policy_type == "expanding_quantile_residual":
        policy = ExpandingQuantileResidualPolicy(
            initial_residuals=tuple(float(value) for value in artifact["initial_residuals"]),
            risk_limit=float(artifact["risk_limit"]),
            horizon_seconds=int(artifact["horizon_seconds"]),
            minimum_residual=float(artifact["minimum_residual"]),
            maximum_residual=float(artifact["maximum_residual"]),
            priority_weights=priority,
        )
    elif policy_type == "contextual_linear_residual":
        policy = _linear_from_artifact(artifact, priority)
    elif policy_type == "guarded_base_quota":
        policy = GuardedBaseQuotaPolicy(
            hp_rise_margin_fraction=float(artifact["hp_rise_margin_fraction"]),
            backlog_bonus_fraction=float(artifact["backlog_bonus_fraction"]),
            emergency_hp_queue_fraction=float(artifact["emergency_hp_queue_fraction"]),
            minimum_residual=float(artifact["minimum_residual"]),
            maximum_residual=float(artifact["maximum_residual"]),
            priority_weights=priority,
        )
    elif policy_type == "srgfs_contextual_mix":
        linear_payload = artifact.get("linear")
        if not isinstance(linear_payload, Mapping):
            raise ValueError("Contextual SRGFS artifact requires a linear mapping.")
        policy = SRGFSContextualMixPolicy(
            linear=_linear_from_artifact(linear_payload, priority),
            hp_rise_trigger_fraction=float(artifact["hp_rise_trigger_fraction"]),
            eviction_failure_trigger=float(artifact["eviction_failure_trigger"]),
            cooldown_seconds=int(artifact["cooldown_seconds"]),
            trigger_lost_gpu_hours=float(artifact["trigger_lost_gpu_hours"]),
            priority_weights=priority,
        )
    else:
        raise ValueError(f"Unsupported baseline policy type {policy_type!r}.")
    manifest = policy.manifest()
    manifest.update({
        "artifact": str(artifact_path.relative_to(suite_base.parent)),
        "artifact_sha256": artifact_sha256,
        "artifact_qualification": qualification,
        "fit_split": artifact.get("fit_split"),
        "source_report_sha256": artifact.get("source_report_sha256"),
    })
    return policy, manifest
