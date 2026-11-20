"""Residual quota wrapper for Contextual Linear and SRGFS base controllers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Protocol

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from research_sim.control import build_control_observation


CONSTRAINT_NAMES = (
    "hp_deadline_violations",
    "lost_gpu_hours",
    "hp_queue_gpu_hours_proxy",
)

RESIDUAL_FEATURE_NAMES = (
    *(f"global_{index}" for index in range(12)),
    *(f"queue_mean_{index}" for index in range(6)),
    *(f"queue_max_{index}" for index in range(6)),
    "base_residual",
    "previous_rl_delta",
    "remaining_hp_budget",
    "remaining_lost_budget",
    "remaining_hp_queue_budget",
    "recent_lost_fraction",
    "recent_hp_queue_fraction",
    "srlinear_quota_fraction",
    "safe_quota_fraction",
    "forecast_span_fraction",
    "forecast_available",
    "week_sin_scaled",
    "week_cos_scaled",
)

SRAC_FEATURE_NAMES = (
    *RESIDUAL_FEATURE_NAMES,
    "previous_uncertainty_protection",
)


@dataclass(frozen=True)
class RiskBudgets:
    """Full split constraint budgets and normalization constants."""

    hp_violation_rate_limit: float
    hp_cohort_count: int
    lost_gpu_hours: float
    hp_queue_gpu_hours: float
    expected_steps: int

    def __post_init__(self) -> None:
        values = (
            self.hp_violation_rate_limit,
            self.lost_gpu_hours,
            self.hp_queue_gpu_hours,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in values):
            raise ValueError("Risk budgets must be finite and positive.")
        if self.hp_cohort_count <= 0 or self.expected_steps <= 0:
            raise ValueError("HP cohort count and expected steps must be positive.")

    @property
    def hp_violation_count(self) -> float:
        return self.hp_violation_rate_limit * self.hp_cohort_count

    @property
    def totals(self) -> np.ndarray:
        return np.asarray(
            (
                self.hp_violation_count,
                self.lost_gpu_hours,
                self.hp_queue_gpu_hours,
            ),
            dtype=np.float64,
        )

    def scale_costs(self, raw_costs: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_costs, dtype=np.float64)
        if raw.shape != (len(CONSTRAINT_NAMES),):
            raise ValueError("Constraint cost vector has the wrong shape.")
        return (raw * float(self.expected_steps) / self.totals).astype(np.float32)

    def manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hp_violation_count"] = self.hp_violation_count
        payload["constraint_names"] = list(CONSTRAINT_NAMES)
        payload["scaled_cost_target_per_step"] = 1.0
        return payload


@dataclass(frozen=True)
class SafetyConfig:
    """Deterministic transform between policy output and simulator action."""

    max_policy_delta_fraction: float = 0.05
    max_delta_change_fraction: float = 0.01
    minimum_total_residual: float = -0.15
    maximum_total_residual: float = 0.15
    block_positive_on_hp_queue: bool = True
    block_positive_after_budget_exhaustion: bool = True
    srlinear_never_exceed_forecast_base: bool = True
    smoothness_penalty: float = 0.01
    max_mix_change: float = 0.25
    max_uncertainty_change: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 < self.max_policy_delta_fraction <= 1.0:
            raise ValueError("Maximum policy delta must lie within (0, 1].")
        if not 0.0 < self.max_delta_change_fraction <= self.max_policy_delta_fraction:
            raise ValueError("Maximum delta change must be positive and no larger than policy delta.")
        if not -1.0 <= self.minimum_total_residual <= self.maximum_total_residual <= 1.0:
            raise ValueError("Total residual bounds must lie within [-1, 1].")
        if self.smoothness_penalty < 0.0 or not math.isfinite(self.smoothness_penalty):
            raise ValueError("Smoothness penalty must be finite and nonnegative.")
        if not 0.0 < self.max_mix_change <= 1.0:
            raise ValueError("Maximum SRAC mix change must lie within (0, 1].")
        if not 0.0 < self.max_uncertainty_change <= 1.0:
            raise ValueError("Maximum SRAC uncertainty change must lie within (0, 1].")

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


class PreparedBase(Protocol):
    """Fresh deterministic base controller prepared after simulator reset."""

    policy: Any
    forecast_provider: Any | None
    manifest: Mapping[str, Any]


class BaseFactory(Protocol):
    def prepare(self, env: gym.Env) -> PreparedBase:
        ...


class QuotaResidualEnv(gym.Env):
    """Expose one bounded RL residual while preserving base admission weights."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        inner: gym.Env,
        *,
        base_factory: BaseFactory,
        risk_budgets: RiskBudgets,
        safety: SafetyConfig,
        method_id: str,
        include_forecast_features: bool,
        action_mode: str = "legacy_residual",
        protection_levels: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0),
        reward_weights: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.base_factory = base_factory
        self.risk_budgets = risk_budgets
        self.safety = safety
        self.method_id = str(method_id)
        self.include_forecast_features = bool(include_forecast_features)
        valid_action_modes = {
            "legacy_residual",
            "continuous_protective",
            "discrete_protective",
            "srac_mix",
        }
        if action_mode not in valid_action_modes:
            raise ValueError(f"Unknown quota RL action mode: {action_mode!r}.")
        levels = tuple(float(level) for level in protection_levels)
        if (
            not levels
            or levels[0] != 0.0
            or levels[-1] != 1.0
            or any(not 0.0 <= level <= 1.0 for level in levels)
            or any(left >= right for left, right in zip(levels, levels[1:]))
        ):
            raise ValueError("Protection levels must increase from zero to one.")
        default_weights = {
            "useful_spot_gpu_hours": 1.0,
            "spot_queue_gpu_hours": 0.0,
            "lost_gpu_hours": 0.0,
            "hp_queue_gpu_hours_proxy": 0.0,
        }
        configured_weights = default_weights if reward_weights is None else {
            **default_weights,
            **{str(key): float(value) for key, value in reward_weights.items()},
        }
        if set(configured_weights) != set(default_weights):
            raise ValueError("Quota RL reward weights contain an unknown component.")
        if not all(math.isfinite(value) and value >= 0.0 for value in configured_weights.values()):
            raise ValueError("Quota RL reward weights must be finite and nonnegative.")
        if configured_weights["useful_spot_gpu_hours"] <= 0.0:
            raise ValueError("Useful Spot work must retain positive reward weight.")
        self.action_mode = str(action_mode)
        self.protection_levels = levels
        self.reward_weights = configured_weights
        action_dimension = 2 if action_mode == "srac_mix" else 1
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(action_dimension,), dtype=np.float32
        )
        self.observation_feature_names = (
            SRAC_FEATURE_NAMES if action_mode == "srac_mix" else RESIDUAL_FEATURE_NAMES
        )
        self.observation_space = spaces.Box(
            -1.0,
            1.0,
            shape=(len(self.observation_feature_names),),
            dtype=np.float32,
        )
        self.base_policy: Any | None = None
        self.forecast_provider: Any | None = None
        self.base_manifest: dict[str, Any] = {}
        self._pending_base_action: np.ndarray | None = None
        self._previous_rl_delta = 0.0
        self._previous_mix_alpha = 0.0
        self._previous_uncertainty_beta = 0.0
        self._cumulative_costs = np.zeros(len(CONSTRAINT_NAMES), dtype=np.float64)
        self._recent_costs = np.zeros(len(CONSTRAINT_NAMES), dtype=np.float64)
        self._requested_actions = 0
        self._shielded_actions = 0
        self._reward_sum = 0.0
        self._useful_sum = 0.0

    @property
    def sim(self) -> Any:
        return getattr(self.inner, "sim")

    @property
    def current_context(self) -> Any:
        return getattr(self.inner, "current_context")

    def _raw_observation(self) -> dict[str, np.ndarray]:
        context = self.current_context
        if context is None:
            return {
                "global_features": np.zeros(12, dtype=np.float32),
                "queue_features": np.zeros((1, 6), dtype=np.float32),
                "queue_mask": np.zeros(1, dtype=bool),
            }
        return build_control_observation(
            context,
            queue_size=int(self.inner.episode_spec.control_queue_size),
        )

    def _refresh_base_action(self) -> None:
        if self.current_context is None:
            self._pending_base_action = None
            return
        action = np.asarray(self.base_policy.action(self.current_context), dtype=np.float32)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise ValueError("Base controller returned an invalid action.")
        if np.any(np.abs(action) > 1.0):
            raise ValueError("Base controller action exceeds simulator bounds.")
        if self.action_mode == "srac_mix":
            endpoints = self._srac_endpoints()
            action[0] = np.float32(
                (endpoints["contextual_quota"] - endpoints["srlinear_quota"])
                / endpoints["capacity"]
            )
        self._pending_base_action = action

    def _srac_endpoints(self) -> dict[str, float]:
        if self.current_context is None or self.forecast_provider is None:
            raise RuntimeError("SRAC endpoints require an active forecast provider.")
        pools = tuple(self.current_context.capacity_by_pool)
        if len(pools) != 1:
            raise ValueError("SRAC requires one GPU pool.")
        pool = pools[0]
        capacity = float(self.current_context.capacity_by_pool[pool])
        srlinear_map = getattr(self.forecast_provider, "srlinear_quota_by_pool", {})
        native_map = getattr(self.forecast_provider, "native_quota_by_pool", {})
        uncertainty_map = getattr(self.forecast_provider, "uncertainty_by_pool", {})
        if pool not in srlinear_map or pool not in native_map or pool not in uncertainty_map:
            raise ValueError("SRAC forecast provider lacks required causal endpoints.")
        residual_for_base = getattr(self.base_policy, "residual_for_base", None)
        if not callable(residual_for_base):
            raise ValueError("SRAC base policy lacks explicit base quota evaluation.")
        srlinear_quota = float(min(srlinear_map[pool].values()))
        native_quota = float(min(native_map[pool].values()))
        contextual_quota = float(np.clip(
            native_quota
            + float(residual_for_base(self.current_context, native_quota)) * capacity,
            0.0,
            capacity,
        ))
        uncertainty = float(uncertainty_map[pool])
        if not all(math.isfinite(value) for value in (
            capacity, srlinear_quota, native_quota, contextual_quota, uncertainty
        )):
            raise ValueError("SRAC endpoint contains a nonfinite value.")
        if capacity <= 0.0 or uncertainty < 0.0:
            raise ValueError("SRAC endpoint capacity or uncertainty is invalid.")
        return {
            "capacity": capacity,
            "srlinear_quota": srlinear_quota,
            "native_quota": native_quota,
            "contextual_quota": contextual_quota,
            "uncertainty": uncertainty,
        }

    def _remaining_budget_features(self) -> np.ndarray:
        remaining = 1.0 - self._cumulative_costs / self.risk_budgets.totals
        return np.clip(remaining, 0.0, 1.0).astype(np.float32)

    def _forecast_features(self) -> np.ndarray:
        if not self.include_forecast_features or self.forecast_provider is None:
            return np.zeros(4, dtype=np.float32)
        context = self.current_context
        if context is None:
            return np.zeros(4, dtype=np.float32)
        pools = tuple(context.capacity_by_pool)
        if len(pools) != 1:
            raise ValueError("Quota RL requires one GPU pool.")
        pool = pools[0]
        capacity = max(1.0, float(context.capacity_by_pool[pool]))
        if self.action_mode == "srac_mix":
            endpoints = self._srac_endpoints()
            return np.asarray(
                (
                    np.clip(endpoints["srlinear_quota"] / capacity, 0.0, 1.0),
                    np.clip(endpoints["contextual_quota"] / capacity, 0.0, 1.0),
                    np.clip(
                        abs(
                            endpoints["srlinear_quota"]
                            - endpoints["contextual_quota"]
                        ) / capacity,
                        0.0,
                        1.0,
                    ),
                    np.clip(endpoints["uncertainty"] / capacity, 0.0, 1.0),
                ),
                dtype=np.float32,
            )
        srlinear_map = getattr(self.forecast_provider, "srlinear_quota_by_pool", {})
        safe_map = getattr(self.forecast_provider, "safe_quota_by_pool", {})
        if pool not in srlinear_map or pool not in safe_map:
            return np.zeros(4, dtype=np.float32)
        srlinear = float(min(srlinear_map[pool].values()))
        safe = float(min(safe_map[pool].values()))
        return np.asarray(
            (
                np.clip(srlinear / capacity, 0.0, 1.0),
                np.clip(safe / capacity, 0.0, 1.0),
                np.clip(abs(srlinear - safe) / capacity, 0.0, 1.0),
                1.0,
            ),
            dtype=np.float32,
        )

    def _observation(self) -> np.ndarray:
        raw = self._raw_observation()
        queue = np.asarray(raw["queue_features"], dtype=np.float32)
        mask = np.asarray(raw["queue_mask"], dtype=bool)
        if np.any(mask):
            queue_mean = queue[mask].mean(axis=0)
            queue_max = queue[mask].max(axis=0)
        else:
            queue_mean = np.zeros(6, dtype=np.float32)
            queue_max = np.zeros(6, dtype=np.float32)
        base_residual = (
            0.0 if self._pending_base_action is None else float(self._pending_base_action[0])
        )
        recent = np.clip(
            self._recent_costs[1:] / np.maximum(self.risk_budgets.totals[1:], 1e-12),
            0.0,
            1.0,
        )
        now = 0 if self.current_context is None else int(self.current_context.time)
        week_phase = 2.0 * math.pi * ((now % (7 * 86400)) / (7 * 86400.0))
        features = np.concatenate((
            np.asarray(raw["global_features"], dtype=np.float32),
            queue_mean,
            queue_max,
            np.asarray((base_residual, self._previous_rl_delta), dtype=np.float32),
            self._remaining_budget_features(),
            recent.astype(np.float32),
            self._forecast_features(),
            np.asarray(
                (0.5 + 0.5 * math.sin(week_phase), 0.5 + 0.5 * math.cos(week_phase)),
                dtype=np.float32,
            ),
            *(
                (np.asarray((self._previous_uncertainty_beta,), dtype=np.float32),)
                if self.action_mode == "srac_mix"
                else ()
            ),
        )).astype(np.float32)
        features = np.clip(features, -1.0, 1.0)
        if not self.observation_space.contains(features):
            raise ValueError("Residual quota observation violates its declared space.")
        return features

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        _, info = self.inner.reset(seed=seed, options=options)
        prepared = self.base_factory.prepare(self.inner)
        self.base_policy = prepared.policy
        self.forecast_provider = prepared.forecast_provider
        self.base_manifest = dict(prepared.manifest)
        self._previous_rl_delta = 0.0
        self._previous_mix_alpha = 0.0
        self._previous_uncertainty_beta = 0.0
        self._cumulative_costs.fill(0.0)
        self._recent_costs.fill(0.0)
        self._requested_actions = 0
        self._shielded_actions = 0
        self._reward_sum = 0.0
        self._useful_sum = 0.0
        self._refresh_base_action()
        return self._observation(), {
            **dict(info),
            "method_id": self.method_id,
            "base_policy": self.base_manifest,
            "risk_budgets": self.risk_budgets.manifest(),
            "safety": self.safety.manifest(),
            "action_mode": self.action_mode,
            "protection_levels": list(self.protection_levels),
            "reward_weights": dict(self.reward_weights),
        }

    def _policy_delta(self, normalized_action: float) -> tuple[float, float]:
        """Map the actor output to a signed quota residual and protection intensity."""
        if self.action_mode == "legacy_residual":
            return (
                float(normalized_action) * self.safety.max_policy_delta_fraction,
                max(0.0, -float(normalized_action)),
            )
        intensity = max(0.0, -float(normalized_action))
        if self.action_mode == "discrete_protective":
            intensity = min(self.protection_levels, key=lambda level: abs(level - intensity))
        return -intensity * self.safety.max_policy_delta_fraction, intensity

    def _shield(self, normalized_action: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        if self._pending_base_action is None or self.current_context is None:
            raise RuntimeError("Residual quota environment has no pending decision.")
        base = self._pending_base_action.copy()
        values = np.asarray(normalized_action, dtype=np.float64)
        if self.action_mode == "srac_mix":
            return self._shield_srac(values, base)
        scalar = float(values[0])
        if abs(scalar) <= 1e-12:
            return base, {
                "normalized_policy_action": scalar,
                "action_mode": self.action_mode,
                "protection_intensity": 0.0,
                "requested_rl_delta_fraction": 0.0,
                "executed_rl_delta_fraction": 0.0,
                "base_residual_fraction": float(base[0]),
                "executed_total_residual_fraction": float(base[0]),
                "base_action": base.astype(float).tolist(),
                "executed_simulator_action": base.astype(float).tolist(),
                "shield_reasons": [],
                "shielded": False,
            }
        requested_delta, protection_intensity = self._policy_delta(scalar)
        lower_ramp = self._previous_rl_delta - self.safety.max_delta_change_fraction
        upper_ramp = self._previous_rl_delta + self.safety.max_delta_change_fraction
        executed_delta = float(np.clip(requested_delta, lower_ramp, upper_ramp))
        reasons: list[str] = []
        if abs(executed_delta - requested_delta) > 1e-12:
            reasons.append("ramp_limit")
        if (
            executed_delta > 0.0
            and self.safety.block_positive_on_hp_queue
            and float(self.current_context.hp_queue_gpu_demand) > 0.0
        ):
            executed_delta = 0.0
            reasons.append("hp_queue")
        if (
            executed_delta > 0.0
            and self.safety.block_positive_after_budget_exhaustion
            and np.any(self._cumulative_costs >= self.risk_budgets.totals)
        ):
            executed_delta = 0.0
            reasons.append("risk_budget")
        combined = float(base[0]) + executed_delta
        clipped = float(np.clip(
            combined,
            self.safety.minimum_total_residual,
            self.safety.maximum_total_residual,
        ))
        if abs(clipped - combined) > 1e-12:
            reasons.append("total_residual_bound")
        combined = clipped
        if (
            self.include_forecast_features
            and self.safety.srlinear_never_exceed_forecast_base
            and combined > 0.0
        ):
            combined = 0.0
            reasons.append("srlinear_forecast_cap")
        executed_delta = combined - float(base[0])
        action = base.copy()
        action[0] = np.float32(combined)
        return action, {
            "normalized_policy_action": scalar,
            "action_mode": self.action_mode,
            "protection_intensity": float(protection_intensity),
            "requested_rl_delta_fraction": requested_delta,
            "executed_rl_delta_fraction": executed_delta,
            "base_residual_fraction": float(base[0]),
            "executed_total_residual_fraction": combined,
            "base_action": base.astype(float).tolist(),
            "executed_simulator_action": action.astype(float).tolist(),
            "shield_reasons": reasons,
            "shielded": bool(reasons),
        }

    def _shield_srac(
        self,
        normalized_action: np.ndarray,
        base: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if normalized_action.shape != (2,):
            raise ValueError("SRAC policy action must contain alpha and beta controls.")
        endpoints = self._srac_endpoints()
        if np.all(np.abs(normalized_action) <= 1e-12):
            base_residual = float(base[0])
            return base, {
                "normalized_policy_action": normalized_action.astype(float).tolist(),
                "action_mode": self.action_mode,
                "requested_alpha": 0.0,
                "requested_beta": 0.0,
                "executed_alpha": 0.0,
                "executed_beta": 0.0,
                "requested_rl_delta_fraction": 0.0,
                "executed_rl_delta_fraction": 0.0,
                "base_residual_fraction": base_residual,
                "executed_total_residual_fraction": base_residual,
                "base_action": base.astype(float).tolist(),
                "executed_simulator_action": base.astype(float).tolist(),
                "srlinear_quota": endpoints["srlinear_quota"],
                "native_quota": endpoints["native_quota"],
                "contextual_quota": endpoints["contextual_quota"],
                "uncertainty_gpu": endpoints["uncertainty"],
                "target_quota": endpoints["contextual_quota"],
                "shield_reasons": [],
                "shielded": False,
            }
        requested_alpha = float(np.clip(max(0.0, normalized_action[0]), 0.0, 1.0))
        requested_beta = float(np.clip(max(0.0, normalized_action[1]), 0.0, 1.0))
        alpha = float(np.clip(
            requested_alpha,
            self._previous_mix_alpha - self.safety.max_mix_change,
            self._previous_mix_alpha + self.safety.max_mix_change,
        ))
        beta = float(np.clip(
            requested_beta,
            self._previous_uncertainty_beta - self.safety.max_uncertainty_change,
            self._previous_uncertainty_beta + self.safety.max_uncertainty_change,
        ))
        reasons: list[str] = []
        if abs(alpha - requested_alpha) > 1e-12:
            reasons.append("alpha_ramp_limit")
        if abs(beta - requested_beta) > 1e-12:
            reasons.append("beta_ramp_limit")
        if (
            self.safety.block_positive_on_hp_queue
            and float(self.current_context.hp_queue_gpu_demand) > 0.0
        ):
            alpha = 0.0
            beta = 1.0
            reasons.append("hp_queue_protection")
        if (
            self.safety.block_positive_after_budget_exhaustion
            and np.any(self._cumulative_costs >= self.risk_budgets.totals)
        ):
            alpha = 0.0
            beta = 1.0
            reasons.append("risk_budget_protection")
        target = (
            alpha * endpoints["srlinear_quota"]
            + (1.0 - alpha) * endpoints["contextual_quota"]
            - beta * endpoints["uncertainty"]
        )
        target = float(np.clip(target, 0.0, endpoints["capacity"]))
        executed_residual = float(np.clip(
            (target - endpoints["srlinear_quota"]) / endpoints["capacity"],
            -1.0,
            1.0,
        ))
        action = base.copy()
        action[0] = np.float32(executed_residual)
        base_residual = float(base[0])
        return action, {
            "normalized_policy_action": normalized_action.astype(float).tolist(),
            "action_mode": self.action_mode,
            "requested_alpha": requested_alpha,
            "requested_beta": requested_beta,
            "executed_alpha": alpha,
            "executed_beta": beta,
            "requested_rl_delta_fraction": float(
                requested_alpha
                * (endpoints["srlinear_quota"] - endpoints["contextual_quota"])
                / endpoints["capacity"]
                - requested_beta * endpoints["uncertainty"] / endpoints["capacity"]
            ),
            "executed_rl_delta_fraction": executed_residual - base_residual,
            "base_residual_fraction": base_residual,
            "executed_total_residual_fraction": executed_residual,
            "base_action": base.astype(float).tolist(),
            "executed_simulator_action": action.astype(float).tolist(),
            "srlinear_quota": endpoints["srlinear_quota"],
            "native_quota": endpoints["native_quota"],
            "contextual_quota": endpoints["contextual_quota"],
            "uncertainty_gpu": endpoints["uncertainty"],
            "target_quota": target,
            "shield_reasons": reasons,
            "shielded": bool(reasons),
        }

    def step(self, action: np.ndarray):
        value = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(value):
            raise ValueError("RL action violates the declared residual action space.")
        simulator_action, diagnostics = self._shield(value)
        previous_delta = self._previous_rl_delta
        previous_alpha = self._previous_mix_alpha
        previous_beta = self._previous_uncertainty_beta
        current_hp_queue = float(self.current_context.hp_queue_gpu_demand)
        current_capacity = float(sum(self.current_context.capacity_by_pool.values()))
        _, _, terminated, truncated, info = self.inner.step(simulator_action)
        observe = getattr(self.base_policy, "observe_transition", None)
        if callable(observe):
            observe(info)
        components = dict(info["reward_components"])
        elapsed_hours = float(info.get("delta_t", 0.0)) / 3600.0
        raw_costs = np.asarray(
            (
                float(info.get("hp_constraint_cost", 0.0)),
                float(components.get("lost_gpu_hours", 0.0)),
                current_hp_queue * elapsed_hours,
            ),
            dtype=np.float64,
        )
        if np.any(raw_costs < 0.0) or not np.all(np.isfinite(raw_costs)):
            raise ValueError("Simulator returned invalid constraint costs.")
        self._recent_costs = raw_costs
        self._cumulative_costs += raw_costs
        self._previous_rl_delta = float(diagnostics["executed_rl_delta_fraction"])
        if self.action_mode == "srac_mix":
            self._previous_mix_alpha = float(diagnostics["executed_alpha"])
            self._previous_uncertainty_beta = float(diagnostics["executed_beta"])
        self._requested_actions += 1
        self._shielded_actions += int(diagnostics["shielded"])
        capacity_hours = max(current_capacity * elapsed_hours, 1e-12)
        useful = float(components.get("useful_spot_gpu_hours", 0.0))
        spot_queue = float(components.get("spot_queue_gpu_hours", 0.0))
        lost = float(components.get("lost_gpu_hours", 0.0))
        hp_queue_proxy = current_hp_queue * elapsed_hours
        smoothness = (
            abs(self._previous_mix_alpha - previous_alpha)
            + abs(self._previous_uncertainty_beta - previous_beta)
            if self.action_mode == "srac_mix"
            else abs(self._previous_rl_delta - previous_delta)
        )
        weighted_utility = (
            self.reward_weights["useful_spot_gpu_hours"] * useful
            - self.reward_weights["spot_queue_gpu_hours"] * spot_queue
            - self.reward_weights["lost_gpu_hours"] * lost
            - self.reward_weights["hp_queue_gpu_hours_proxy"] * hp_queue_proxy
        )
        reward = weighted_utility / capacity_hours - self.safety.smoothness_penalty * smoothness
        self._reward_sum += reward
        self._useful_sum += useful
        extended = {
            **dict(info),
            "rl_action": diagnostics,
            "constraint_names": list(CONSTRAINT_NAMES),
            "constraint_costs_raw": raw_costs.astype(float).tolist(),
            "constraint_costs_scaled": self.risk_budgets.scale_costs(raw_costs).tolist(),
            "constraint_costs_cumulative": self._cumulative_costs.astype(float).tolist(),
            "remaining_budget_fraction": self._remaining_budget_features().astype(float).tolist(),
            "training_reward": float(reward),
            "training_reward_terms": {
                "useful_spot_gpu_hours": useful,
                "spot_queue_gpu_hours": spot_queue,
                "lost_gpu_hours": lost,
                "hp_queue_gpu_hours_proxy": hp_queue_proxy,
                "weighted_utility": weighted_utility,
                "smoothness": smoothness,
            },
        }
        self._refresh_base_action()
        return self._observation(), float(reward), terminated, truncated, extended

    def terminate_external(self, reason: str = "max_control_steps_reached") -> None:
        if self.sim is not None:
            self.sim.terminate(reason=reason, externally_truncated=True)

    def runtime_diagnostics(self) -> dict[str, Any]:
        provider_manifest = (
            None
            if self.forecast_provider is None
            else self.forecast_provider.manifest()
        )
        return {
            "method_id": self.method_id,
            "requested_actions": self._requested_actions,
            "shielded_actions": self._shielded_actions,
            "shield_rate": self._shielded_actions / max(1, self._requested_actions),
            "training_reward_sum": self._reward_sum,
            "useful_spot_gpu_hours_sum": self._useful_sum,
            "action_mode": self.action_mode,
            "protection_levels": list(self.protection_levels),
            "reward_weights": dict(self.reward_weights),
            "constraint_costs_raw": dict(zip(
                CONSTRAINT_NAMES, self._cumulative_costs.astype(float).tolist()
            )),
            "base_policy": self.base_manifest,
            "forecast_provider_runtime": provider_manifest,
        }
