"""Gymnasium environment for version 4 quota and admission control."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .control import (
    ADMISSION_FEATURE_DIM,
    CONTROL_GLOBAL_FEATURE_DIM,
    CONTROL_SCHEMA_VERSION,
    build_control_observation,
)
from .core import ResearchSimulator
from .trace import EpisodeTraceView
from .types import ControlDecisionContext, EpisodeSpec, Phase, ResearchSimError
from .verifier import NoCheckVerifier


class UnifiedControlEnv(gym.Env):
    """Controls GFS quota residuals and Spot queue priority every 300 seconds."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        spec: EpisodeSpec,
        view: EpisodeTraceView,
        *,
        scheduler_profile: str = "native_gfs",
        duration_estimator: Optional[Any] = None,
        quota_provider: Optional[Any] = None,
        use_warmup_snapshot: bool = False,
        snapshot_dir: str = "runs/warmup_snapshots_v5",
        snapshot_missing_mode: str = "raise",
        log_dir: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        super().__init__()
        if snapshot_missing_mode not in {"generate", "raise"}:
            raise ValueError("snapshot_missing_mode must be 'generate' or 'raise'.")
        self.episode_spec = spec
        self.view = view
        self.scheduler_profile = str(scheduler_profile)
        self.duration_estimator = duration_estimator
        self.quota_provider = quota_provider
        self.use_warmup_snapshot = bool(use_warmup_snapshot)
        self.snapshot_dir = str(snapshot_dir)
        self.snapshot_missing_mode = snapshot_missing_mode
        self.log_dir = log_dir
        self.run_id = run_id
        self.episode_counter = 0
        self.sim: Optional[ResearchSimulator] = None
        self.current_context: Optional[ControlDecisionContext] = None

        action_dim = 1 + ADMISSION_FEATURE_DIM
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict({
            "global_features": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(CONTROL_GLOBAL_FEATURE_DIM,),
                dtype=np.float32,
            ),
            "queue_features": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(spec.control_queue_size, ADMISSION_FEATURE_DIM),
                dtype=np.float32,
            ),
            "queue_mask": spaces.Box(
                low=0,
                high=1,
                shape=(spec.control_queue_size,),
                dtype=bool,
            ),
        })

    def _empty_observation(self) -> Dict[str, np.ndarray]:
        return {
            "global_features": np.zeros(CONTROL_GLOBAL_FEATURE_DIM, dtype=np.float32),
            "queue_features": np.zeros(
                (self.episode_spec.control_queue_size, ADMISSION_FEATURE_DIM),
                dtype=np.float32,
            ),
            "queue_mask": np.zeros(self.episode_spec.control_queue_size, dtype=bool),
        }

    def _observation(self, context: Optional[ControlDecisionContext]) -> Dict[str, np.ndarray]:
        obs = (
            self._empty_observation()
            if context is None
            else build_control_observation(
                context,
                queue_size=self.episode_spec.control_queue_size,
            )
        )
        if not self.observation_space.contains(obs):
            raise ResearchSimError("Version 4 observation violates its declared Gymnasium space.")
        return obs

    @staticmethod
    def _delta_components(delta: Any) -> Dict[str, float]:
        lost_gpu_hours = float(getattr(delta, "lost_gpu_hours", 0.0))
        queue_gpu_hours = float(getattr(delta, "spot_queue_gpu_hours", 0.0))
        control_reward = getattr(delta, "control_reward", None)
        if control_reward is None:
            control_reward = float(delta.reward) - queue_gpu_hours - lost_gpu_hours
        return {
            "useful_spot_gpu_hours": float(delta.reward),
            "spot_queue_gpu_hours": queue_gpu_hours,
            "lost_gpu_hours": lost_gpu_hours,
            "control_reward": float(control_reward),
            "hp_deadline_violations": float(sum(
                1 for event in delta.events if event.event_type == "hp_deadline"
            )),
        }

    def _new_simulator(self, episode_id: str) -> ResearchSimulator:
        if not self.use_warmup_snapshot:
            return ResearchSimulator(
                spec=self.episode_spec,
                view=self.view,
                policy_type="unified_control",
                scheduler_profile=self.scheduler_profile,
                verifier=NoCheckVerifier(),
                duration_estimator=self.duration_estimator,
                quota_provider=self.quota_provider,
                filter_mode="passthrough",
                log_dir=self.log_dir,
                run_id=self.run_id,
                episode_id=episode_id,
            )

        from .warmup_snapshot import (
            generate_warmup_snapshot,
            get_snapshot_path,
            has_warmup_snapshot,
            load_warmup_snapshot,
        )

        snapshot_path = get_snapshot_path(self.episode_spec, self.snapshot_dir)
        if not has_warmup_snapshot(self.episode_spec, self.snapshot_dir):
            if self.snapshot_missing_mode == "raise":
                raise FileNotFoundError(
                    f"Required warmup snapshot is missing: {snapshot_path}"
                )
            generate_warmup_snapshot(
                spec=self.episode_spec,
                view=self.view,
                snapshot_dir=self.snapshot_dir,
                quota_provider=self.quota_provider,
            )
        sim = load_warmup_snapshot(
            filepath=snapshot_path,
            policy_type="unified_control",
            verifier=NoCheckVerifier(),
            log_dir=self.log_dir,
            spec=self.episode_spec,
            view=self.view,
            duration_estimator=self.duration_estimator,
            quota_provider=self.quota_provider,
            filter_mode="passthrough",
            run_id=self.run_id,
            episode_id=episode_id,
        )
        sim.set_scheduler_profile(self.scheduler_profile)
        sim.forecast_cache = None
        return sim

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        episode_ordinal = self.episode_counter
        self.episode_counter += 1
        episode_id = (options or {}).get(
            "episode_id",
            f"{self.episode_spec.node_filter}_control_{self.episode_spec.policy_seed}_ep{episode_ordinal:04d}",
        )
        self.sim = self._new_simulator(episode_id)

        if (
            self.sim.time == self.episode_spec.score_start
            and self.sim.phase == Phase.RUNNING
            and self.sim.control_paused_context is None
        ):
            self.sim.phase = Phase.CONTROL
            self.sim.control_paused_context = self.sim._build_control_context()
            self.sim.decisions_count += 1
            context = self.sim.control_paused_context
            delta = self.sim._consume_control_delta()
        else:
            context, delta = self.sim.advance_until_decision()

        if context is not None and not isinstance(context, ControlDecisionContext):
            raise ResearchSimError("Unified control replay reached an unexpected placement decision.")
        self.current_context = context
        obs = self._observation(context)
        info = {
            "time": self.sim.time,
            "control_schema_version": CONTROL_SCHEMA_VERSION,
            "base_quota": context.base_quota_by_pool if context is not None else None,
            "delta_t": delta.elapsed_sim_seconds,
            "reward_components": self._delta_components(delta),
            "terminated": bool(delta.terminated),
            "truncated": bool(delta.truncated),
            "terminal_reason": delta.terminal_reason,
        }
        return obs, info

    def step(self, action: np.ndarray):
        if self.sim is None or self.current_context is None:
            raise ResearchSimError("Call reset() before step(), and do not step a finished replay.")
        if not self.action_space.contains(np.asarray(action, dtype=np.float32)):
            raise ResearchSimError("Control action violates the declared action space.")
        action_events = self.sim.apply_control(np.asarray(action, dtype=np.float32))
        context, delta = self.sim.advance_until_decision()
        if context is not None and not isinstance(context, ControlDecisionContext):
            raise ResearchSimError("Unified control replay reached an unexpected placement decision.")
        self.current_context = context
        terminated = bool(delta.terminated)
        truncated = bool(delta.truncated)
        obs = self._observation(context)
        components = self._delta_components(delta)
        control_event = action_events[0]
        info = {
            "time": self.sim.time,
            "control_schema_version": CONTROL_SCHEMA_VERSION,
            "delta_t": delta.elapsed_sim_seconds,
            "base_quota": (
                context.base_quota_by_pool if context is not None else None
            ),
            "executed_quota": control_event.details["executed_quota"],
            "quota_delta_fraction": control_event.details["quota_delta_fraction"],
            "priority_weights": control_event.details["priority_weights"],
            "reward_components": components,
            "hp_constraint_cost": components["hp_deadline_violations"],
            "terminated": terminated,
            "truncated": truncated,
            "terminal_reason": delta.terminal_reason,
        }
        return obs, float(components["control_reward"]), terminated, truncated, info
