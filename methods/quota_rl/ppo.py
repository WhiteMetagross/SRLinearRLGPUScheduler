"""On policy recurrent PPO with explicit Lagrangian constraint costs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .model import RecurrentQuotaActorCritic


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    pre_tanh_actions: torch.Tensor
    old_log_probs: torch.Tensor
    rewards: torch.Tensor
    costs: torch.Tensor
    dones: torch.Tensor
    reward_values: torch.Tensor
    cost_values: torch.Tensor
    reward_advantages: torch.Tensor
    cost_advantages: torch.Tensor
    reward_returns: torch.Tensor
    cost_returns: torch.Tensor
    initial_hidden: torch.Tensor
    next_observation: torch.Tensor
    terminal: bool
    infos: list[dict[str, Any]]

    @property
    def steps(self) -> int:
        return int(self.observations.shape[0])


def _gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    bootstrap: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(bootstrap)
    next_value = bootstrap
    for index in range(rewards.shape[0] - 1, -1, -1):
        active = 1.0 - dones[index]
        delta = rewards[index] + gamma * next_value * active - values[index]
        running = delta + gamma * gae_lambda * active * running
        advantages[index] = running
        next_value = values[index]
    return advantages, advantages + values


def collect_rollout(
    env: Any,
    model: RecurrentQuotaActorCritic,
    observation: np.ndarray,
    hidden: torch.Tensor,
    *,
    rollout_steps: int,
    device: torch.device,
    deterministic: bool = False,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[RolloutBatch, np.ndarray, torch.Tensor]:
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive.")
    if not 0.0 < gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("Gamma and GAE lambda must lie within their valid ranges.")
    observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    pre_tanh_actions: list[torch.Tensor] = []
    log_probs: list[torch.Tensor] = []
    rewards: list[float] = []
    costs: list[np.ndarray] = []
    dones: list[float] = []
    reward_values: list[torch.Tensor] = []
    cost_values: list[torch.Tensor] = []
    infos: list[dict[str, Any]] = []
    initial_hidden = hidden.detach().clone()
    terminal = False
    model.eval()
    for _ in range(int(rollout_steps)):
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            output = model.step(obs_tensor, hidden, deterministic=deterministic)
        action = output["action"]
        next_observation, reward, terminated, truncated, info = env.step(
            action.squeeze(0).cpu().numpy().astype(np.float32)
        )
        done = bool(terminated or truncated)
        observations.append(obs_tensor.squeeze(0))
        actions.append(action.squeeze(0))
        pre_tanh_actions.append(output["pre_tanh_action"].squeeze(0))
        log_probs.append(output["log_prob"].squeeze(0))
        reward_values.append(output["reward_value"].squeeze(0))
        cost_values.append(output["cost_values"].squeeze(0))
        rewards.append(float(reward))
        scaled_costs = np.asarray(info["constraint_costs_scaled"], dtype=np.float32)
        if scaled_costs.shape != (model.constraint_count,):
            raise ValueError("Environment constraint vector differs from model heads.")
        costs.append(scaled_costs)
        dones.append(float(done))
        infos.append(dict(info))
        observation = np.asarray(next_observation, dtype=np.float32)
        hidden = output["hidden"].detach()
        if done:
            terminal = True
            break
    if not observations:
        raise RuntimeError("Rollout collected no transitions.")
    if terminal:
        bootstrap_reward = torch.zeros((), device=device)
        bootstrap_cost = torch.zeros(model.constraint_count, device=device)
    else:
        next_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            bootstrap = model.step(next_tensor, hidden, deterministic=True)
        bootstrap_reward = bootstrap["reward_value"].squeeze(0)
        bootstrap_cost = bootstrap["cost_values"].squeeze(0)
    reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    cost_tensor = torch.as_tensor(np.stack(costs), dtype=torch.float32, device=device)
    done_tensor = torch.as_tensor(dones, dtype=torch.float32, device=device)
    reward_value_tensor = torch.stack(reward_values)
    cost_value_tensor = torch.stack(cost_values)
    reward_advantages, reward_returns = _gae(
        reward_tensor,
        reward_value_tensor,
        done_tensor,
        bootstrap_reward,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    cost_advantages = torch.zeros_like(cost_tensor)
    cost_returns = torch.zeros_like(cost_tensor)
    for constraint in range(model.constraint_count):
        advantage, returns = _gae(
            cost_tensor[:, constraint],
            cost_value_tensor[:, constraint],
            done_tensor,
            bootstrap_cost[constraint],
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        cost_advantages[:, constraint] = advantage
        cost_returns[:, constraint] = returns
    batch = RolloutBatch(
        observations=torch.stack(observations),
        actions=torch.stack(actions),
        pre_tanh_actions=torch.stack(pre_tanh_actions),
        old_log_probs=torch.stack(log_probs),
        rewards=reward_tensor,
        costs=cost_tensor,
        dones=done_tensor,
        reward_values=reward_value_tensor,
        cost_values=cost_value_tensor,
        reward_advantages=reward_advantages,
        cost_advantages=cost_advantages,
        reward_returns=reward_returns,
        cost_returns=cost_returns,
        initial_hidden=initial_hidden,
        next_observation=torch.as_tensor(observation, dtype=torch.float32, device=device),
        terminal=terminal,
        infos=infos,
    )
    return batch, observation, hidden


class ConstrainedRecurrentPPO:
    """Clipped recurrent PPO with one dual variable per explicit constraint."""

    def __init__(
        self,
        model: RecurrentQuotaActorCritic,
        *,
        learning_rate: float = 3e-4,
        clip_epsilon: float = 0.2,
        value_coefficient: float = 0.5,
        entropy_coefficient: float = 0.005,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        dual_learning_rate: float = 0.02,
        initial_lagrange: float = 0.0,
        dual_mode: str = "integral",
        pid_kp: float = 0.05,
        pid_ki: float = 0.005,
        pid_kd: float = 0.02,
        pid_ema_alpha: float = 0.9,
    ) -> None:
        if ppo_epochs <= 0:
            raise ValueError("ppo_epochs must be positive.")
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)
        self.clip_epsilon = float(clip_epsilon)
        self.value_coefficient = float(value_coefficient)
        self.entropy_coefficient = float(entropy_coefficient)
        self.max_grad_norm = float(max_grad_norm)
        self.ppo_epochs = int(ppo_epochs)
        self.dual_learning_rate = float(dual_learning_rate)
        if dual_mode not in {"integral", "pid"}:
            raise ValueError("dual_mode must be integral or pid.")
        if min(pid_kp, pid_ki, pid_kd) < 0.0 or not 0.0 <= pid_ema_alpha < 1.0:
            raise ValueError("PID gains must be nonnegative and EMA alpha must lie in [0, 1).")
        self.dual_mode = str(dual_mode)
        self.pid_kp = float(pid_kp)
        self.pid_ki = float(pid_ki)
        self.pid_kd = float(pid_kd)
        self.pid_ema_alpha = float(pid_ema_alpha)
        self.lagrange = np.full(
            model.constraint_count, float(initial_lagrange), dtype=np.float64
        )
        self.pid_integral = np.zeros(model.constraint_count, dtype=np.float64)
        self.pid_filtered_error = np.zeros(model.constraint_count, dtype=np.float64)
        self.pid_previous_error = np.zeros(model.constraint_count, dtype=np.float64)
        self.updates = 0

    def update(self, batch: RolloutBatch) -> dict[str, Any]:
        lagrange = torch.as_tensor(
            self.lagrange,
            dtype=torch.float32,
            device=batch.observations.device,
        )
        combined_advantage = batch.reward_advantages - torch.sum(
            batch.cost_advantages * lagrange.unsqueeze(0), dim=-1
        )
        combined_advantage = (
            combined_advantage - combined_advantage.mean()
        ) / (combined_advantage.std(unbiased=False) + 1e-8)
        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
        }
        self.model.train()
        for _ in range(self.ppo_epochs):
            output = self.model.evaluate_actions(
                batch.observations.unsqueeze(1),
                batch.initial_hidden,
                batch.pre_tanh_actions.unsqueeze(1),
                batch.actions.unsqueeze(1),
            )
            new_log_prob = output["log_prob"].squeeze(1)
            ratio = torch.exp(new_log_prob - batch.old_log_probs)
            unclipped = ratio * combined_advantage
            clipped = torch.clamp(
                ratio,
                1.0 - self.clip_epsilon,
                1.0 + self.clip_epsilon,
            ) * combined_advantage
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            reward_value = output["reward_value"].squeeze(1)
            cost_values = output["cost_values"].squeeze(1)
            reward_value_loss = torch.mean((reward_value - batch.reward_returns) ** 2)
            cost_value_loss = torch.mean((cost_values - batch.cost_returns) ** 2)
            value_loss = reward_value_loss + cost_value_loss
            entropy = output["entropy"].mean()
            loss = (
                policy_loss
                + self.value_coefficient * value_loss
                - self.entropy_coefficient * entropy
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            totals["policy_loss"] += float(policy_loss.detach().cpu())
            totals["value_loss"] += float(value_loss.detach().cpu())
            totals["entropy"] += float(entropy.detach().cpu())
            totals["total_loss"] += float(loss.detach().cpu())
        mean_cost = batch.costs.mean(dim=0).detach().cpu().numpy()
        dual_error = mean_cost - 1.0
        if self.dual_mode == "integral":
            self.lagrange = np.maximum(
                0.0,
                self.lagrange + self.dual_learning_rate * dual_error,
            )
        else:
            filtered = (
                self.pid_ema_alpha * self.pid_filtered_error
                + (1.0 - self.pid_ema_alpha) * dual_error
            )
            self.pid_integral = np.clip(
                self.pid_integral + dual_error,
                0.0,
                100.0,
            )
            derivative = filtered - self.pid_previous_error
            self.lagrange = np.maximum(
                0.0,
                self.pid_kp * filtered
                + self.pid_ki * self.pid_integral
                + self.pid_kd * derivative,
            )
            self.pid_previous_error = filtered.copy()
            self.pid_filtered_error = filtered
        self.updates += 1
        result = {key: value / self.ppo_epochs for key, value in totals.items()}
        result.update({
            "update": self.updates,
            "rollout_steps": batch.steps,
            "mean_reward": float(batch.rewards.mean().detach().cpu()),
            "mean_scaled_costs": mean_cost.astype(float).tolist(),
            "lagrange": self.lagrange.astype(float).tolist(),
            "dual_mode": self.dual_mode,
            "dual_error": dual_error.astype(float).tolist(),
        })
        return result

    @torch.no_grad()
    def recompute_hidden(
        self,
        observations: torch.Tensor,
        *,
        burn_in_steps: int,
    ) -> torch.Tensor:
        sequence = observations[-max(1, int(burn_in_steps)):].unsqueeze(1)
        hidden = self.model.initial_hidden(1, sequence.device)
        _, _, _, hidden = self.model.forward_sequence(sequence, hidden)
        return hidden.detach()

    def state_dict(self) -> dict[str, Any]:
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lagrange": self.lagrange.tolist(),
            "dual_mode": self.dual_mode,
            "pid_integral": self.pid_integral.tolist(),
            "pid_filtered_error": self.pid_filtered_error.tolist(),
            "pid_previous_error": self.pid_previous_error.tolist(),
            "updates": self.updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.lagrange = np.asarray(state["lagrange"], dtype=np.float64)
        saved_mode = str(state.get("dual_mode", "integral"))
        if saved_mode != self.dual_mode:
            raise ValueError("Saved dual controller differs from the configured controller.")
        self.pid_integral = np.asarray(
            state.get("pid_integral", np.zeros_like(self.lagrange)), dtype=np.float64
        )
        self.pid_filtered_error = np.asarray(
            state.get("pid_filtered_error", np.zeros_like(self.lagrange)), dtype=np.float64
        )
        self.pid_previous_error = np.asarray(
            state.get("pid_previous_error", np.zeros_like(self.lagrange)), dtype=np.float64
        )
        self.updates = int(state["updates"])
