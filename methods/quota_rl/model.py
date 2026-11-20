"""Compact recurrent actor critic for bounded residual quota control."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.distributions import Normal


LOG_PROB_EPSILON = 1e-6


class RecurrentQuotaActorCritic(nn.Module):
    """One GRU policy with reward and explicit constraint value heads."""

    def __init__(
        self,
        observation_dim: int,
        *,
        hidden_size: int = 64,
        constraint_count: int = 3,
        action_dim: int = 1,
        initial_log_std: float = -2.3,
    ) -> None:
        super().__init__()
        if observation_dim <= 0 or hidden_size <= 0 or constraint_count <= 0 or action_dim <= 0:
            raise ValueError("Model dimensions must be positive.")
        self.observation_dim = int(observation_dim)
        self.hidden_size = int(hidden_size)
        self.constraint_count = int(constraint_count)
        self.action_dim = int(action_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.observation_dim, self.hidden_size),
            nn.Tanh(),
        )
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.actor_mean = nn.Linear(self.hidden_size, self.action_dim)
        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), float(initial_log_std))
        )
        self.reward_value = nn.Linear(self.hidden_size, 1)
        self.cost_values = nn.Linear(self.hidden_size, self.constraint_count)
        nn.init.zeros_(self.actor_mean.weight)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.reward_value.weight, gain=1.0)
        nn.init.zeros_(self.reward_value.bias)
        nn.init.orthogonal_(self.cost_values.weight, gain=1.0)
        nn.init.zeros_(self.cost_values.bias)

    def initial_hidden(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.zeros(1, int(batch_size), self.hidden_size, device=device)

    def forward_sequence(
        self,
        observations: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if observations.ndim != 3 or observations.shape[-1] != self.observation_dim:
            raise ValueError("Observations must have shape [time, batch, observation_dim].")
        encoded = self.encoder(observations)
        recurrent, next_hidden = self.gru(encoded, hidden)
        mean = self.actor_mean(recurrent)
        reward_value = self.reward_value(recurrent).squeeze(-1)
        cost_values = self.cost_values(recurrent)
        return mean, reward_value, cost_values, next_hidden

    def step(
        self,
        observations: torch.Tensor,
        hidden: torch.Tensor,
        *,
        deterministic: bool,
    ) -> dict[str, torch.Tensor]:
        if observations.ndim != 2:
            raise ValueError("Step observations must have shape [batch, observation_dim].")
        mean, reward_value, cost_values, next_hidden = self.forward_sequence(
            observations.unsqueeze(0), hidden
        )
        mean = mean.squeeze(0)
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        distribution = Normal(mean, std)
        pre_tanh = mean if deterministic else distribution.sample()
        action = torch.tanh(pre_tanh)
        log_prob = self.tanh_log_prob(distribution, pre_tanh, action)
        return {
            "action": action,
            "pre_tanh_action": pre_tanh,
            "log_prob": log_prob,
            "reward_value": reward_value.squeeze(0),
            "cost_values": cost_values.squeeze(0),
            "hidden": next_hidden,
            "entropy": distribution.entropy().sum(dim=-1),
        }

    @staticmethod
    def tanh_log_prob(
        distribution: Normal,
        pre_tanh: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        base = distribution.log_prob(pre_tanh).sum(dim=-1)
        correction = torch.log(1.0 - action.pow(2) + LOG_PROB_EPSILON).sum(dim=-1)
        return base - correction

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        hidden: torch.Tensor,
        pre_tanh_actions: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mean, reward_value, cost_values, next_hidden = self.forward_sequence(
            observations, hidden
        )
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        distribution = Normal(mean, std)
        return {
            "log_prob": self.tanh_log_prob(distribution, pre_tanh_actions, actions),
            "entropy": distribution.entropy().sum(dim=-1),
            "reward_value": reward_value,
            "cost_values": cost_values,
            "hidden": next_hidden,
        }

    def manifest(self) -> dict[str, Any]:
        return {
            "model": "recurrent_quota_actor_critic",
            "observation_dim": self.observation_dim,
            "hidden_size": self.hidden_size,
            "constraint_count": self.constraint_count,
            "action_dim": self.action_dim,
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "policy_distribution": "tanh_gaussian",
            "actor_initialization": "zero_mean_residual",
        }
