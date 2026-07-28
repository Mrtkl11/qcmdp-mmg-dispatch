from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributions import Normal


def mlp(
    input_dim: int, widths: Iterable[int], output_dim: int, activation: type[nn.Module]
) -> nn.Sequential:
    layers = []
    current = int(input_dim)
    for width in widths:
        layers.extend([nn.Linear(current, int(width)), activation()])
        current = int(width)
    layers.append(nn.Linear(current, int(output_dim)))
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: Iterable[int],
        log_std_init: float,
        log_std_min: float,
        log_std_max: float,
    ):
        super().__init__()
        widths = tuple(int(width) for width in hidden)
        self.backbone = mlp(obs_dim, widths, widths[-1], nn.Tanh)
        self.mean = nn.Linear(widths[-1], act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.eps = 1e-6

    def _distribution(self, obs: torch.Tensor) -> Normal:
        features = self.backbone(obs)
        mean = self.mean(features)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        return Normal(mean, log_std.exp())

    def _log_prob(
        self, distribution: Normal, latent: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return distribution.log_prob(latent).sum(dim=-1) - torch.log1p(
            -action.pow(2) + self.eps
        ).sum(dim=-1)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        distribution = self._distribution(obs)
        latent = distribution.mean if deterministic else distribution.rsample()
        action = torch.tanh(latent)
        return action, self._log_prob(distribution, latent, action)

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        clipped = action.clamp(-1.0 + self.eps, 1.0 - self.eps)
        latent = 0.5 * (torch.log1p(clipped) - torch.log1p(-clipped))
        return self._log_prob(self._distribution(obs), latent, clipped)

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        return self._distribution(obs).entropy().sum(dim=-1)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden: Iterable[int]):
        super().__init__()
        widths = tuple(int(width) for width in hidden)
        self.network = mlp(obs_dim, widths, 1, nn.ReLU)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs).squeeze(-1)


class TwinQNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: Iterable[int]):
        super().__init__()
        widths = tuple(int(width) for width in hidden)
        self.q1 = mlp(obs_dim + act_dim, widths, 1, nn.ReLU)
        self.q2 = mlp(obs_dim + act_dim, widths, 1, nn.ReLU)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat([obs, action], dim=-1)
        return self.q1(inputs).squeeze(-1), self.q2(inputs).squeeze(-1)


class SACActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: Iterable[int],
        log_std_min: float,
        log_std_max: float,
    ):
        super().__init__()
        widths = tuple(int(width) for width in hidden)
        self.backbone = mlp(obs_dim, widths, widths[-1], nn.ReLU)
        self.mean = nn.Linear(widths[-1], act_dim)
        self.log_std = nn.Linear(widths[-1], act_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.eps = 1e-6

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        distribution = Normal(mean, log_std.exp())
        latent = distribution.rsample()
        action = torch.tanh(latent)
        log_prob = distribution.log_prob(latent).sum(dim=-1)
        log_prob -= torch.log1p(-action.pow(2) + self.eps).sum(dim=-1)
        return action, log_prob, torch.tanh(mean)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if deterministic:
            mean, _ = self.forward(obs)
            return torch.tanh(mean), torch.zeros(obs.shape[0], device=obs.device)
        action, log_prob, _ = self.sample(obs)
        return action, log_prob
