from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn, optim

from .common import (
    compute_gae,
    constraint_cost,
    normalize,
    reset_observation,
    step_environment,
)
from .models import GaussianActor, ValueNetwork


class OnPolicyTrainer:
    def __init__(self, env: Any, config: Dict[str, Any], algorithm: str):
        self.env = env
        self.config = config
        self.algorithm = algorithm
        self.device = config["device"]
        obs_dim = int(env.observation_space.shape[0])
        act_dim = int(env.action_space.shape[0])
        self.actor = GaussianActor(
            obs_dim,
            act_dim,
            config["actor_hidden"],
            config["log_std_init"],
            config["log_std_min"],
            config["log_std_max"],
        ).to(self.device)
        self.reward_critic = self._build_reward_critic(obs_dim).to(self.device)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=float(config["actor_lr"])
        )
        self.reward_optimizer = optim.Adam(
            self.reward_critic.parameters(), lr=float(config["critic_lr"])
        )
        self.cost_critic = None
        self.cost_optimizer = None
        if algorithm == "constrained_ppo":
            self.cost_critic = ValueNetwork(obs_dim, config["critic_hidden"]).to(
                self.device
            )
            self.cost_optimizer = optim.Adam(
                self.cost_critic.parameters(), lr=float(config["critic_lr"])
            )
            self.lagrangian_multiplier = float(config["initial_lambda"])
        else:
            self.lagrangian_multiplier = 0.0
        self.history: List[Dict[str, float]] = []

    def _build_reward_critic(self, obs_dim: int) -> nn.Module:
        return ValueNetwork(obs_dim, self.config["critic_hidden"])

    def state_dict_bundle(self) -> Dict[str, Any]:
        bundle = {
            "actor": self.actor.state_dict(),
            "reward_critic": self.reward_critic.state_dict(),
            "lagrange_multiplier": float(self.lagrangian_multiplier),
        }
        if self.cost_critic is not None:
            bundle["cost_critic"] = self.cost_critic.state_dict()
        return bundle

    def load_state_dict_bundle(self, bundle: Dict[str, Any]) -> None:
        self.actor.load_state_dict(bundle["actor"])
        self.reward_critic.load_state_dict(bundle["reward_critic"])
        if self.cost_critic is not None and "cost_critic" in bundle:
            self.cost_critic.load_state_dict(bundle["cost_critic"])
        self.lagrangian_multiplier = float(bundle.get("lagrange_multiplier", 0.0))

    def _value(self, critic: nn.Module, observation: np.ndarray) -> float:
        tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            return float(critic(tensor).item())

    def collect_rollout(self) -> Tuple[Dict[str, np.ndarray], float, float]:
        observation = reset_observation(self.env)
        storage: Dict[str, List[Any]] = {
            key: []
            for key in (
                "obs",
                "act",
                "rew",
                "cost",
                "done",
                "logp",
                "value",
                "next_value",
                "cost_value",
                "next_cost_value",
            )
        }
        episode_returns: List[float] = []
        episode_costs: List[float] = []
        current_return = 0.0
        current_cost = 0.0
        for _ in range(int(self.config["horizon"])):
            obs_tensor = torch.as_tensor(
                observation, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                action_tensor, logp_tensor = self.actor.act(obs_tensor)
                reward_value = self.reward_critic(obs_tensor)
                cost_value = (
                    self.cost_critic(obs_tensor)
                    if self.cost_critic is not None
                    else torch.zeros_like(reward_value)
                )
            action = action_tensor.squeeze(0).cpu().numpy()
            next_observation, reward, done, info = step_environment(self.env, action)
            cost = constraint_cost(info)
            next_reward_value = (
                0.0 if done else self._value(self.reward_critic, next_observation)
            )
            next_cost_value = (
                0.0
                if done or self.cost_critic is None
                else self._value(self.cost_critic, next_observation)
            )
            storage["obs"].append(observation)
            storage["act"].append(action)
            storage["rew"].append(reward)
            storage["cost"].append(cost * float(self.config["cost_scale"]))
            storage["done"].append(float(done))
            storage["logp"].append(float(logp_tensor.item()))
            storage["value"].append(float(reward_value.item()))
            storage["next_value"].append(next_reward_value)
            storage["cost_value"].append(float(cost_value.item()))
            storage["next_cost_value"].append(next_cost_value)
            current_return += reward
            current_cost += cost
            if done:
                episode_returns.append(current_return)
                episode_costs.append(current_cost)
                current_return = 0.0
                current_cost = 0.0
                observation = reset_observation(self.env)
            else:
                observation = next_observation
        if not episode_returns:
            episode_returns.append(current_return)
            episode_costs.append(current_cost)
        batch = {
            key: np.asarray(value, dtype=np.float32) for key, value in storage.items()
        }
        return batch, float(np.mean(episode_returns)), float(np.mean(episode_costs))

    def _prepare_batch(self, batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        reward_advantage, reward_return = compute_gae(
            batch["rew"],
            batch["value"],
            batch["next_value"],
            batch["done"],
            float(self.config["gamma"]),
            float(self.config["gae_lambda"]),
        )
        data = {
            "obs": torch.as_tensor(
                batch["obs"], dtype=torch.float32, device=self.device
            ),
            "act": torch.as_tensor(
                batch["act"], dtype=torch.float32, device=self.device
            ),
            "old_logp": torch.as_tensor(
                batch["logp"], dtype=torch.float32, device=self.device
            ),
            "reward_advantage": torch.as_tensor(
                reward_advantage, dtype=torch.float32, device=self.device
            ),
            "reward_return": torch.as_tensor(
                reward_return, dtype=torch.float32, device=self.device
            ),
        }
        if self.cost_critic is not None:
            cost_advantage, cost_return = compute_gae(
                batch["cost"],
                batch["cost_value"],
                batch["next_cost_value"],
                batch["done"],
                float(self.config["gamma"]),
                float(self.config["gae_lambda"]),
            )
            data["cost_advantage"] = torch.as_tensor(
                cost_advantage, dtype=torch.float32, device=self.device
            )
            data["cost_return"] = torch.as_tensor(
                cost_return, dtype=torch.float32, device=self.device
            )
            compound = (
                reward_advantage
                - float(self.lagrangian_multiplier)
                * float(self.config["cost_adv_scale"])
                * cost_advantage
            )
            data["policy_advantage"] = torch.as_tensor(
                normalize(compound), dtype=torch.float32, device=self.device
            )
        else:
            data["policy_advantage"] = torch.as_tensor(
                normalize(reward_advantage), dtype=torch.float32, device=self.device
            )
        return data

    def _update_critics(
        self, data: Dict[str, torch.Tensor], indices: torch.Tensor
    ) -> None:
        values = self.reward_critic(data["obs"][indices])
        reward_loss = (
            0.5
            * (values - data["reward_return"][indices]).pow(2).mean()
            * float(self.config["value_coef"])
        )
        self.reward_optimizer.zero_grad(set_to_none=True)
        reward_loss.backward()
        nn.utils.clip_grad_norm_(
            self.reward_critic.parameters(), float(self.config["max_grad_norm"])
        )
        self.reward_optimizer.step()
        if self.cost_critic is not None:
            cost_values = self.cost_critic(data["obs"][indices])
            cost_loss = (
                0.5
                * (cost_values - data["cost_return"][indices]).pow(2).mean()
                * float(self.config["value_coef"])
                * float(self.config["cost_critic_coef"])
            )
            self.cost_optimizer.zero_grad(set_to_none=True)
            cost_loss.backward()
            nn.utils.clip_grad_norm_(
                self.cost_critic.parameters(), float(self.config["max_grad_norm"])
            )
            self.cost_optimizer.step()

    def _actor_loss(
        self, data: Dict[str, torch.Tensor], indices: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        logp = self.actor.log_prob(data["obs"][indices], data["act"][indices])
        entropy = self.actor.entropy(data["obs"][indices]).mean()
        advantage = data["policy_advantage"][indices]
        if self.algorithm == "a2c":
            loss = (
                -(logp * advantage).mean()
                - float(self.config["entropy_coef"]) * entropy
            )
            approx_kl = float((data["old_logp"][indices] - logp).mean().detach().cpu())
            return loss, approx_kl
        ratio = torch.exp(logp - data["old_logp"][indices])
        clipped = ratio.clamp(
            1.0 - float(self.config["clip_ratio"]),
            1.0 + float(self.config["clip_ratio"]),
        )
        loss = (
            -torch.minimum(ratio * advantage, clipped * advantage).mean()
            - float(self.config["entropy_coef"]) * entropy
        )
        approx_kl = float((data["old_logp"][indices] - logp).mean().detach().cpu())
        return loss, approx_kl

    def update(self, batch: Dict[str, np.ndarray]) -> float:
        data = self._prepare_batch(batch)
        size = data["obs"].shape[0]
        if self.algorithm == "a2c":
            approx_kl = 0.0
            batch_size = int(self.config["batch_size"])
            for _ in range(int(self.config["update_epochs"])):
                permutation = torch.randperm(size, device=self.device)
                for start in range(0, size, batch_size):
                    index = permutation[start : start + batch_size]
                    self._update_critics(data, index)
                    actor_loss, step_kl = self._actor_loss(data, index)
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.actor.parameters(), float(self.config["max_grad_norm"])
                    )
                    self.actor_optimizer.step()
                    approx_kl = max(approx_kl, abs(step_kl))
        else:
            approx_kl = 0.0
            batch_size = int(self.config["batch_size"])
            for _ in range(int(self.config["ppo_epochs"])):
                permutation = torch.randperm(size, device=self.device)
                for start in range(0, size, batch_size):
                    index = permutation[start : start + batch_size]
                    self._update_critics(data, index)
                    actor_loss, step_kl = self._actor_loss(data, index)
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.actor.parameters(), float(self.config["max_grad_norm"])
                    )
                    self.actor_optimizer.step()
                    approx_kl = max(approx_kl, abs(step_kl))
                    if approx_kl > float(self.config["target_kl"]):
                        break
                if approx_kl > float(self.config["target_kl"]):
                    break
        if self.cost_critic is not None:
            mean_cost = float(np.mean(batch["cost"])) / max(
                float(self.config["cost_scale"]), 1e-8
            )
            horizon_cost = mean_cost * float(self.config["dual_update_horizon"])
            error = horizon_cost - float(self.config["cost_limit"])
            self.lagrangian_multiplier = float(
                max(
                    0.0,
                    self.lagrangian_multiplier
                    + float(self.config["lambda_lr"]) * error,
                )
            )
        return float(approx_kl)

    def train(self) -> List[Dict[str, float]]:
        for update_index in range(1, int(self.config["updates"]) + 1):
            batch, mean_return, mean_cost = self.collect_rollout()
            approx_kl = self.update(batch)
            row = {
                "update": float(update_index),
                "rollout_return": float(batch["rew"].sum()),
                "episode_return": mean_return,
                "episode_cost": mean_cost,
                "lagrange_multiplier": float(self.lagrangian_multiplier),
                "approx_kl": approx_kl,
            }
            self.history.append(row)
            interval = int(self.config["log_interval"])
            if (
                update_index == 1
                or update_index % interval == 0
                or update_index == int(self.config["updates"])
            ):
                print(
                    f"{self.algorithm} update {update_index}/{self.config['updates']} return={mean_return:.4f} cost={mean_cost:.4f}"
                )
        return self.history


class A2CTrainer(OnPolicyTrainer):
    def __init__(self, env: Any, config: Dict[str, Any]):
        super().__init__(env, config, "a2c")


class PPOTrainer(OnPolicyTrainer):
    def __init__(self, env: Any, config: Dict[str, Any]):
        super().__init__(env, config, "ppo")


class ConstrainedPPOTrainer(OnPolicyTrainer):
    def __init__(self, env: Any, config: Dict[str, Any]):
        super().__init__(env, config, "constrained_ppo")
