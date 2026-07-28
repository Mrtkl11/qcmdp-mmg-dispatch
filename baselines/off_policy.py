from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn, optim

from .common import constraint_cost, reset_observation, step_environment
from .models import SACActor, TwinQNetwork


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = int(capacity)
        self.observations = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_observations = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        index = self.position
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_observations[index] = next_observation
        self.dones[index] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str) -> Tuple[torch.Tensor, ...]:
        indices = np.random.randint(0, self.size, size=int(batch_size))
        arrays = (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices],
        )
        return tuple(
            torch.as_tensor(array, dtype=torch.float32, device=device)
            for array in arrays
        )


class SACTrainer:
    def __init__(self, env: Any, config: Dict[str, Any]):
        self.env = env
        self.config = config
        self.device = config["device"]
        obs_dim = int(env.observation_space.shape[0])
        act_dim = int(env.action_space.shape[0])
        hidden = config["critic_hidden"]
        self.actor = SACActor(
            obs_dim,
            act_dim,
            config["actor_hidden"],
            config["log_std_min"],
            config["log_std_max"],
        ).to(self.device)
        self.critic = TwinQNetwork(obs_dim, act_dim, hidden).to(self.device)
        self.target_critic = TwinQNetwork(obs_dim, act_dim, hidden).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=float(config["actor_lr"])
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=float(config["critic_lr"])
        )
        self.log_alpha = torch.tensor(
            float(config["log_alpha_init"]),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optimizer = optim.Adam(
            [self.log_alpha], lr=float(config["alpha_lr"])
        )
        self.target_entropy = float(config["target_entropy"])
        self.replay = ReplayBuffer(int(config["buffer_size"]), obs_dim, act_dim)
        self.history: List[Dict[str, float]] = []

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def state_dict_bundle(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }

    def load_state_dict_bundle(self, bundle: Dict[str, Any]) -> None:
        self.actor.load_state_dict(bundle["actor"])
        self.critic.load_state_dict(bundle["critic"])
        self.target_critic.load_state_dict(
            bundle.get("target_critic", bundle["critic"])
        )
        if "log_alpha" in bundle:
            self.log_alpha.data.copy_(
                torch.as_tensor(
                    bundle["log_alpha"], dtype=torch.float32, device=self.device
                )
            )

    def update(self) -> float:
        if self.replay.size < max(
            int(self.config["batch_size"]), int(self.config["update_after"])
        ):
            return 0.0
        observations, actions, rewards, next_observations, dones = self.replay.sample(
            int(self.config["batch_size"]), self.device
        )
        with torch.no_grad():
            next_actions, next_logp, _ = self.actor.sample(next_observations)
            target_q1, target_q2 = self.target_critic(next_observations, next_actions)
            target_q = (
                torch.minimum(target_q1, target_q2) - self.alpha.detach() * next_logp
            )
            target = rewards + float(self.config["gamma"]) * (1.0 - dones) * target_q
        q1, q2 = self.critic(observations, actions)
        critic_loss = 0.5 * ((q1 - target).pow(2) + (q2 - target).pow(2)).mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic.parameters(), float(self.config["max_grad_norm"])
        )
        self.critic_optimizer.step()
        new_actions, logp, _ = self.actor.sample(observations)
        actor_q1, actor_q2 = self.critic(observations, new_actions)
        actor_loss = (
            self.alpha.detach() * logp - torch.minimum(actor_q1, actor_q2)
        ).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor.parameters(), float(self.config["max_grad_norm"])
        )
        self.actor_optimizer.step()
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        tau = float(self.config["tau"])
        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_parameter.mul_(1.0 - tau).add_(tau * parameter)
        return float(critic_loss.detach().cpu())

    def train(self) -> List[Dict[str, float]]:
        observation = reset_observation(self.env)
        total_steps = 0
        episode_return = 0.0
        episode_cost = 0.0
        episode_returns: List[float] = []
        episode_costs: List[float] = []
        for update_index in range(1, int(self.config["updates"]) + 1):
            for _ in range(int(self.config["horizon"])):
                if total_steps < int(self.config["start_steps"]):
                    action = np.asarray(
                        self.env.action_space.sample(), dtype=np.float32
                    )
                else:
                    observation_tensor = torch.as_tensor(
                        observation, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    with torch.no_grad():
                        action_tensor, _ = self.actor.act(observation_tensor)
                    action = action_tensor.squeeze(0).cpu().numpy()
                next_observation, reward, done, info = step_environment(
                    self.env, action
                )
                self.replay.add(observation, action, reward, next_observation, done)
                episode_return += reward
                episode_cost += constraint_cost(info)
                total_steps += 1
                for _ in range(int(self.config["update_epochs"])):
                    self.update()
                if done:
                    episode_returns.append(episode_return)
                    episode_costs.append(episode_cost)
                    episode_return = 0.0
                    episode_cost = 0.0
                    observation = reset_observation(self.env)
                else:
                    observation = next_observation
            report_window = int(self.config["report_window"])
            mean_return = (
                float(np.mean(episode_returns[-report_window:]))
                if episode_returns
                else float(episode_return)
            )
            mean_cost = (
                float(np.mean(episode_costs[-report_window:]))
                if episode_costs
                else float(episode_cost)
            )
            row = {
                "update": float(update_index),
                "rollout_return": (
                    float(sum(episode_returns[-report_window:]))
                    if episode_returns
                    else float(episode_return)
                ),
                "episode_return": mean_return,
                "episode_cost": mean_cost,
                "alpha": float(self.alpha.detach().cpu()),
                "replay_size": float(self.replay.size),
            }
            self.history.append(row)
            interval = int(self.config["log_interval"])
            if (
                update_index == 1
                or update_index % interval == 0
                or update_index == int(self.config["updates"])
            ):
                print(
                    f"sac update {update_index}/{self.config['updates']} return={mean_return:.4f} cost={mean_cost:.4f}"
                )
        return self.history
