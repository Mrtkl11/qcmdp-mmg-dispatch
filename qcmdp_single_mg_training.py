from __future__ import annotations

import os
import time
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from ieee33_mmg_env import IEEE33SingleMGEnv
from qcmdp_model_pennylane import GaussianActor, QuantumCriticConfig, VQCCritic
from release_config import load_optional_json, merge_overrides, resolve_required_data_path


@dataclass(frozen=True)
class QCMDPConfig:
    device: str = "cpu"
    n_qubits: int = 4
    horizon: int = 256
    batch_size: int = 64
    epochs: int = 6
    updates: int = 1000
    lr: float = 2e-4
    critic_lr: float = 1e-4
    gamma: float = 0.995
    gae_lambda: float = 0.98
    clip: float = 0.2
    ent_coef: float = 0.003
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    cost_limit: float = 850.0
    initial_lambda: float = 0.01
    lambda_lr: float = 0.06
    lambda_decay: float = 0.995
    lambda_max: float = 5.0
    cost_scale: float = 0.01
    cost_critic_coef: float = 1.0
    reward_adv_scale: float = 1.0
    cost_adv_scale: float = 1.0
    dual_update_horizon: int = 96
    target_kl: float = 0.05

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "QCMDPConfig":
        return cls(**values)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class RolloutBatch:
    obs: np.ndarray
    act: np.ndarray
    rew: np.ndarray
    cost: np.ndarray
    real_cost: np.ndarray
    done: np.ndarray
    logp: np.ndarray
    val_r: np.ndarray
    next_val_r: np.ndarray
    val_c: np.ndarray
    next_val_c: np.ndarray

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "obs": self.obs,
            "act": self.act,
            "rew": self.rew,
            "cost": self.cost,
            "real_cost": self.real_cost,
            "done": self.done,
            "logp": self.logp,
            "val_r": self.val_r,
            "next_val_r": self.next_val_r,
            "val_c": self.val_c,
            "next_val_c": self.next_val_c,
        }


@dataclass
class UpdateStats:
    lambda_value: float
    approx_kl: float
    horizon_cost: float


@dataclass
class TrainIterationStats:
    update_index: int
    episode_return: float
    episode_constraint_cost: float
    lambda_value: float
    approx_kl: float
    horizon_cost: float
    elapsed_seconds: float


def moving_average(values: Iterable[float], window: int = 10) -> np.ndarray:
    values = np.asarray(list(values), dtype=np.float32)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window, dtype=np.float32) / window, mode="valid")


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    next_values = np.asarray(next_values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)

    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for index in reversed(range(len(rewards))):
        not_done = 1.0 - dones[index]
        td_error = rewards[index] + gamma * next_values[index] * not_done - values[index]
        gae = td_error + gamma * gae_lambda * not_done * gae
        advantages[index] = gae
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


def normalize_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (values - values.mean()) / (values.std() + 1e-8)


class QCMDPSingleMGTrainer:
    def __init__(self, env: IEEE33SingleMGEnv, config: Mapping[str, object] | QCMDPConfig):
        self.env = env
        self.config = config if isinstance(config, QCMDPConfig) else QCMDPConfig.from_mapping(config)
        self.device = self.config.device

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        qcfg = QuantumCriticConfig(n_qubits=self.config.n_qubits)

        self.actor = GaussianActor(obs_dim, act_dim).to(self.device)
        self.reward_critic = VQCCritic(obs_dim, 1, qcfg).to(self.device)
        self.safety_critic = VQCCritic(obs_dim, 1, qcfg).to(self.device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.config.lr)
        self.reward_critic_opt = optim.Adam(self.reward_critic.parameters(), lr=self.config.critic_lr)
        self.safety_critic_opt = optim.Adam(self.safety_critic.parameters(), lr=self.config.critic_lr)

        self.lagrange_multiplier = float(self.config.initial_lambda)
        self.training_history: List[TrainIterationStats] = []

    def state_dict_bundle(self) -> Dict[str, object]:
        return {
            "actor": self.actor.state_dict(),
            "reward_critic": self.reward_critic.state_dict(),
            "safety_critic": self.safety_critic.state_dict(),
            "lagrange_multiplier": self.lagrange_multiplier,
            "config": self.config.to_dict(),
        }

    def load_state_dict_bundle(self, bundle: Mapping[str, object]) -> None:
        self.actor.load_state_dict(bundle["actor"])
        self.reward_critic.load_state_dict(bundle["reward_critic"])
        self.safety_critic.load_state_dict(bundle["safety_critic"])
        self.lagrange_multiplier = float(bundle["lagrange_multiplier"])

    def collect_rollout(self) -> Tuple[RolloutBatch, float, float]:
        obs = self.env.reset()
        storage: Dict[str, List[float]] = {
            "obs": [],
            "act": [],
            "rew": [],
            "cost": [],
            "real_cost": [],
            "done": [],
            "logp": [],
            "val_r": [],
            "next_val_r": [],
            "val_c": [],
            "next_val_c": [],
        }

        episode_rewards: List[float] = []
        episode_costs: List[float] = []
        cumulative_reward = 0.0
        cumulative_cost = 0.0

        for _ in range(self.config.horizon):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action, log_prob = self.actor.act(obs_tensor, deterministic=False)
                reward_value = self.reward_critic(obs_tensor)
                safety_value = self.safety_critic(obs_tensor)

            action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
            next_obs, reward, done, info = self.env.step(action_np)
            real_cost = float(info.get("constraint_cost", 0.0))
            scaled_cost = real_cost * self.config.cost_scale

            with torch.no_grad():
                if done:
                    next_reward_value = 0.0
                    next_safety_value = 0.0
                else:
                    next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    next_reward_value = float(self.reward_critic(next_obs_tensor).item())
                    next_safety_value = float(self.safety_critic(next_obs_tensor).item())

            storage["obs"].append(obs)
            storage["act"].append(action_np)
            storage["rew"].append(float(reward))
            storage["cost"].append(scaled_cost)
            storage["real_cost"].append(real_cost)
            storage["done"].append(bool(done))
            storage["logp"].append(float(log_prob.item()))
            storage["val_r"].append(float(reward_value.item()))
            storage["val_c"].append(float(safety_value.item()))
            storage["next_val_r"].append(next_reward_value)
            storage["next_val_c"].append(next_safety_value)

            cumulative_reward += float(reward)
            cumulative_cost += real_cost

            if done:
                episode_rewards.append(cumulative_reward)
                episode_costs.append(cumulative_cost)
                cumulative_reward = 0.0
                cumulative_cost = 0.0
                obs = self.env.reset()
            else:
                obs = next_obs

        rollout = RolloutBatch(
            obs=np.asarray(storage["obs"], dtype=np.float32),
            act=np.asarray(storage["act"], dtype=np.float32),
            rew=np.asarray(storage["rew"], dtype=np.float32),
            cost=np.asarray(storage["cost"], dtype=np.float32),
            real_cost=np.asarray(storage["real_cost"], dtype=np.float32),
            done=np.asarray(storage["done"], dtype=np.float32),
            logp=np.asarray(storage["logp"], dtype=np.float32),
            val_r=np.asarray(storage["val_r"], dtype=np.float32),
            next_val_r=np.asarray(storage["next_val_r"], dtype=np.float32),
            val_c=np.asarray(storage["val_c"], dtype=np.float32),
            next_val_c=np.asarray(storage["next_val_c"], dtype=np.float32),
        )

        avg_reward = float(np.mean(episode_rewards)) if episode_rewards else float(cumulative_reward)
        avg_cost = float(np.mean(episode_costs)) if episode_costs else float(cumulative_cost)
        return rollout, avg_reward, avg_cost

    def _update_critics(
        self,
        obs_tensor: torch.Tensor,
        reward_returns: torch.Tensor,
        cost_returns: torch.Tensor,
        batch_indices: np.ndarray,
    ) -> None:
        mb_obs = obs_tensor[batch_indices]
        mb_reward_returns = reward_returns[batch_indices]
        mb_cost_returns = cost_returns[batch_indices]

        reward_prediction = self.reward_critic(mb_obs)
        reward_loss = 0.5 * (reward_prediction - mb_reward_returns).pow(2).mean() * self.config.vf_coef
        self.reward_critic_opt.zero_grad()
        reward_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.reward_critic.parameters(), self.config.max_grad_norm)
        self.reward_critic_opt.step()

        safety_prediction = self.safety_critic(mb_obs)
        safety_loss = 0.5 * (safety_prediction - mb_cost_returns).pow(2).mean() * self.config.vf_coef * self.config.cost_critic_coef
        self.safety_critic_opt.zero_grad()
        safety_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.safety_critic.parameters(), self.config.max_grad_norm)
        self.safety_critic_opt.step()

    def _update_actor(
        self,
        obs_tensor: torch.Tensor,
        act_tensor: torch.Tensor,
        old_logp_tensor: torch.Tensor,
        advantages_tensor: torch.Tensor,
        batch_indices: np.ndarray,
    ) -> float:
        mb_obs = obs_tensor[batch_indices]
        mb_act = act_tensor[batch_indices]
        mb_old_logp = old_logp_tensor[batch_indices]
        mb_adv = advantages_tensor[batch_indices]

        new_logp = self.actor.log_prob(mb_obs, mb_act)
        entropy = self.actor.entropy(mb_obs).mean()
        ratio = torch.exp(new_logp - mb_old_logp)
        surrogate_1 = ratio * mb_adv
        surrogate_2 = torch.clamp(ratio, 1.0 - self.config.clip, 1.0 + self.config.clip) * mb_adv
        actor_loss = -torch.min(surrogate_1, surrogate_2).mean() - self.config.ent_coef * entropy

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_opt.step()
        return float((mb_old_logp - new_logp).mean().item())

    def update(self, rollout: RolloutBatch) -> UpdateStats:
        reward_advantages, reward_returns = compute_gae(
            rollout.rew,
            rollout.val_r,
            rollout.next_val_r,
            rollout.done,
            self.config.gamma,
            self.config.gae_lambda,
        )
        cost_advantages, cost_returns = compute_gae(
            rollout.cost,
            rollout.val_c,
            rollout.next_val_c,
            rollout.done,
            self.config.gamma,
            self.config.gae_lambda,
        )

        reward_advantages = normalize_vector(reward_advantages)
        cost_advantages = normalize_vector(cost_advantages)
        compound_advantages = normalize_vector(
            self.config.reward_adv_scale * reward_advantages
            - float(self.lagrange_multiplier) * self.config.cost_adv_scale * cost_advantages
        )

        obs_tensor = torch.tensor(rollout.obs, dtype=torch.float32, device=self.device)
        act_tensor = torch.tensor(rollout.act, dtype=torch.float32, device=self.device)
        old_logp_tensor = torch.tensor(rollout.logp, dtype=torch.float32, device=self.device)
        reward_returns_tensor = torch.tensor(reward_returns, dtype=torch.float32, device=self.device)
        cost_returns_tensor = torch.tensor(cost_returns, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.tensor(compound_advantages, dtype=torch.float32, device=self.device)

        data_size = len(obs_tensor)
        indices = np.arange(data_size)
        approx_kl = 0.0

        for _ in range(self.config.epochs):
            np.random.shuffle(indices)
            early_stop = False
            for start in range(0, data_size, self.config.batch_size):
                batch_indices = indices[start : start + self.config.batch_size]
                self._update_critics(obs_tensor, reward_returns_tensor, cost_returns_tensor, batch_indices)
                approx_kl = self._update_actor(obs_tensor, act_tensor, old_logp_tensor, advantages_tensor, batch_indices)
                if approx_kl > 1.5 * self.config.target_kl:
                    early_stop = True
                    break
            if early_stop:
                break

        mean_step_cost = float(np.mean(rollout.real_cost)) if len(rollout.real_cost) > 0 else 0.0
        horizon_cost = mean_step_cost * max(1, self.config.dual_update_horizon)
        dual_error = (horizon_cost - self.config.cost_limit) / max(self.config.cost_limit, 1e-6)
        new_lambda = self.config.lambda_decay * self.lagrange_multiplier + self.config.lambda_lr * dual_error
        self.lagrange_multiplier = float(np.clip(new_lambda, 0.0, self.config.lambda_max))

        return UpdateStats(
            lambda_value=self.lagrange_multiplier,
            approx_kl=approx_kl,
            horizon_cost=horizon_cost,
        )

    def train(self) -> List[TrainIterationStats]:
        self.training_history = []
        for update_index in range(1, self.config.updates + 1):
            start_time = time.time()
            rollout, avg_reward, avg_cost = self.collect_rollout()
            update_stats = self.update(rollout)
            elapsed = time.time() - start_time

            iteration_stats = TrainIterationStats(
                update_index=update_index,
                episode_return=avg_reward,
                episode_constraint_cost=avg_cost,
                lambda_value=update_stats.lambda_value,
                approx_kl=update_stats.approx_kl,
                horizon_cost=update_stats.horizon_cost,
                elapsed_seconds=elapsed,
            )
            self.training_history.append(iteration_stats)

            if update_index % 10 == 0 or update_index == 1:
                print(
                    f"Update {update_index:04d}/{self.config.updates} | "
                    f"Ep Ret: {avg_reward:7.3f} | "
                    f"Ep Cost: {avg_cost:8.3f} | "
                    f"DualCost: {update_stats.horizon_cost:8.3f} | "
                    f"Lambda: {update_stats.lambda_value:6.3f} | "
                    f"KL: {update_stats.approx_kl:.4f} | "
                    f"Time: {elapsed:.2f}s"
                )
        return self.training_history

    def save_checkpoint(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        bundle = self.state_dict_bundle()
        torch.save(bundle["actor"], output_dir / "actor_weights.pt")
        torch.save(bundle["reward_critic"], output_dir / "reward_critic_weights.pt")
        torch.save(bundle["safety_critic"], output_dir / "safety_critic_weights.pt")
        torch.save({"lagrange_multiplier": bundle["lagrange_multiplier"], "config": bundle["config"]}, output_dir / "trainer_state.pt")


def operation_history_to_dataframes(operation_history: List[Mapping[str, object]]):
    if not operation_history:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    history_df = pd.DataFrame(operation_history).copy()
    history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
    history_df = history_df.sort_values("timestamp").reset_index(drop=True)

    voltage_rows: List[Dict[str, object]] = []
    line_rows: List[Dict[str, object]] = []

    for _, row in history_df.iterrows():
        for bus_index, voltage_value in enumerate(row.get("bus_voltage_pu", []) or [], start=1):
            voltage_rows.append(
                {
                    "timestamp": row["timestamp"],
                    "bus": bus_index,
                    "voltage_pu": float(voltage_value),
                    "mg": row.get("mg", ""),
                }
            )

        line_loading = row.get("line_loading_pu", {})
        line_p_flow = row.get("line_p_flow_mw", {})
        line_loss = row.get("line_loss_kw", {})
        for branch, loading in line_loading.items():
            line_rows.append(
                {
                    "timestamp": row["timestamp"],
                    "branch": branch,
                    "loading_pu": float(loading),
                    "p_flow_mw": float(line_p_flow.get(branch, np.nan)),
                    "loss_kw": float(line_loss.get(branch, np.nan)),
                }
            )

    voltage_df = pd.DataFrame(voltage_rows)
    line_df = pd.DataFrame(line_rows)
    metrics_df = history_df[
        [
            "timestamp",
            "pv_power",
            "wind_power",
            "load_demand",
            "battery_power",
            "diesel_power",
            "grid_power",
            "battery_soc",
            "fuel_level",
            "buy_price",
            "sell_price",
            "curtailment",
            "load_shedding",
            "reward",
            "network_total_loss_kw",
            "network_v_min_pu",
            "network_v_max_pu",
            "network_avg_voltage_dev_pu",
            "network_voltage_violation_sum",
            "slack_p_mw",
            "slack_q_mvar",
            "constraint_cost",
            "economic_cost",
        ]
    ].copy()
    return history_df, voltage_df, line_df, metrics_df


def plot_power_balance(history_df: pd.DataFrame, save_path: str, title: str):
    x = np.arange(len(history_df), dtype=int)
    pv_raw = history_df["pv_power"].astype(float).values
    wind_raw = history_df["wind_power"].astype(float).values
    diesel = history_df["diesel_power"].astype(float).values
    battery = history_df["battery_power"].astype(float).values
    grid = history_df["grid_power"].astype(float).values
    raw_load = history_df["load_demand"].astype(float).values
    curtailment = history_df["curtailment"].astype(float).values
    load_shedding = history_df["load_shedding"].astype(float).values
    buy_price = history_df["buy_price"].astype(float).values
    balance_error = history_df["power_balance_error"].astype(float).values

    renewable_total = np.maximum(pv_raw + wind_raw, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pv_share = np.divide(pv_raw, renewable_total, out=np.zeros_like(pv_raw), where=renewable_total > 1e-9)
        wind_share = np.divide(wind_raw, renewable_total, out=np.zeros_like(wind_raw), where=renewable_total > 1e-9)

    pv_used = np.maximum(pv_raw - curtailment * pv_share, 0.0)
    wind_used = np.maximum(wind_raw - curtailment * wind_share, 0.0)
    bess_discharge = np.maximum(battery, 0.0)
    bess_charge = np.maximum(-battery, 0.0)
    grid_import = np.maximum(grid, 0.0)
    grid_export = np.maximum(-grid, 0.0)
    served_load = np.maximum(raw_load - load_shedding, 0.0)

    positive_stack = {
        "PV": pv_used,
        "WIND": wind_used,
        "DIESEL": diesel,
        "BESS discharge": bess_discharge,
        "GRID import": grid_import,
    }
    negative_stack = {
        "LOAD": served_load,
        "BESS charge": bess_charge,
        "GRID export": grid_export,
    }

    residual = sum(positive_stack.values()) - sum(negative_stack.values())
    residual_pos = np.maximum(residual, 0.0)
    residual_neg = np.maximum(-residual, 0.0)

    if np.any(residual_pos > 1e-9):
        negative_stack["Residual absorption"] = residual_pos
    if np.any(residual_neg > 1e-9):
        positive_stack["Residual supply"] = residual_neg

    fig, ax = plt.subplots(figsize=(16, 8))
    color_map = {
        "LOAD": "#4e79a7",
        "PV": "#f28e2b",
        "WIND": "#59a14f",
        "DIESEL": "#e15759",
        "BESS discharge": "#b07aa1",
        "BESS charge": "#af7aa1",
        "GRID import": "#76b7b2",
        "GRID export": "#9c755f",
        "Residual supply": "#bab0ab",
        "Residual absorption": "#bab0ab",
    }

    bottom_pos = np.zeros(len(history_df), dtype=float)
    for label in ["PV", "WIND", "DIESEL", "BESS discharge", "GRID import", "Residual supply"]:
        if label in positive_stack:
            ax.bar(x, positive_stack[label], bottom=bottom_pos, width=0.92, label=label, color=color_map[label], alpha=0.95)
            bottom_pos += positive_stack[label]

    bottom_neg = np.zeros(len(history_df), dtype=float)
    for label in ["LOAD", "BESS charge", "GRID export", "Residual absorption"]:
        if label in negative_stack:
            ax.bar(x, -negative_stack[label], bottom=-bottom_neg, width=0.92, label=label, color=color_map[label], alpha=0.95)
            bottom_neg += negative_stack[label]

    ax.axhline(0.0, linewidth=1.0, color="black")
    ax.plot(x, balance_error, color="black", linestyle="--", linewidth=1.0, label="Balance error")
    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Generation(+)/Demand(-) (kW)")
    ax.grid(alpha=0.20, axis="y")

    ax2 = ax.twinx()
    ax2.plot(x, buy_price, color="blue", marker="o", markersize=2.5, linewidth=1.0, label="Buy Price")
    ax2.set_ylabel("Electricity Price")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", ncol=2, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_voltage_profile(voltage_df: pd.DataFrame, save_path: str, title: str):
    if voltage_df.empty:
        return
    aggregated = voltage_df.groupby("bus")["voltage_pu"].agg(["min", "mean", "max"]).reset_index()
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.fill_between(aggregated["bus"], aggregated["min"], aggregated["max"], alpha=0.30, label="Min-Max band")
    ax.plot(aggregated["bus"], aggregated["mean"], marker="o", linewidth=1.8, label="Mean voltage")
    ax.set_xlabel("Bus index")
    ax.set_ylabel("Voltage (p.u.)")
    ax.set_title(title)
    ax.grid(alpha=0.22, linestyle="--")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_voltage_heatmap(voltage_df: pd.DataFrame, save_path: str, title: str):
    if voltage_df.empty:
        return
    pivot = voltage_df.pivot(index="timestamp", columns="bus", values="voltage_pu").sort_index()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    image = ax.imshow(pivot.values, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Bus index")
    ax.set_ylabel("Time step")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(list(pivot.columns), rotation=90)
    fig.colorbar(image, ax=ax, label="Voltage (p.u.)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_network_power_metrics(metrics_df: pd.DataFrame, save_path: str):
    if metrics_df.empty:
        return
    x = np.arange(len(metrics_df))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(x, metrics_df["network_total_loss_kw"].astype(float).values, label="Network loss (kW)", color="tab:blue")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Network loss (kW)", color="tab:blue")
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(x, metrics_df["slack_p_mw"].astype(float).values, label="Slack P (MW)", color="tab:orange")
    ax2.set_ylabel("Slack P (MW)", color="tab:orange")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right")
    ax.set_title("Network Power Metrics")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_line_loading(line_df: pd.DataFrame, save_path: str, title: str):
    if line_df.empty:
        return
    latest_timestamp = line_df["timestamp"].max()
    snapshot = line_df[line_df["timestamp"] == latest_timestamp].copy().sort_values("branch")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(snapshot)), snapshot["loading_pu"].astype(float).values)
    ax.axhline(1.0, linestyle="--", color="red", label="Loading Limit = 1.0 p.u.")
    ax.set_xticks(np.arange(len(snapshot)))
    ax.set_xticklabels(snapshot["branch"].tolist(), rotation=90)
    ax.set_ylabel("Loading (p.u.)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_training_rewards(training_history: List[TrainIterationStats], save_path: str):
    episode_returns = [entry.episode_return for entry in training_history]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(1, len(episode_returns) + 1)
    ax.plot(x, episode_returns, alpha=0.4, label="Raw Return")

    ma_window = 10
    ma_values = moving_average(episode_returns, window=ma_window)
    x_ma = np.arange(ma_window, len(episode_returns) + 1) if len(episode_returns) >= ma_window else x
    ax.plot(x_ma, ma_values, linewidth=2, label=f"MA({ma_window})")
    ax.set_xlabel("Update Iterations")
    ax.set_ylabel("Episode Return")
    ax.set_title("Q-CMDP Training Convergence")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_training_history(training_history: List[TrainIterationStats], save_path: str | Path) -> pd.DataFrame:
    history_df = pd.DataFrame([asdict(entry) for entry in training_history])
    history_df.to_csv(save_path, index=False)
    return history_df


def run_evaluation_and_plot(trainer: QCMDPSingleMGTrainer, env: IEEE33SingleMGEnv, mg_id: int, output_dir: str = "output_eval"):
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    obs = env.reset()

    print(f"[{output_dir}] running deterministic 96-step dispatch evaluation")
    for _ in range(96):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=trainer.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = trainer.actor.act(obs_tensor, deterministic=True)
        obs, _, done, _ = env.step(action.squeeze(0).cpu().numpy())
        if done:
            break

    history = getattr(env, "operation_history", [])
    if not history:
        print(f"[{output_dir}] operation_history is empty; skip plotting")
        return

    history_df, voltage_df, line_df, metrics_df = operation_history_to_dataframes(history)
    history_df.to_csv(output_dir_path / "operation_history.csv", index=False)
    voltage_df.to_csv(output_dir_path / "bus_voltage_timeseries.csv", index=False)
    line_df.to_csv(output_dir_path / "line_loading_timeseries.csv", index=False)
    metrics_df.to_csv(output_dir_path / "network_metrics_timeseries.csv", index=False)

    mg_name = f"MG{mg_id}"
    plot_power_balance(history_df, str(output_dir_path / "power_balance.png"), f"Power Balance ({mg_name})")
    plot_voltage_profile(voltage_df, str(output_dir_path / "voltage_profile_summary.png"), f"Voltage Profile ({mg_name})")
    plot_voltage_heatmap(voltage_df, str(output_dir_path / "voltage_heatmap.png"), f"Bus Voltage Heatmap ({mg_name})")
    plot_network_power_metrics(metrics_df, str(output_dir_path / "network_power_metrics.png"))
    plot_line_loading(line_df, str(output_dir_path / "line_loading_last_step.png"), f"Line Loading Snapshot ({mg_name})")

    print(f"[{output_dir}] evaluation artifacts saved")


def build_default_config(device: Optional[str] = None, updates: int = 1000) -> QCMDPConfig:
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return QCMDPConfig(device=resolved_device, updates=updates)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-microgrid Q-CMDP training entrypoint.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to the external dataset CSV.")
    parser.add_argument("--mg-id", type=int, default=1, help="Microgrid index in [1, 5].")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for weights, logs, and figures.")
    parser.add_argument("--config-json", type=str, default=None, help="Optional JSON file with QCMDPConfig overrides.")
    parser.add_argument("--updates", type=int, default=1000, help="Number of local updates.")
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cpu or cuda.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    data_path = resolve_required_data_path(args.data_path)
    mg_id = int(args.mg_id)
    output_dir = Path(args.output_dir or f"q_cmdp_single_mg{mg_id}_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    config_overrides = load_optional_json(args.config_json)
    config = QCMDPConfig.from_mapping(
        merge_overrides(
            build_default_config(device=args.device, updates=args.updates).to_dict(),
            config_overrides,
        )
    )

    print("=" * 50)
    print(f"Start Q-CMDP training (MG_ID={mg_id})")
    print(f"Device: {config.device}")
    print(f"Data path: {data_path}")
    print("=" * 50)

    env = IEEE33SingleMGEnv(data_path=data_path, mg_id=mg_id)
    trainer = QCMDPSingleMGTrainer(env, config)
    training_history = trainer.train()

    trainer.save_checkpoint(output_dir)
    plot_training_rewards(training_history, str(output_dir / "training_convergence.png"))
    save_training_history(training_history, output_dir / "training_convergence.csv")

    env.random_episode_start = False
    run_evaluation_and_plot(trainer, env, mg_id=mg_id, output_dir=str(output_dir))


if __name__ == "__main__":
    main()


QCMDP_SingleMG_Trainer = QCMDPSingleMGTrainer
