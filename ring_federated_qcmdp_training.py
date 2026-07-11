from __future__ import annotations

import copy
import io
import os
import random
import sys
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ieee33_mmg_env import IEEE33SingleMGEnv
from qcmdp_single_mg_training import (
    QCMDPConfig,
    QCMDPSingleMGTrainer,
    build_default_config,
    run_evaluation_and_plot,
)
from release_config import load_optional_json, merge_overrides, resolve_required_data_path


@dataclass(frozen=True)
class FederatedConfig:
    rounds: int = 50
    local_updates: int = 5
    client_ids: Sequence[int] = (1, 2, 3, 4, 5)
    active_indicators: Sequence[int] = (1, 1, 1, 1, 1)
    ring_alpha: float = 0.2
    seed: int = 2026
    data_path: str = "Environment_data_2018.csv"
    global_dir: str = "global_ring_qcmdp_weights"
    evaluation_dir: str = "global_evaluation_results"
    trainer_config: Optional[QCMDPConfig] = None

    @classmethod
    def from_environment(cls, script_dir: str, data_path: str, trainer_overrides: Optional[Mapping[str, object]] = None) -> "FederatedConfig":
        rounds = int(os.getenv("FL_ROUNDS", "50"))
        local_updates = int(os.getenv("LOCAL_UPDATES", "5"))
        active_indicators = tuple(int(value) for value in os.getenv("ACTIVE_INDICATORS", "1,1,1,1,1").split(","))
        ring_alpha = float(os.getenv("RING_ALPHA", "0.2"))
        seed = int(os.getenv("FED_SEED", "2026"))
        trainer_config = QCMDPConfig.from_mapping(
            merge_overrides(build_default_config(device="cpu", updates=local_updates).to_dict(), trainer_overrides)
        )
        return cls(
            rounds=rounds,
            local_updates=local_updates,
            active_indicators=active_indicators,
            ring_alpha=ring_alpha,
            seed=seed,
            data_path=data_path,
            global_dir=os.path.join(script_dir, "global_ring_qcmdp_weights"),
            evaluation_dir=os.path.join(script_dir, "global_evaluation_results"),
            trainer_config=trainer_config,
        )


@dataclass
class LocalClientResult:
    mg_id: int
    is_active: bool
    mean_return: float
    mean_constraint_cost: float
    lagrange_multiplier: float
    actor_state: Optional[Dict[str, torch.Tensor]]
    reward_critic_state: Optional[Dict[str, torch.Tensor]]
    safety_critic_state: Optional[Dict[str, torch.Tensor]]


@dataclass
class FederatedRoundStats:
    federated_round: int
    global_avg_reward: float
    global_avg_constraint_cost: float
    local_train_return_mean: float
    local_constraint_cost_mean: float
    global_lagrange_multiplier: float
    active_client_count: int
    per_mg_metrics: Dict[str, float]

    def to_row(self) -> Dict[str, float]:
        base = {
            "Federated_Round": self.federated_round,
            "Global_Avg_Reward": self.global_avg_reward,
            "Global_Avg_ConstraintCost": self.global_avg_constraint_cost,
            "Local_Train_Return_Mean": self.local_train_return_mean,
            "Local_Constraint_Cost_Mean": self.local_constraint_cost_mean,
            "Global_LagrangeMultiplier": self.global_lagrange_multiplier,
            "Active_Client_Count": self.active_client_count,
        }
        base.update(self.per_mg_metrics)
        return base


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clone_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in state_dict.items()
    }


class RingFederatedAggregator:
    def __init__(self, ring_alpha: float):
        if not 0.0 < ring_alpha < 1.0:
            raise ValueError("ring_alpha must be in (0, 1)")
        self.ring_alpha = float(ring_alpha)

    def _mix_parameter(self, parameter_name: str, accumulated: torch.Tensor, local_value: torch.Tensor) -> torch.Tensor:
        accumulated_cpu = accumulated.detach().cpu()
        local_cpu = local_value.detach().cpu()
        if torch.is_floating_point(accumulated_cpu) and "q_weights" in parameter_name:
            phase_diff = (local_cpu.float() - accumulated_cpu.float() + np.pi) % (2.0 * np.pi) - np.pi
            mixed = accumulated_cpu.float() + (1.0 - self.ring_alpha) * phase_diff
            return mixed.to(dtype=accumulated_cpu.dtype)
        if torch.is_floating_point(accumulated_cpu):
            mixed = self.ring_alpha * accumulated_cpu.float() + (1.0 - self.ring_alpha) * local_cpu.float()
            return mixed.to(dtype=accumulated_cpu.dtype)
        return accumulated_cpu.clone()

    def aggregate_state_dicts(self, ordered_states: Sequence[Mapping[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not ordered_states:
            raise ValueError("ordered_states must not be empty")
        aggregated = clone_state_dict(ordered_states[0])
        for local_state in ordered_states[1:]:
            for key in aggregated.keys():
                aggregated[key] = self._mix_parameter(key, aggregated[key], local_state[key])
        return aggregated

    def aggregate_scalar(self, ordered_values: Sequence[float]) -> float:
        if not ordered_values:
            raise ValueError("ordered_values must not be empty")
        aggregated = float(ordered_values[0])
        for value in ordered_values[1:]:
            aggregated = self.ring_alpha * aggregated + (1.0 - self.ring_alpha) * float(value)
        return float(aggregated)


class RingFederatedTrainer:
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.aggregator = RingFederatedAggregator(config.ring_alpha)
        self.script_dir = str(Path(config.data_path).resolve().parent)
        self.global_dir = Path(config.global_dir)
        self.evaluation_dir = Path(config.evaluation_dir)
        self.global_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        self.round_history: List[FederatedRoundStats] = []

        if config.trainer_config is None:
            raise ValueError("trainer_config must be provided")

        self.trainer_config = config.trainer_config
        self.client_ids = list(config.client_ids)
        self.active_indicators = list(config.active_indicators)
        if len(self.client_ids) != len(self.active_indicators):
            raise ValueError("client_ids and active_indicators must have the same length")

        _, init_trainer = self.make_trainer(self.client_ids[0])
        self.global_actor = clone_state_dict(init_trainer.actor.state_dict())
        self.global_reward_critic = clone_state_dict(init_trainer.reward_critic.state_dict())
        self.global_safety_critic = clone_state_dict(init_trainer.safety_critic.state_dict())
        self.global_lagrange_multiplier = float(self.trainer_config.initial_lambda)

    def active_client_ids(self) -> List[int]:
        return [client_id for client_id, is_active in zip(self.client_ids, self.active_indicators) if int(is_active) == 1]

    def make_trainer(self, mg_id: int):
        env = IEEE33SingleMGEnv(data_path=self.config.data_path, mg_id=mg_id)
        trainer = QCMDPSingleMGTrainer(env, self.trainer_config)
        return env, trainer

    def load_global_state(self, trainer: QCMDPSingleMGTrainer) -> None:
        trainer.actor.load_state_dict(self.global_actor)
        trainer.reward_critic.load_state_dict(self.global_reward_critic)
        trainer.safety_critic.load_state_dict(self.global_safety_critic)
        trainer.lagrange_multiplier = float(self.global_lagrange_multiplier)

    def train_one_client(self, mg_id: int, is_active: bool) -> LocalClientResult:
        if not is_active:
            return LocalClientResult(
                mg_id=mg_id,
                is_active=False,
                mean_return=np.nan,
                mean_constraint_cost=np.nan,
                lagrange_multiplier=np.nan,
                actor_state=None,
                reward_critic_state=None,
                safety_critic_state=None,
            )

        _, trainer = self.make_trainer(mg_id)
        self.load_global_state(trainer)
        original_stdout = sys.stdout
        try:
            sys.stdout = io.StringIO()
            history = trainer.train()
        finally:
            sys.stdout = original_stdout

        last_return = float(np.mean([item.episode_return for item in history])) if history else np.nan
        last_cost = float(np.mean([item.episode_constraint_cost for item in history])) if history else np.nan
        return LocalClientResult(
            mg_id=mg_id,
            is_active=True,
            mean_return=last_return,
            mean_constraint_cost=last_cost,
            lagrange_multiplier=float(trainer.lagrange_multiplier),
            actor_state=clone_state_dict(trainer.actor.state_dict()),
            reward_critic_state=clone_state_dict(trainer.reward_critic.state_dict()),
            safety_critic_state=clone_state_dict(trainer.safety_critic.state_dict()),
        )

    def aggregate_round(self, local_results: Sequence[LocalClientResult]) -> None:
        active_results = [result for result in local_results if result.is_active]
        if not active_results:
            raise RuntimeError("No active clients available for ring aggregation")

        self.global_actor = self.aggregator.aggregate_state_dicts([result.actor_state for result in active_results if result.actor_state is not None])
        self.global_reward_critic = self.aggregator.aggregate_state_dicts([result.reward_critic_state for result in active_results if result.reward_critic_state is not None])
        self.global_safety_critic = self.aggregator.aggregate_state_dicts([result.safety_critic_state for result in active_results if result.safety_critic_state is not None])
        self.global_lagrange_multiplier = self.aggregator.aggregate_scalar([result.lagrange_multiplier for result in active_results])

    def save_global_checkpoint(self, federated_round: int) -> None:
        torch.save(self.global_actor, self.global_dir / f"global_actor_round_{federated_round}.pt")
        torch.save(self.global_reward_critic, self.global_dir / f"global_reward_critic_round_{federated_round}.pt")
        torch.save(self.global_safety_critic, self.global_dir / f"global_safety_critic_round_{federated_round}.pt")
        torch.save(
            {
                "lagrange_multiplier": self.global_lagrange_multiplier,
                "federated_config": asdict(self.config),
                "trainer_config": self.trainer_config.to_dict(),
            },
            self.global_dir / f"global_lambda_round_{federated_round}.pt",
        )

    def evaluate_global_policy(self, make_final_plots: bool = False):
        rewards: List[float] = []
        constraint_costs: List[float] = []
        per_mg_metrics: Dict[str, float] = {}

        for mg_id in self.client_ids:
            env_eval, trainer_eval = self.make_trainer(mg_id)
            trainer_eval.actor.load_state_dict(self.global_actor)
            trainer_eval.reward_critic.load_state_dict(self.global_reward_critic)
            trainer_eval.safety_critic.load_state_dict(self.global_safety_critic)
            trainer_eval.lagrange_multiplier = float(self.global_lagrange_multiplier)

            env_eval.random_episode_start = False
            obs = env_eval.reset()
            ep_reward = 0.0
            ep_constraint_cost = 0.0

            for _ in range(getattr(env_eval, "episode_horizon", 96)):
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.trainer_config.device).unsqueeze(0)
                with torch.no_grad():
                    action, _ = trainer_eval.actor.act(obs_tensor, deterministic=True)
                obs, reward, done, info = env_eval.step(action.squeeze(0).cpu().numpy())
                ep_reward += float(reward)
                ep_constraint_cost += float(info.get("constraint_cost", 0.0))
                if done:
                    break

            rewards.append(ep_reward)
            constraint_costs.append(ep_constraint_cost)
            per_mg_metrics[f"MG{mg_id}_Reward"] = ep_reward
            per_mg_metrics[f"MG{mg_id}_ConstraintCost"] = ep_constraint_cost

            if make_final_plots:
                output_dir = self.evaluation_dir / f"Final_MG{mg_id}_plots"
                run_evaluation_and_plot(trainer_eval, env_eval, mg_id=mg_id, output_dir=str(output_dir))

        return float(np.mean(rewards)), float(np.mean(constraint_costs)), per_mg_metrics

    def run(self) -> List[FederatedRoundStats]:
        self.round_history = []
        for federated_round in range(1, self.config.rounds + 1):
            print(f"\n{'=' * 20} Federated round {federated_round}/{self.config.rounds} {'=' * 20}")
            local_results: List[LocalClientResult] = []

            for mg_id, is_active in zip(self.client_ids, self.active_indicators):
                if int(is_active) != 1:
                    print(f"  [Local] MG{mg_id}: offline, bypass in ring aggregation")
                    local_results.append(self.train_one_client(mg_id, False))
                    continue
                print(f"  [Local] MG{mg_id}: load global model and train ...", end=" ")
                result = self.train_one_client(mg_id, True)
                local_results.append(result)
                print(
                    f"done, local mean return={result.mean_return:.3f}, "
                    f"local mean constraint cost={result.mean_constraint_cost:.3f}"
                )

            print("  [Aggregate] sequential ring aggregation over active clients")
            self.aggregate_round(local_results)
            self.save_global_checkpoint(federated_round)

            avg_reward, avg_constraint_cost, per_mg_metrics = self.evaluate_global_policy(
                make_final_plots=(federated_round == self.config.rounds)
            )

            active_results = [result for result in local_results if result.is_active]
            round_stats = FederatedRoundStats(
                federated_round=federated_round,
                global_avg_reward=avg_reward,
                global_avg_constraint_cost=avg_constraint_cost,
                local_train_return_mean=float(np.nanmean([result.mean_return for result in active_results])),
                local_constraint_cost_mean=float(np.nanmean([result.mean_constraint_cost for result in active_results])),
                global_lagrange_multiplier=self.global_lagrange_multiplier,
                active_client_count=len(active_results),
                per_mg_metrics=per_mg_metrics,
            )
            self.round_history.append(round_stats)
            print(
                f"  >> Round {federated_round}: eval avg reward={avg_reward:.3f}, "
                f"avg constraint cost={avg_constraint_cost:.3f}, "
                f"lambda={self.global_lagrange_multiplier:.4f}"
            )
        return self.round_history

    def save_round_history(self) -> pd.DataFrame:
        history_df = pd.DataFrame([round_stats.to_row() for round_stats in self.round_history])
        history_df.to_csv(self.evaluation_dir / "ring_federated_qcmdp_convergence.csv", index=False)
        return history_df

    def plot_round_history(self, history_df: pd.DataFrame) -> None:
        reward_fig_path = self.evaluation_dir / "ring_federated_qcmdp_reward_convergence.png"
        cost_fig_path = self.evaluation_dir / "ring_federated_qcmdp_constraint_cost_convergence.png"

        plt.figure(figsize=(8, 5))
        plt.plot(history_df["Federated_Round"], history_df["Global_Avg_Reward"], marker="o", color="tab:red", linewidth=2)
        plt.title("Ring-Federated Q-CMDP Reward Convergence")
        plt.xlabel("Communication Rounds")
        plt.ylabel("Average Episode Return")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(reward_fig_path, dpi=300)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(history_df["Federated_Round"], history_df["Global_Avg_ConstraintCost"], marker="o", color="tab:blue", linewidth=2)
        plt.title("Ring-Federated Q-CMDP Constraint-Cost Convergence")
        plt.xlabel("Communication Rounds")
        plt.ylabel("Average Constraint Cost")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(cost_fig_path, dpi=300)
        plt.close()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Ring-federated Q-CMDP training entrypoint.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to the external dataset CSV.")
    parser.add_argument("--trainer-config-json", type=str, default=None, help="Optional JSON file with QCMDPConfig overrides.")
    args = parser.parse_args()

    data_path = resolve_required_data_path(args.data_path)
    trainer_overrides = load_optional_json(args.trainer_config_json)
    config = FederatedConfig.from_environment(script_dir, data_path=data_path, trainer_overrides=trainer_overrides)
    set_seed(config.seed)

    trainer = RingFederatedTrainer(config)
    print("=" * 72)
    print("Start ring-federated Q-CMDP training for multi-microgrid dispatch")
    print(f"Rounds: {config.rounds}, local updates: {config.local_updates}")
    print(f"Ring mixing coefficient: {config.ring_alpha}, active clients: {trainer.active_client_ids()}")
    print(f"Data path: {config.data_path}")
    print("=" * 72)

    trainer.run()
    history_df = trainer.save_round_history()
    trainer.plot_round_history(history_df)

    print("\nRing-federated Q-CMDP training completed.")
    print(f"  - Convergence CSV: {trainer.evaluation_dir / 'ring_federated_qcmdp_convergence.csv'}")
    print(f"  - Reward plot: {trainer.evaluation_dir / 'ring_federated_qcmdp_reward_convergence.png'}")
    print(f"  - Constraint-cost plot: {trainer.evaluation_dir / 'ring_federated_qcmdp_constraint_cost_convergence.png'}")
    print(f"  - Final physical plots: {trainer.evaluation_dir / 'Final_MG*_plots'}")


if __name__ == "__main__":
    main()
