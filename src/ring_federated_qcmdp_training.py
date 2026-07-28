from __future__ import annotations

import copy
import io
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

try:
    from .ieee33_mmg_env import IEEE33SingleMGEnv
    from .qcmdp_single_mg_training import (
        QCMDPConfig,
        QCMDPSingleMGTrainer,
        run_evaluation_and_plot,
    )
    from .release_config import (
        load_json,
        require_fields,
        resolve_relative_path,
        resolve_required_data_path,
    )
except ImportError:
    from ieee33_mmg_env import IEEE33SingleMGEnv
    from qcmdp_single_mg_training import (
        QCMDPConfig,
        QCMDPSingleMGTrainer,
        run_evaluation_and_plot,
    )
    from release_config import (
        load_json,
        require_fields,
        resolve_relative_path,
        resolve_required_data_path,
    )


@dataclass(frozen=True)
class FederatedConfig:
    rounds: int
    local_updates: int
    client_ids: Sequence[int]
    inter_round_momentum: float
    seed: int
    outages: Sequence[Mapping[str, int]]
    topology_reconfiguration: bool
    data_path: str
    global_dir: str
    evaluation_dir: str
    trainer_config: QCMDPConfig
    environment_config: Mapping[str, object]


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
        key: (
            value.detach().cpu().clone()
            if torch.is_tensor(value)
            else copy.deepcopy(value)
        )
        for key, value in state_dict.items()
    }


class RingFederatedAggregator:
    def __init__(self, inter_round_momentum: float):
        if not 0.0 <= inter_round_momentum < 1.0:
            raise ValueError("inter_round_momentum must be in [0, 1)")
        self.momentum = float(inter_round_momentum)

    def _mean_parameter(
        self, parameter_name: str, values: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        reference = values[0].detach().cpu()
        if not torch.is_floating_point(reference):
            return reference.clone()
        stacked = torch.stack([value.detach().cpu().float() for value in values])
        if "q_weights" in parameter_name:
            return torch.atan2(stacked.sin().mean(0), stacked.cos().mean(0)).to(
                reference.dtype
            )
        return stacked.mean(0).to(reference.dtype)

    def _apply_momentum(
        self,
        parameter_name: str,
        previous: torch.Tensor,
        averaged: torch.Tensor,
    ) -> torch.Tensor:
        previous_cpu = previous.detach().cpu()
        if not torch.is_floating_point(previous_cpu):
            return averaged.detach().cpu().clone()
        previous_float = previous_cpu.float()
        averaged_float = averaged.detach().cpu().float()
        if "q_weights" in parameter_name:
            difference = (averaged_float - previous_float + np.pi) % (
                2.0 * np.pi
            ) - np.pi
            result = previous_float + (1.0 - self.momentum) * difference
        else:
            result = (
                self.momentum * previous_float + (1.0 - self.momentum) * averaged_float
            )
        return result.to(previous_cpu.dtype)

    def aggregate_state_dicts(
        self,
        previous_state: Mapping[str, torch.Tensor],
        ordered_states: Sequence[Mapping[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        if not ordered_states:
            raise ValueError("ordered_states must not be empty")
        return {
            key: self._apply_momentum(
                key,
                previous_state[key],
                self._mean_parameter(
                    key, [local_state[key] for local_state in ordered_states]
                ),
            )
            for key in previous_state
        }

    def aggregate_scalar(
        self, previous_value: float, ordered_values: Sequence[float]
    ) -> float:
        if not ordered_values:
            raise ValueError("ordered_values must not be empty")
        averaged = float(np.mean(ordered_values))
        return self.momentum * float(previous_value) + (1.0 - self.momentum) * averaged


class RingFederatedTrainer:
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.aggregator = RingFederatedAggregator(config.inter_round_momentum)
        self.script_dir = str(Path(config.data_path).resolve().parent)
        self.global_dir = Path(config.global_dir)
        self.evaluation_dir = Path(config.evaluation_dir)
        self.global_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        self.round_history: List[FederatedRoundStats] = []

        self.trainer_config = config.trainer_config
        self.client_ids = list(config.client_ids)

        _, init_trainer = self.make_trainer(self.client_ids[0])
        self.global_actor = clone_state_dict(init_trainer.actor.state_dict())
        self.global_reward_critic = clone_state_dict(
            init_trainer.reward_critic.state_dict()
        )
        self.global_safety_critic = clone_state_dict(
            init_trainer.safety_critic.state_dict()
        )
        self.global_lagrange_multiplier = float(self.trainer_config.initial_lambda)

    def active_client_ids(self, federated_round: int) -> List[int]:
        inactive = {
            int(outage["client_id"])
            for outage in self.config.outages
            if int(outage["start_round"]) <= federated_round <= int(outage["end_round"])
        }
        return [client_id for client_id in self.client_ids if client_id not in inactive]

    def make_trainer(self, mg_id: int):
        env = IEEE33SingleMGEnv(
            data_path=self.config.data_path,
            config=dict(self.config.environment_config),
            mg_id=mg_id,
        )
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

        last_return = (
            float(np.mean([item.episode_return for item in history]))
            if history
            else np.nan
        )
        last_cost = (
            float(np.mean([item.episode_constraint_cost for item in history]))
            if history
            else np.nan
        )
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

        self.global_actor = self.aggregator.aggregate_state_dicts(
            self.global_actor,
            [
                result.actor_state
                for result in active_results
                if result.actor_state is not None
            ],
        )
        self.global_reward_critic = self.aggregator.aggregate_state_dicts(
            self.global_reward_critic,
            [
                result.reward_critic_state
                for result in active_results
                if result.reward_critic_state is not None
            ],
        )
        self.global_safety_critic = self.aggregator.aggregate_state_dicts(
            self.global_safety_critic,
            [
                result.safety_critic_state
                for result in active_results
                if result.safety_critic_state is not None
            ],
        )
        self.global_lagrange_multiplier = self.aggregator.aggregate_scalar(
            self.global_lagrange_multiplier,
            [result.lagrange_multiplier for result in active_results],
        )

    def save_global_checkpoint(self, federated_round: int) -> None:
        torch.save(
            self.global_actor,
            self.global_dir / f"global_actor_round_{federated_round}.pt",
        )
        torch.save(
            self.global_reward_critic,
            self.global_dir / f"global_reward_critic_round_{federated_round}.pt",
        )
        torch.save(
            self.global_safety_critic,
            self.global_dir / f"global_safety_critic_round_{federated_round}.pt",
        )
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

            for _ in range(env_eval.episode_horizon):
                obs_tensor = torch.tensor(
                    obs, dtype=torch.float32, device=self.trainer_config.device
                ).unsqueeze(0)
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
                run_evaluation_and_plot(
                    trainer_eval, env_eval, mg_id=mg_id, output_dir=str(output_dir)
                )

        return float(np.mean(rewards)), float(np.mean(constraint_costs)), per_mg_metrics

    def run(self) -> List[FederatedRoundStats]:
        self.round_history = []
        for federated_round in range(1, self.config.rounds + 1):
            print(
                f"\n{'=' * 20} Federated round {federated_round}/{self.config.rounds} {'=' * 20}"
            )
            local_results: List[LocalClientResult] = []
            active_ids = set(self.active_client_ids(federated_round))

            if not self.config.topology_reconfiguration and len(active_ids) < len(
                self.client_ids
            ):
                print("  [Aggregate] ring interrupted; global state remains unchanged")
                self.save_global_checkpoint(federated_round)
                avg_reward, avg_constraint_cost, per_mg_metrics = (
                    self.evaluate_global_policy(
                        make_final_plots=(federated_round == self.config.rounds)
                    )
                )
                self.round_history.append(
                    FederatedRoundStats(
                        federated_round=federated_round,
                        global_avg_reward=avg_reward,
                        global_avg_constraint_cost=avg_constraint_cost,
                        local_train_return_mean=np.nan,
                        local_constraint_cost_mean=np.nan,
                        global_lagrange_multiplier=self.global_lagrange_multiplier,
                        active_client_count=len(active_ids),
                        per_mg_metrics=per_mg_metrics,
                    )
                )
                continue

            for mg_id in self.client_ids:
                if mg_id not in active_ids:
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

            print("  [Aggregate] equal-weight ring aggregation over active clients")
            self.aggregate_round(local_results)
            self.save_global_checkpoint(federated_round)

            avg_reward, avg_constraint_cost, per_mg_metrics = (
                self.evaluate_global_policy(
                    make_final_plots=(federated_round == self.config.rounds)
                )
            )

            active_results = [result for result in local_results if result.is_active]
            round_stats = FederatedRoundStats(
                federated_round=federated_round,
                global_avg_reward=avg_reward,
                global_avg_constraint_cost=avg_constraint_cost,
                local_train_return_mean=float(
                    np.nanmean([result.mean_return for result in active_results])
                ),
                local_constraint_cost_mean=float(
                    np.nanmean(
                        [result.mean_constraint_cost for result in active_results]
                    )
                ),
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
        history_df = pd.DataFrame(
            [round_stats.to_row() for round_stats in self.round_history]
        )
        history_df.to_csv(
            self.evaluation_dir / "ring_federated_qcmdp_convergence.csv", index=False
        )
        return history_df

    def plot_round_history(self, history_df: pd.DataFrame) -> None:
        reward_fig_path = (
            self.evaluation_dir / "ring_federated_qcmdp_reward_convergence.png"
        )
        cost_fig_path = (
            self.evaluation_dir / "ring_federated_qcmdp_constraint_cost_convergence.png"
        )

        plt.figure(figsize=(8, 5))
        plt.plot(
            history_df["Federated_Round"],
            history_df["Global_Avg_Reward"],
            marker="o",
            color="tab:red",
            linewidth=2,
        )
        plt.title("Ring-Federated Q-CMDP Reward Convergence")
        plt.xlabel("Communication Rounds")
        plt.ylabel("Average Episode Return")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(reward_fig_path, dpi=300)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(
            history_df["Federated_Round"],
            history_df["Global_Avg_ConstraintCost"],
            marker="o",
            color="tab:blue",
            linewidth=2,
        )
        plt.title("Ring-Federated Q-CMDP Constraint-Cost Convergence")
        plt.xlabel("Communication Rounds")
        plt.ylabel("Average Constraint Cost")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(cost_fig_path, dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Ring-federated Q-CMDP training entrypoint."
    )
    parser.add_argument(
        "--data-path", type=str, default=None, help="Path to the external dataset CSV."
    )
    parser.add_argument(
        "--config-json",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "experiment_config.template.json"
        ),
        help="JSON file containing environment, trainer, and federation configuration.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--federated-config-json", type=str, default=None)
    args = parser.parse_args()

    data_path = resolve_required_data_path(args.data_path, "QCMDP_DATA_PATH")
    payload = load_json(args.config_json)
    require_fields(
        payload, {"environment_config", "training", "federated"}, "experiment"
    )
    training_values = dict(payload["training"])
    federated_values = (
        load_json(args.federated_config_json)
        if args.federated_config_json
        else dict(payload["federated"])
    )
    require_fields(
        federated_values,
        {
            "rounds",
            "local_updates",
            "client_ids",
            "inter_round_momentum",
            "seed",
            "outages",
            "topology_reconfiguration",
        },
        "federated",
    )
    training_values["updates"] = int(federated_values["local_updates"])
    if training_values.get("device") == "auto":
        training_values["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    trainer_config = QCMDPConfig.from_mapping(training_values)
    environment_config = load_json(
        resolve_relative_path(args.config_json, str(payload["environment_config"]))
    )
    output_root = Path(
        args.output_dir
        or Path(__file__).resolve().parents[1] / "results" / "federated" / "qcmdp"
    )
    config = FederatedConfig(
        rounds=int(federated_values["rounds"]),
        local_updates=int(federated_values["local_updates"]),
        client_ids=tuple(int(value) for value in federated_values["client_ids"]),
        inter_round_momentum=float(federated_values["inter_round_momentum"]),
        seed=int(federated_values["seed"]),
        outages=tuple(dict(value) for value in federated_values["outages"]),
        topology_reconfiguration=bool(federated_values["topology_reconfiguration"]),
        data_path=data_path,
        global_dir=str(output_root / "weights"),
        evaluation_dir=str(output_root / "evaluation"),
        trainer_config=trainer_config,
        environment_config=environment_config,
    )
    set_seed(config.seed)

    trainer = RingFederatedTrainer(config)
    print("=" * 72)
    print("Start ring-federated Q-CMDP training for multi-microgrid dispatch")
    print(f"Rounds: {config.rounds}, local updates: {config.local_updates}")
    print(
        f"Inter-round momentum: {config.inter_round_momentum}, clients: {config.client_ids}"
    )
    print(f"Data path: {config.data_path}")
    print("=" * 72)

    trainer.run()
    history_df = trainer.save_round_history()
    trainer.plot_round_history(history_df)

    print("\nRing-federated Q-CMDP training completed.")
    print(
        f"  - Convergence CSV: {trainer.evaluation_dir / 'ring_federated_qcmdp_convergence.csv'}"
    )
    print(
        f"  - Reward plot: {trainer.evaluation_dir / 'ring_federated_qcmdp_reward_convergence.png'}"
    )
    print(
        f"  - Constraint-cost plot: {trainer.evaluation_dir / 'ring_federated_qcmdp_constraint_cost_convergence.png'}"
    )
    print(f"  - Final physical plots: {trainer.evaluation_dir / 'Final_MG*_plots'}")


if __name__ == "__main__":
    main()
