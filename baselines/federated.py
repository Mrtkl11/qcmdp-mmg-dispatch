from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .common import (
    constraint_cost,
    reset_observation,
    save_json,
    set_seed,
    step_environment,
)
from .config import load_algorithm_config, read_json, resolve_data_path, resolve_device
from .evaluation import run_evaluation_and_plot
from .runner import create_trainer, project_root


def aggregate_value(
    name: str, previous: Any, local_values: List[Any], momentum: float
) -> Any:
    if isinstance(previous, dict):
        return {
            key: aggregate_value(
                f"{name}.{key}",
                previous[key],
                [local_value[key] for local_value in local_values],
                momentum,
            )
            for key in previous
        }
    if torch.is_tensor(previous):
        previous_cpu = previous.detach().cpu()
        if not torch.is_floating_point(previous_cpu):
            return local_values[0].detach().cpu().clone()
        stacked = torch.stack([value.detach().cpu().float() for value in local_values])
        if "q_weights" in name:
            averaged = torch.atan2(stacked.sin().mean(0), stacked.cos().mean(0))
            difference = (averaged - previous_cpu.float() + np.pi) % (
                2.0 * np.pi
            ) - np.pi
            result = previous_cpu.float() + (1.0 - momentum) * difference
        else:
            averaged = stacked.mean(0)
            result = momentum * previous_cpu.float() + (1.0 - momentum) * averaged
        return result.to(previous_cpu.dtype)
    if isinstance(previous, (float, int)):
        averaged = float(np.mean([float(value) for value in local_values]))
        return momentum * float(previous) + (1.0 - momentum) * averaged
    return copy.deepcopy(local_values[0])


def aggregate_bundles(
    previous: Dict[str, Any], local_bundles: List[Dict[str, Any]], momentum: float
) -> Dict[str, Any]:
    if not local_bundles:
        raise ValueError("At least one local bundle is required")
    return {
        key: aggregate_value(
            key,
            value,
            [local_bundle[key] for local_bundle in local_bundles],
            momentum,
        )
        for key, value in previous.items()
    }


def active_clients_for_round(
    clients: List[int], outages: List[Dict[str, Any]], round_index: int
) -> List[int]:
    unavailable = {
        int(outage["client_id"])
        for outage in outages
        if int(outage["start_round"]) <= round_index <= int(outage["end_round"])
    }
    return [client for client in clients if client not in unavailable]


def evaluate_bundle(
    algorithm: str,
    data_path: str,
    clients: List[int],
    config: Dict[str, Any],
    bundle: Dict[str, Any],
    horizon: int,
    output_dir: Path | None = None,
) -> Dict[str, float]:
    rows: Dict[str, float] = {}
    rewards = []
    costs = []
    for mg_id in clients:
        trainer = create_trainer(algorithm, data_path, mg_id, config)
        trainer.load_state_dict_bundle(bundle)
        trainer.env.random_episode_start = False
        observation = reset_observation(trainer.env)
        reward_total = 0.0
        cost_total = 0.0
        for _ in range(int(horizon)):
            observation_tensor = torch.as_tensor(
                observation, dtype=torch.float32, device=trainer.device
            ).unsqueeze(0)
            with torch.no_grad():
                action, _ = trainer.actor.act(observation_tensor, deterministic=True)
            observation, reward, done, info = step_environment(
                trainer.env, action.squeeze(0).cpu().numpy()
            )
            reward_total += reward
            cost_total += constraint_cost(info)
            if done:
                observation = reset_observation(trainer.env)
        rewards.append(reward_total)
        costs.append(cost_total)
        rows[f"MG{mg_id}_Reward"] = reward_total
        rows[f"MG{mg_id}_ConstraintCost"] = cost_total
        if output_dir is not None:
            trainer.env.random_episode_start = False
            run_evaluation_and_plot(
                trainer, trainer.env, mg_id, str(output_dir / f"MG{mg_id}")
            )
    rows["Global_Avg_Reward"] = float(np.mean(rewards))
    rows["Global_Avg_ConstraintCost"] = float(np.mean(costs))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ring-federated training for comparison baselines"
    )
    parser.add_argument(
        "--algorithm",
        choices=("a2c", "ppo", "qppo", "constrained_ppo", "sac"),
        default="constrained_ppo",
    )
    parser.add_argument("--data-path", default=None)
    parser.add_argument(
        "--config-json",
        default=str(project_root() / "configs" / "baselines.template.json"),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--local-updates", type=int, default=None)
    parser.add_argument("--federated-config-json", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = read_json(args.config_json)
    config = load_algorithm_config(args.config_json, args.algorithm)
    if args.device:
        config["device"] = resolve_device(args.device)
    federated = (
        read_json(args.federated_config_json)
        if args.federated_config_json
        else payload.get("federated", {})
    )
    rounds = int(args.rounds if args.rounds is not None else federated["rounds"])
    local_updates = int(
        args.local_updates
        if args.local_updates is not None
        else federated["local_updates"]
    )
    momentum = float(federated["inter_round_momentum"])
    clients = [int(value) for value in federated["client_ids"]]
    outages = [dict(value) for value in federated["outages"]]
    topology_reconfiguration = bool(federated["topology_reconfiguration"])
    if not 0.0 <= momentum < 1.0:
        raise ValueError("inter_round_momentum must be in [0, 1)")
    config["updates"] = local_updates
    data_path = resolve_data_path(
        args.data_path, str(project_root() / "data" / "Environment_data_2018.csv")
    )
    output = Path(
        args.output_dir or project_root() / "results" / "federated" / args.algorithm
    )
    weights_dir = output / "weights"
    evaluation_dir = output / "evaluation"
    weights_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(config["seed"]))
    initial_trainer = create_trainer(args.algorithm, data_path, clients[0], config)
    global_bundle = initial_trainer.state_dict_bundle()
    history = []
    for round_index in range(1, rounds + 1):
        active_clients = active_clients_for_round(clients, outages, round_index)
        if not active_clients:
            raise RuntimeError(f"No active clients in federated round {round_index}")
        local_returns = []
        local_costs = []
        local_bundles = []
        route = []
        ring_interrupted = not topology_reconfiguration and len(active_clients) < len(
            clients
        )
        if not ring_interrupted:
            for mg_id in active_clients:
                trainer = create_trainer(args.algorithm, data_path, mg_id, config)
                trainer.load_state_dict_bundle(global_bundle)
                local_history = trainer.train()
                local_returns.append(
                    float(np.mean([row["episode_return"] for row in local_history]))
                )
                local_costs.append(
                    float(np.mean([row["episode_cost"] for row in local_history]))
                )
                route.append(f"MG{mg_id}")
                local_bundles.append(trainer.state_dict_bundle())
            global_bundle = aggregate_bundles(global_bundle, local_bundles, momentum)
        torch.save(global_bundle, weights_dir / f"round_{round_index:04d}.pt")
        final_plots = (
            evaluation_dir / f"round_{round_index:04d}"
            if round_index == rounds
            else None
        )
        metrics = evaluate_bundle(
            args.algorithm,
            data_path,
            clients,
            config,
            global_bundle,
            int(config["evaluation_horizon"]),
            final_plots,
        )
        metrics.update(
            {
                "Federated_Round": round_index,
                "Ring_Route": (
                    "disconnected"
                    if ring_interrupted
                    else " -> ".join(route + [route[0]])
                ),
                "Active_Client_Count": len(active_clients),
                "Local_Train_Return_Mean": (
                    float(np.mean(local_returns)) if local_returns else np.nan
                ),
                "Local_Constraint_Cost_Mean": (
                    float(np.mean(local_costs)) if local_costs else np.nan
                ),
            }
        )
        history.append(metrics)
        print(
            f"round={round_index}/{rounds} reward={metrics['Global_Avg_Reward']:.4f} cost={metrics['Global_Avg_ConstraintCost']:.4f}"
        )
    history_df = pd.DataFrame(history)
    history_df.to_csv(evaluation_dir / "convergence.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    x = history_df["Federated_Round"].to_numpy()
    axes[0].plot(
        x, history_df["Global_Avg_Reward"].to_numpy(), color="tab:blue", linewidth=2
    )
    axes[0].set_ylabel("Average reward")
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        x,
        history_df["Global_Avg_ConstraintCost"].to_numpy(),
        color="tab:red",
        linewidth=2,
    )
    axes[1].set_xlabel("Federated round")
    axes[1].set_ylabel("Average constraint cost")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(evaluation_dir / "convergence.png", dpi=220)
    plt.close(fig)
    resolved_federated = dict(federated)
    resolved_federated.update({"rounds": rounds, "local_updates": local_updates})
    save_json(
        output / "resolved_config.json",
        {
            "algorithm": args.algorithm,
            "trainer": config,
            "federated": resolved_federated,
        },
    )
    print(f"artifacts={output}")


if __name__ == "__main__":
    main()
