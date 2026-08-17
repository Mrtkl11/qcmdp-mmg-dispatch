from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Type

import torch

from src.ieee33_mmg_env import IEEE33SingleMGEnv
from .common import save_json, save_training_history, set_seed
from .config import load_algorithm_config, merge_cli_overrides, resolve_data_path
from .evaluation import run_evaluation_and_plot
from .on_policy import A2CTrainer, ConstrainedPPOTrainer, PPOTrainer
from .off_policy import SACTrainer
from .quantum_on_policy import QPPOTrainer


TRAINERS: Dict[str, Type[Any]] = {
    "a2c": A2CTrainer,
    "ppo": PPOTrainer,
    "qppo": QPPOTrainer,
    "constrained_ppo": ConstrainedPPOTrainer,
    "sac": SACTrainer,
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_parser(default_algorithm: Optional[str] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Comparison baseline training for IEEE 33-bus multi-microgrid dispatch"
    )
    if default_algorithm is None:
        parser.add_argument("--algorithm", choices=sorted(TRAINERS), default="ppo")
    else:
        parser.set_defaults(algorithm=default_algorithm)
    parser.add_argument("--data-path", default=None)
    parser.add_argument(
        "--config-json",
        default=str(project_root() / "configs" / "baselines.template.json"),
    )
    parser.add_argument("--mg-id", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--updates", type=int, default=None)
    return parser


def create_trainer(
    algorithm: str, data_path: str, mg_id: int, config: Dict[str, Any]
) -> Any:
    environment_config = config.get("environment", {})
    env = IEEE33SingleMGEnv(data_path=data_path, config=environment_config, mg_id=mg_id)
    trainer_config = dict(config)
    trainer_config.pop("environment", None)
    return TRAINERS[algorithm](env, trainer_config)


def run(
    algorithm: str, data_path: str, mg_id: int, output_dir: str, config: Dict[str, Any]
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    set_seed(int(config["seed"]))
    trainer = create_trainer(algorithm, data_path, mg_id, config)
    history = trainer.train()
    torch.save(trainer.state_dict_bundle(), output / "model.pt")
    save_training_history(output / "training_history.csv", history)
    save_json(output / "resolved_config.json", config)
    plot_path = output / "training_convergence.png"
    import matplotlib.pyplot as plt
    import numpy as np

    returns = [float(row["episode_return"]) for row in history]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(returns) + 1), returns, color="tab:blue", linewidth=1.8)
    ax.set_xlabel("Update")
    ax.set_ylabel("Episode return")
    ax.set_title(f"{algorithm} training convergence")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)
    trainer.env.random_episode_start = False
    run_evaluation_and_plot(
        trainer, trainer.env, mg_id=mg_id, output_dir=str(output / "evaluation")
    )
    return output


def main(default_algorithm: Optional[str] = None) -> None:
    args = build_parser(default_algorithm).parse_args()
    config = load_algorithm_config(args.config_json, args.algorithm)
    config = merge_cli_overrides(config, args.device, args.updates)
    data_path = resolve_data_path(
        args.data_path, str(project_root() / "data" / "Environment_data_2018.csv")
    )
    output_dir = args.output_dir or str(
        project_root() / "results" / args.algorithm / f"mg{args.mg_id}"
    )
    print(
        f"algorithm={args.algorithm} mg_id={args.mg_id} device={config['device']} data={data_path}"
    )
    result = run(args.algorithm, data_path, args.mg_id, output_dir, config)
    print(f"artifacts={result}")


if __name__ == "__main__":
    main()
