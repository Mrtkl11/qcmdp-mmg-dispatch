from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from src.ieee33_mmg_env import IEEE33SingleMGEnv
from src.qcmdp_single_mg_training import QCMDPConfig, QCMDPSingleMGTrainer
from src.release_config import load_json, require_fields, resolve_relative_path


def resolve_path(config_path: str, value: str) -> Path:
    return Path(resolve_relative_path(config_path, value))


def collect_extreme_load_states(
    environment: IEEE33SingleMGEnv, rollout_steps: int
) -> Dict[str, Dict[str, Any]]:
    observation = environment.reset()
    records = []
    for _ in range(rollout_steps):
        _, _, load, _ = environment._zone_inputs()
        records.append(
            {
                "observation": np.asarray(observation, dtype=np.float32),
                "load_kw": float(load),
                "timestamp": str(environment.data.index[environment.current_step]),
            }
        )
        observation, _, done, _ = environment.step(
            np.zeros(environment.action_space.shape, dtype=np.float32)
        )
        if done:
            break
    if not records:
        raise RuntimeError("No environment states were collected")
    loads = np.asarray([record["load_kw"] for record in records])
    return {
        "minimum_load": records[int(loads.argmin())],
        "maximum_load": records[int(loads.argmax())],
    }


def tensor_to_list(value: torch.Tensor) -> Any:
    return value.detach().cpu().numpy().tolist()


def critic_payload(critic, states: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    observations = torch.as_tensor(
        np.stack([state["observation"] for state in states.values()]),
        dtype=torch.float32,
        device=critic.q_weights.device,
    )
    with torch.no_grad():
        encoded = torch.atan(critic.state_encoder(observations))
        expectation_rows = []
        for encoded_state in encoded:
            output = critic.qnode(encoded_state, critic.q_weights)
            if isinstance(output, (list, tuple)):
                output = torch.stack(output)
            expectation_rows.append(output.to(dtype=critic.q_weights.dtype))
        expectations = torch.stack(expectation_rows)
        values = critic.value_head(expectations).squeeze(-1)
    return {
        "encoded_angles": tensor_to_list(encoded),
        "variational_angles": tensor_to_list(critic.q_weights),
        "local_expectations": tensor_to_list(expectations),
        "local_values": tensor_to_list(values),
        "value_head": {
            key: tensor_to_list(value)
            for key, value in critic.value_head.state_dict().items()
        },
    }


def build_payload(config_path: str) -> Dict[str, Any]:
    hardware = load_json(config_path)
    require_fields(
        hardware,
        {
            "experiment_config",
            "data_path",
            "checkpoint_dir",
            "mg_id",
            "rollout_steps",
            "shots",
            "repetitions",
            "payload_path",
        },
        "hardware evaluation",
    )
    experiment_path = resolve_path(config_path, hardware["experiment_config"])
    experiment = load_json(str(experiment_path))
    training = dict(experiment["training"])
    training["device"] = "cpu"
    environment_config = load_json(
        resolve_relative_path(
            str(experiment_path), str(experiment["environment_config"])
        )
    )
    environment = IEEE33SingleMGEnv(
        str(resolve_path(config_path, hardware["data_path"])),
        environment_config,
        int(hardware["mg_id"]),
    )
    trainer = QCMDPSingleMGTrainer(environment, QCMDPConfig.from_mapping(training))
    checkpoint_dir = resolve_path(config_path, hardware["checkpoint_dir"])
    trainer.reward_critic.load_state_dict(
        torch.load(
            checkpoint_dir / "reward_critic_weights.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    trainer.safety_critic.load_state_dict(
        torch.load(
            checkpoint_dir / "safety_critic_weights.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    trainer.reward_critic.eval()
    trainer.safety_critic.eval()
    states = collect_extreme_load_states(environment, int(hardware["rollout_steps"]))
    labels = list(states)
    return {
        "schema_version": 1,
        "shots": int(hardware["shots"]),
        "repetitions": int(hardware["repetitions"]),
        "n_qubits": trainer.config.n_qubits,
        "n_layers": trainer.config.n_layers,
        "state_labels": labels,
        "states": {
            label: {
                "timestamp": states[label]["timestamp"],
                "load_kw": states[label]["load_kw"],
            }
            for label in labels
        },
        "circuit": {
            "encoding": ["RX", "RY", "RZ"],
            "variational": ["RY", "RZ"],
            "entangling": "nearest-neighbor CZ cascade",
            "measurement": "PauliZ on every qubit",
        },
        "critics": {
            "reward": critic_payload(trainer.reward_critic, states),
            "safety": critic_payload(trainer.safety_critic, states),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export paper-aligned dual-VQC payloads for authorized QPU execution"
    )
    parser.add_argument(
        "--config-json",
        default=str(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "hardware_evaluation.template.json"
        ),
    )
    args = parser.parse_args()
    configuration = load_json(args.config_json)
    payload = build_payload(args.config_json)
    output_path = resolve_path(args.config_json, configuration["payload_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"payload={output_path}")


if __name__ == "__main__":
    main()
