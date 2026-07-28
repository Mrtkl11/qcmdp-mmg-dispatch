from pathlib import Path

import torch

from src.qcmdp_model_pennylane import QuantumCriticConfig, VQCCritic
from src.release_config import load_json


def test_paper_aligned_vqc_structure_and_gradient():
    root = Path(__file__).resolve().parents[1]
    training = load_json(root / "configs" / "experiment_config.template.json")[
        "training"
    ]
    configuration = QuantumCriticConfig(
        n_qubits=int(training["n_qubits"]),
        n_layers=int(training["n_layers"]),
        encoder_hidden=int(training["critic_encoder_hidden"]),
        weight_scale=float(training["quantum_weight_scale"]),
        shots=training["shots"],
    )
    critic = VQCCritic(75, 1, configuration)
    observation = torch.zeros(2, 75, requires_grad=True)
    values = critic(observation)
    values.sum().backward()

    assert critic.q_input_dim == 3 * configuration.n_qubits
    assert tuple(critic.q_weights.shape) == (
        configuration.n_layers,
        2 * configuration.n_qubits,
    )
    assert critic.value_head.in_features == configuration.n_qubits
    assert critic.value_head.out_features == 1
    assert tuple(values.shape) == (2,)
    assert observation.grad is not None
    assert torch.isfinite(observation.grad).all()
