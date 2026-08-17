from __future__ import annotations

from typing import Any, Dict

from src.qcmdp_model_pennylane import QuantumCriticConfig, VQCCritic

from .on_policy import OnPolicyTrainer


class QPPOTrainer(OnPolicyTrainer):
    def _build_reward_critic(self, obs_dim: int):
        config = self.config
        quantum_config = QuantumCriticConfig(
            n_qubits=int(config["n_qubits"]),
            n_layers=int(config["n_layers"]),
            encoder_hidden=int(config["critic_encoder_hidden"]),
            weight_scale=float(config["quantum_weight_scale"]),
            shots=config["shots"],
        )
        return VQCCritic(obs_dim, 1, quantum_config)

    def __init__(self, env: Any, config: Dict[str, Any]):
        super().__init__(env, config, "qppo")
