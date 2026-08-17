from dataclasses import dataclass
from typing import Optional, Sequence

import pennylane as qml
import torch as th
import torch.nn as nn


@dataclass(frozen=True)
class QuantumCriticConfig:
    n_qubits: int
    n_layers: int
    encoder_hidden: int
    weight_scale: float
    shots: Optional[int]


def build_quantum_device(config: QuantumCriticConfig):
    shots = config.shots
    analytic = shots is None or int(shots) <= 0
    for device_name in ("lightning.gpu", "lightning.qubit", "default.qubit"):
        try:
            if analytic:
                return qml.device(device_name, wires=config.n_qubits)
            return qml.device(device_name, wires=config.n_qubits, shots=int(shots))
        except Exception:
            continue
    return qml.device("default.qubit", wires=config.n_qubits, shots=shots)


class VQCCritic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        qcfg: QuantumCriticConfig,
    ):
        super().__init__()
        self.qcfg = qcfg
        self.n_qubits = self.qcfg.n_qubits
        self.n_layers = self.qcfg.n_layers
        self.q_input_dim = 3 * self.n_qubits

        self.state_encoder = nn.Sequential(
            nn.Linear(input_dim, self.qcfg.encoder_hidden),
            nn.Tanh(),
            nn.Linear(self.qcfg.encoder_hidden, self.q_input_dim),
        )
        weight_shape = (self.n_layers, 2 * self.n_qubits)
        self.q_weights = nn.Parameter(th.randn(*weight_shape) * self.qcfg.weight_scale)

        self.value_head = nn.Linear(self.n_qubits, output_dim)

        self.qnode = self._build_qnode()

    def _build_qnode(self):
        device = build_quantum_device(self.qcfg)
        qnode_kwargs = {"interface": "torch"}
        if self.qcfg.shots is None or int(self.qcfg.shots) <= 0:
            qnode_kwargs["diff_method"] = "adjoint"
        else:
            qnode_kwargs["diff_method"] = "parameter-shift"

        @qml.qnode(device, **qnode_kwargs)
        def circuit(inputs, weights):
            for qubit_index in range(self.n_qubits):
                base = 3 * qubit_index
                qml.RX(inputs[base], wires=qubit_index)
                qml.RY(inputs[base + 1], wires=qubit_index)
                qml.RZ(inputs[base + 2], wires=qubit_index)
            for layer_index in range(self.n_layers):
                for qubit_index in range(self.n_qubits):
                    qml.RY(weights[layer_index, qubit_index], wires=qubit_index)
                    qml.RZ(
                        weights[layer_index, qubit_index + self.n_qubits],
                        wires=qubit_index,
                    )
                for qubit_index in range(self.n_qubits - 1):
                    qml.CZ(wires=[qubit_index, qubit_index + 1])

            return [
                qml.expval(qml.PauliZ(qubit_index))
                for qubit_index in range(self.n_qubits)
            ]

        return circuit

    def _to_reference_tensor(self, value: th.Tensor) -> th.Tensor:
        if not isinstance(value, th.Tensor):
            value = th.as_tensor(value)
        return value.to(device=self.q_weights.device, dtype=self.q_weights.dtype)

    def forward(self, x: th.Tensor) -> th.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        encoded = th.atan(self.state_encoder(self._to_reference_tensor(x)))
        quantum_features = []
        for encoded_state in encoded.unbind(dim=0):
            circuit_output = self.qnode(encoded_state, self.q_weights)
            if isinstance(circuit_output, Sequence):
                circuit_output = th.stack(list(circuit_output))
            quantum_features.append(self._to_reference_tensor(circuit_output))
        quantum_features = th.stack(quantum_features, dim=0)
        return self.value_head(quantum_features).squeeze(-1)


class GaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Sequence[int],
        log_std_init: float,
        log_std_min: float,
        log_std_max: float,
    ):
        super().__init__()
        layers = []
        last_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(last_dim, hidden_dim), nn.Tanh()])
            last_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.mu_layer = nn.Linear(last_dim, act_dim)
        self.log_std = nn.Parameter(th.full((act_dim,), float(log_std_init)))
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.eps = 1e-6

    def _distribution(self, obs: th.Tensor):
        mu = self.mu_layer(self.backbone(obs))
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = th.exp(log_std)
        return th.distributions.Normal(mu, std)

    def _squashed_log_prob(
        self, dist, pre_tanh_action: th.Tensor, squashed_action: th.Tensor
    ) -> th.Tensor:
        log_prob = dist.log_prob(pre_tanh_action).sum(dim=-1)
        squash_correction = th.log1p(-squashed_action.pow(2) + self.eps).sum(dim=-1)
        return log_prob - squash_correction

    def act(self, obs: th.Tensor, deterministic: bool = False):
        dist = self._distribution(obs)
        pre_tanh_action = dist.mean if deterministic else dist.rsample()
        action = th.tanh(pre_tanh_action)
        log_prob = self._squashed_log_prob(dist, pre_tanh_action, action)
        return action, log_prob

    def log_prob(self, obs: th.Tensor, action: th.Tensor) -> th.Tensor:
        action = action.clamp(-1.0 + self.eps, 1.0 - self.eps)
        pre_tanh_action = 0.5 * (th.log1p(action) - th.log1p(-action))
        dist = self._distribution(obs)
        return self._squashed_log_prob(dist, pre_tanh_action, action)

    def entropy(self, obs: th.Tensor) -> th.Tensor:
        return self._distribution(obs).entropy().sum(dim=-1)


PennyLaneCritic = VQCCritic
