from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def reset_observation(env: Any) -> np.ndarray:
    result = env.reset()
    return np.asarray(
        result[0] if isinstance(result, tuple) else result, dtype=np.float32
    )


def step_environment(
    env: Any, action: np.ndarray
) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
    result = env.step(action)
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
    else:
        observation, reward, done, info = result
    return (
        np.asarray(observation, dtype=np.float32),
        float(reward),
        bool(done),
        dict(info),
    )


def constraint_cost(info: Dict[str, Any]) -> float:
    return float(info.get("constraint_cost", info.get("net_pen", 0.0)))


def compute_gae(
    rewards: Iterable[float],
    values: Iterable[float],
    next_values: Iterable[float],
    dones: Iterable[float],
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    reward_array = np.asarray(list(rewards), dtype=np.float32)
    value_array = np.asarray(list(values), dtype=np.float32)
    next_value_array = np.asarray(list(next_values), dtype=np.float32)
    done_array = np.asarray(list(dones), dtype=np.float32)
    advantages = np.zeros_like(reward_array)
    running = 0.0
    for index in reversed(range(len(reward_array))):
        not_done = 1.0 - done_array[index]
        delta = (
            reward_array[index]
            + gamma * next_value_array[index] * not_done
            - value_array[index]
        )
        running = delta + gamma * gae_lambda * not_done * running
        advantages[index] = running
    return advantages, advantages + value_array


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (values - values.mean()) / (values.std() + 1e-8)


def clone_state(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value.detach().cpu().clone() if torch.is_tensor(value) else value
        for key, value in state.items()
    }


def save_json(path: Path, value: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def save_training_history(path: Path, history: List[Dict[str, Any]]) -> None:
    import pandas as pd

    pd.DataFrame(history).to_csv(path, index=False)
