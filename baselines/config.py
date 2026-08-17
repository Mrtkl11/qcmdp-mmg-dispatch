from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch


def read_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a JSON object")
    return value


def resolve_device(value: Optional[str]) -> str:
    if value in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(value)


def load_algorithm_config(path: str, algorithm: str) -> Dict[str, Any]:
    payload = read_json(path)
    environment_reference = payload.get("environment_config")
    if not isinstance(environment_reference, str):
        raise ValueError("environment_config must be a JSON path")
    environment_path = Path(environment_reference)
    if not environment_path.is_absolute():
        environment_path = Path(path).resolve().parent / environment_path
    environment = read_json(str(environment_path))
    shared = payload.get("shared", {})
    specific = payload.get(algorithm, {})
    if not isinstance(shared, dict) or not isinstance(specific, dict):
        raise ValueError(
            "The shared and algorithm configuration sections must be objects"
        )
    config = dict(shared)
    config.update(specific)
    config["device"] = resolve_device(config.get("device"))
    config["algorithm"] = algorithm
    config["environment"] = environment
    required = {
        "horizon",
        "updates",
        "gamma",
        "actor_lr",
        "critic_lr",
        "actor_hidden",
        "critic_hidden",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(
            f"Missing configuration fields for {algorithm}: {', '.join(missing)}"
        )
    return config


def merge_cli_overrides(
    config: Mapping[str, Any], device: Optional[str], updates: Optional[int]
) -> Dict[str, Any]:
    merged = dict(config)
    if device:
        merged["device"] = resolve_device(device)
    if updates is not None:
        merged["updates"] = int(updates)
    return merged


def resolve_data_path(cli_value: Optional[str], default: Optional[str] = None) -> str:
    candidates = []
    if cli_value:
        candidates.append(Path(cli_value))
    if default:
        candidates.append(Path(default))
    candidates.append(Path.cwd() / "data" / "Environment_data_2018.csv")
    candidates.append(Path.cwd() / "Environment_data_2018.csv")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Dataset CSV was not found. Searched: {searched}")
