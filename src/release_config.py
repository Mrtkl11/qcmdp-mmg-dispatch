from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


def load_json(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("Configuration JSON must be a top-level object")
    return data


def require_fields(
    values: Mapping[str, Any], required: Iterable[str], section: str
) -> None:
    missing = sorted(set(required).difference(values))
    if missing:
        raise ValueError(f"Missing {section} fields: {', '.join(missing)}")


def resolve_required_data_path(cli_value: Optional[str], env_var: str) -> str:
    if cli_value:
        path = Path(cli_value)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return str(path.resolve())
    env_value = os.getenv(env_var)
    if env_value:
        path = Path(env_value)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return str(path.resolve())
    raise FileNotFoundError(
        f"Dataset path is required; pass --data-path or set {env_var}"
    )


def resolve_relative_path(config_path: str, value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = Path(config_path).resolve().parent / path
    return str(path.resolve())
