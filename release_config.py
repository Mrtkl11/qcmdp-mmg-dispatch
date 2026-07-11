from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def load_optional_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Experiment config JSON must be a top-level object.")
    return data


def resolve_required_data_path(cli_value: Optional[str], env_var: str = "QCMDP_DATA_PATH") -> str:
    candidate = cli_value or Path.cwd().joinpath("").as_posix()
    if cli_value:
        path = Path(cli_value)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return str(path)

    import os

    env_value = os.getenv(env_var)
    if env_value:
        path = Path(env_value)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return str(path)

    raise FileNotFoundError(
        "Dataset path is required for this public release. "
        f"Pass --data-path or set {env_var}. "
        "The repository only includes a schema template, not the full experiment data."
    )


def merge_overrides(base: Mapping[str, Any], overrides: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    merged = dict(base)
    if overrides:
        merged.update(dict(overrides))
    return merged
