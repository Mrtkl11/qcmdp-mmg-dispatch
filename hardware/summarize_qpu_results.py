from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.release_config import load_json, require_fields, resolve_relative_path


def resolve_path(config_path: str, value: str) -> Path:
    return Path(resolve_relative_path(config_path, value))


def apply_value_head(expectations: np.ndarray, state: dict) -> np.ndarray:
    weight = np.asarray(state["weight"], dtype=float)
    bias = np.asarray(state["bias"], dtype=float)
    return (expectations @ weight.T + bias).reshape(-1)


def summarize(config_path: str) -> pd.DataFrame:
    configuration = load_json(config_path)
    require_fields(
        configuration,
        {"payload_path", "measurement_path", "summary_path"},
        "hardware evaluation",
    )
    payload = json.loads(
        resolve_path(config_path, configuration["payload_path"]).read_text(
            encoding="utf-8"
        )
    )
    measurements = pd.read_csv(
        resolve_path(config_path, configuration["measurement_path"])
    )
    required = {"repetition", "state_label", "critic"}
    missing = sorted(required.difference(measurements.columns))
    if missing:
        raise ValueError(f"Missing measurement columns: {', '.join(missing)}")
    z_columns = [f"z{index}" for index in range(int(payload["n_qubits"]))]
    missing_z = sorted(set(z_columns).difference(measurements.columns))
    if missing_z:
        raise ValueError(f"Missing expectation columns: {', '.join(missing_z)}")
    rows = []
    labels = list(payload["state_labels"])
    for critic_name, critic in payload["critics"].items():
        local_values = dict(zip(labels, critic["local_values"]))
        subset = measurements[measurements["critic"] == critic_name].copy()
        subset["hardware_value"] = apply_value_head(
            subset[z_columns].to_numpy(dtype=float), critic["value_head"]
        )
        for state_label in labels:
            values = subset.loc[
                subset["state_label"] == state_label, "hardware_value"
            ].to_numpy(dtype=float)
            if values.size == 0:
                raise ValueError(
                    f"No measurements for critic={critic_name}, state={state_label}"
                )
            mean = float(values.mean())
            standard_deviation = float(values.std(ddof=1)) if values.size > 1 else 0.0
            local_value = float(local_values[state_label])
            relative_error = (
                100.0 * abs(mean - local_value) / max(abs(local_value), 1e-12)
            )
            rows.append(
                {
                    "critic": critic_name,
                    "state_label": state_label,
                    "repetitions": int(values.size),
                    "shots": int(payload["shots"]),
                    "local_value": local_value,
                    "hardware_mean": mean,
                    "hardware_standard_deviation": standard_deviation,
                    "confidence_95_half_width": 1.96
                    * standard_deviation
                    / np.sqrt(values.size),
                    "relative_error_percent": relative_error,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize repeated QPU measurements for the dual VQC critics"
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
    summary = summarize(args.config_json)
    output_path = resolve_path(args.config_json, configuration["summary_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"summary={output_path}")


if __name__ == "__main__":
    main()
