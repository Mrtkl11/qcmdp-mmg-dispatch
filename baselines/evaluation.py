from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .common import reset_observation, step_environment


def history_frames(
    history: Iterable[Mapping[str, Any]]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    history_frame = pd.DataFrame(list(history)).copy()
    if history_frame.empty:
        return history_frame, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    history_frame["timestamp"] = pd.to_datetime(history_frame["timestamp"])
    history_frame = history_frame.sort_values("timestamp").reset_index(drop=True)
    voltage_rows = []
    line_rows = []
    for _, row in history_frame.iterrows():
        for bus, value in enumerate(row.get("bus_voltage_pu", []) or [], start=1):
            voltage_rows.append(
                {
                    "timestamp": row["timestamp"],
                    "bus": bus,
                    "voltage_pu": float(value),
                    "mg": row.get("mg", ""),
                }
            )
        loading = row.get("line_loading_pu", {}) or {}
        active_flow = row.get("line_p_flow_mw", {}) or {}
        loss = row.get("line_loss_kw", {}) or {}
        for branch, value in loading.items():
            line_rows.append(
                {
                    "timestamp": row["timestamp"],
                    "branch": branch,
                    "loading_pu": float(value),
                    "p_flow_mw": float(active_flow.get(branch, np.nan)),
                    "loss_kw": float(loss.get(branch, np.nan)),
                }
            )
    metric_columns = [
        "timestamp",
        "pv_power",
        "wind_power",
        "load_demand",
        "battery_power",
        "diesel_power",
        "grid_power",
        "battery_soc",
        "fuel_level",
        "buy_price",
        "sell_price",
        "reward",
        "economic_cost",
        "constraint_cost",
        "network_total_loss_kw",
        "network_v_min_pu",
        "network_v_max_pu",
        "network_avg_voltage_dev_pu",
        "network_voltage_violation_sum",
        "slack_p_mw",
        "slack_q_mvar",
    ]
    metrics_frame = history_frame[
        [column for column in metric_columns if column in history_frame.columns]
    ].copy()
    return (
        history_frame,
        pd.DataFrame(voltage_rows),
        pd.DataFrame(line_rows),
        metrics_frame,
    )


def plot_power_balance(history: pd.DataFrame, path: Path, title: str) -> None:
    x = np.arange(len(history))
    supply = (
        history["pv_power"].astype(float)
        + history["wind_power"].astype(float)
        + history["diesel_power"].astype(float)
    )
    battery = history["battery_power"].astype(float)
    grid = history["grid_power"].astype(float)
    load = history["load_demand"].astype(float)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(x, supply, label="PV + wind + diesel", linewidth=1.8)
    ax.plot(x, battery, label="Battery", linewidth=1.4)
    ax.plot(x, grid, label="PCC exchange", linewidth=1.4)
    ax.plot(x, load, label="Load", color="black", linewidth=2.0)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Power (kW)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_voltage_profile(
    voltage: pd.DataFrame,
    path: Path,
    title: str,
    voltage_min: float,
    voltage_max: float,
) -> None:
    if voltage.empty:
        return
    summary = (
        voltage.groupby("bus")["voltage_pu"].agg(["min", "mean", "max"]).reset_index()
    )
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.fill_between(
        summary["bus"], summary["min"], summary["max"], alpha=0.28, label="Min-max"
    )
    ax.plot(summary["bus"], summary["mean"], marker="o", markersize=3, label="Mean")
    ax.axhline(voltage_min, color="tab:red", linestyle="--", linewidth=1)
    ax.axhline(voltage_max, color="tab:red", linestyle="--", linewidth=1)
    ax.set_xlabel("Bus index")
    ax.set_ylabel("Voltage (p.u.)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_voltage_heatmap(
    voltage: pd.DataFrame,
    path: Path,
    title: str,
    voltage_min: float,
    voltage_max: float,
) -> None:
    if voltage.empty:
        return
    matrix = voltage.pivot(
        index="timestamp", columns="bus", values="voltage_pu"
    ).sort_index()
    fig, ax = plt.subplots(figsize=(12, 5.2))
    image = ax.imshow(
        matrix.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        vmin=voltage_min,
        vmax=voltage_max,
        extent=(0.5, matrix.shape[1] + 0.5, matrix.shape[0] - 0.5, -0.5),
    )
    ax.set_xlabel("Bus index")
    ax.set_ylabel("Time step")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Voltage (p.u.)")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_network_metrics(
    metrics: pd.DataFrame, path: Path, voltage_min: float, voltage_max: float
) -> None:
    if metrics.empty:
        return
    x = np.arange(len(metrics))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(x, metrics["network_total_loss_kw"].astype(float), color="tab:blue")
    axes[0].set_ylabel("Network loss (kW)")
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        x,
        metrics["network_v_min_pu"].astype(float),
        label="Minimum",
        color="tab:orange",
    )
    axes[1].plot(
        x, metrics["network_v_max_pu"].astype(float), label="Maximum", color="tab:green"
    )
    axes[1].axhline(voltage_min, color="tab:red", linestyle="--", linewidth=1)
    axes[1].axhline(voltage_max, color="tab:red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Voltage (p.u.)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_line_loading(
    lines: pd.DataFrame, path: Path, title: str, safety_margin: float
) -> None:
    if lines.empty:
        return
    timestamp = lines["timestamp"].max()
    snapshot = lines[lines["timestamp"] == timestamp].sort_values("branch")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(snapshot)), snapshot["loading_pu"].astype(float))
    ax.axhline(safety_margin, color="tab:red", linestyle="--", label="Safety limit")
    ax.set_xticks(np.arange(len(snapshot)))
    ax.set_xticklabels(snapshot["branch"], rotation=90)
    ax.set_ylabel("Loading (p.u.)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run_evaluation_and_plot(
    trainer: Any, env: Any, mg_id: int, output_dir: str
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    observation = reset_observation(env)
    print(f"[{output}] running deterministic dispatch evaluation")
    for _ in range(env.episode_horizon):
        observation_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=trainer.device
        ).unsqueeze(0)
        with torch.no_grad():
            action, _ = trainer.actor.act(observation_tensor, deterministic=True)
        observation, _, done, _ = step_environment(env, action.squeeze(0).cpu().numpy())
        if done:
            break
    history = getattr(env, "operation_history", [])
    history_frame, voltage_frame, line_frame, metrics_frame = history_frames(history)
    if history_frame.empty:
        raise RuntimeError("Evaluation did not produce operation history")
    history_frame.to_csv(output / "operation_history.csv", index=False)
    voltage_frame.to_csv(output / "bus_voltage_timeseries.csv", index=False)
    line_frame.to_csv(output / "line_loading_timeseries.csv", index=False)
    metrics_frame.to_csv(output / "network_metrics_timeseries.csv", index=False)
    label = f"MG{mg_id}"
    plot_power_balance(
        history_frame, output / "power_balance.png", f"Power Balance ({label})"
    )
    plot_voltage_profile(
        voltage_frame,
        output / "voltage_profile_summary.png",
        f"Voltage Profile ({label})",
        env.voltage_min_pu,
        env.voltage_max_pu,
    )
    plot_voltage_heatmap(
        voltage_frame,
        output / "voltage_heatmap.png",
        f"Bus Voltage Heatmap ({label})",
        env.voltage_min_pu,
        env.voltage_max_pu,
    )
    plot_network_metrics(
        metrics_frame,
        output / "network_power_metrics.png",
        env.voltage_min_pu,
        env.voltage_max_pu,
    )
    plot_line_loading(
        line_frame,
        output / "line_loading_last_step.png",
        f"Line Loading Snapshot ({label})",
        env.line_safety_margin,
    )
    print(f"[{output}] evaluation artifacts saved")
