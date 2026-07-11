from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import gym
    from gym.spaces import Box
except ImportError:
    import gymnasium as gym
    from gymnasium.spaces import Box


@dataclass(frozen=True)
class IEEE33EnvConfig:
    time_step: float = 0.25
    load_demand_scale: float = 1.0
    grid_buy_price: float = 0.12
    grid_sell_price: float = 0.08
    curtailment_penalty: float = 0.05
    load_shedding_penalty: float = 0.2
    line_violation_penalty_coef: float = 0.05
    network_loss_penalty_coef: float = 0.01
    voltage_violation_penalty_coef: float = 40.0
    slack_import_cost_coef: float = 0.005
    voltage_min_pu: float = 0.95
    voltage_max_pu: float = 1.05
    base_kv: float = 12.66
    base_mva: float = 100.0
    load_q_ratio_scale: float = 1.0
    pv_capacity: Tuple[float, ...] = (400, 450, 500, 420, 450)
    wind_capacity: Tuple[float, ...] = (300, 350, 420, 320, 350)
    diesel_capacity: Tuple[float, ...] = (220, 240, 260, 220, 260)
    battery_capacity: Tuple[float, ...] = (450, 500, 550, 500, 600)
    battery_max_power_ratio: Tuple[float, ...] = (0.4, 0.4, 0.4, 0.4, 0.4)
    battery_soc_min: Tuple[float, ...] = (0.2, 0.2, 0.2, 0.2, 0.2)
    battery_soc_max: Tuple[float, ...] = (0.95, 0.95, 0.95, 0.95, 0.95)
    battery_efficiency: Tuple[float, ...] = (0.95, 0.95, 0.95, 0.95, 0.95)
    initial_soc: Tuple[float, ...] = (0.50, 0.55, 0.50, 0.45, 0.60)
    diesel_cost: Tuple[float, ...] = (0.18, 0.18, 0.19, 0.18, 0.20)
    diesel_fuel_capacity: Tuple[float, ...] = (1200, 1200, 1300, 1200, 1300)
    fuel_consumption_rate: Tuple[float, ...] = (0.10, 0.10, 0.10, 0.10, 0.10)
    line_capacity_pv: Tuple[float, ...] = (450, 500, 550, 450, 500)
    line_capacity_wind: Tuple[float, ...] = (350, 400, 450, 350, 400)
    line_capacity_diesel: Tuple[float, ...] = (250, 280, 300, 250, 300)
    line_capacity_battery: Tuple[float, ...] = (250, 280, 300, 250, 300)
    line_capacity_grid: Tuple[float, ...] = (500, 500, 600, 500, 600)
    line_capacity_load: Tuple[float, ...] = (500, 550, 600, 550, 600)
    line_safety_margin: float = 0.9
    reward_voltage_cost_coef: float = 1.0
    reward_line_cost_coef: float = 1.0
    nrows: int = 288
    zone_load_weights: Tuple[float, ...] = (0.14, 0.18, 0.22, 0.20, 0.26)
    zone_pv_weights: Tuple[float, ...] = (0.18, 0.20, 0.24, 0.16, 0.22)
    zone_wind_weights: Tuple[float, ...] = (0.10, 0.15, 0.30, 0.20, 0.25)
    load_wave_amplitude: float = 0.0
    episode_horizon: int = 96
    random_episode_start: bool = False
    train_full_day: bool = False
    reward_scale: float = 1.0
    battery_action_deadband: float = 0.05
    diesel_min_output_ratio: float = 0.10

    @classmethod
    def from_mapping(cls, overrides: Optional[Mapping[str, Any]] = None) -> "IEEE33EnvConfig":
        base = cls()
        if not overrides:
            return base
        values = asdict(base)
        values.update(dict(overrides))
        return cls(**values)


DEFAULT_PARTITIONS: Dict[str, List[int]] = {
    "MG1": [1, 2, 3, 4, 5, 6],
    "MG2": [7, 8, 9, 10, 11, 12],
    "MG3": [13, 14, 15, 16, 17, 18],
    "MG4": [19, 20, 21, 22, 23, 24, 25],
    "MG5": [26, 27, 28, 29, 30, 31, 32, 33],
}

DEFAULT_DEVICE_MAP: Dict[str, Dict[str, int]] = {
    "MG1": {"diesel_bus": 4, "pv_bus": 5, "wind_bus": 6, "bess_bus": 3},
    "MG2": {"diesel_bus": 9, "pv_bus": 8, "wind_bus": 11, "bess_bus": 12},
    "MG3": {"diesel_bus": 16, "pv_bus": 14, "wind_bus": 13, "bess_bus": 15},
    "MG4": {"diesel_bus": 19, "pv_bus": 22, "wind_bus": 24, "bess_bus": 25},
    "MG5": {"diesel_bus": 27, "pv_bus": 30, "wind_bus": 33, "bess_bus": 32},
}

IEEE33_BUS_LOADS_KW: Dict[int, Tuple[float, float]] = {
    1: (0.0, 0.0), 2: (100.0, 60.0), 3: (90.0, 40.0), 4: (120.0, 80.0), 5: (60.0, 30.0), 6: (60.0, 20.0),
    7: (200.0, 100.0), 8: (200.0, 100.0), 9: (60.0, 20.0), 10: (60.0, 20.0), 11: (45.0, 30.0), 12: (60.0, 35.0),
    13: (60.0, 35.0), 14: (120.0, 80.0), 15: (60.0, 10.0), 16: (60.0, 20.0), 17: (60.0, 20.0), 18: (90.0, 40.0),
    19: (90.0, 40.0), 20: (90.0, 40.0), 21: (90.0, 40.0), 22: (90.0, 40.0), 23: (90.0, 50.0), 24: (420.0, 200.0),
    25: (420.0, 200.0), 26: (60.0, 25.0), 27: (60.0, 25.0), 28: (60.0, 20.0), 29: (120.0, 70.0), 30: (200.0, 600.0),
    31: (150.0, 70.0), 32: (210.0, 100.0), 33: (60.0, 40.0),
}

IEEE33_BRANCHES: List[Tuple[int, int, float, float]] = [
    (1, 2, 0.0922, 0.0470), (2, 3, 0.4930, 0.2511), (3, 4, 0.3660, 0.1864), (4, 5, 0.3811, 0.1941),
    (5, 6, 0.8190, 0.7070), (6, 7, 0.1872, 0.6188), (7, 8, 0.7114, 0.2351), (8, 9, 1.0300, 0.7400),
    (9, 10, 1.0440, 0.7400), (10, 11, 0.1966, 0.0650), (11, 12, 0.3744, 0.1238), (12, 13, 1.4680, 1.1550),
    (13, 14, 0.5416, 0.7129), (14, 15, 0.5910, 0.5260), (15, 16, 0.7463, 0.5450), (16, 17, 1.2890, 1.7210),
    (17, 18, 0.7320, 0.5740), (2, 19, 0.1640, 0.1565), (19, 20, 1.5042, 1.3554), (20, 21, 0.4095, 0.4784),
    (21, 22, 0.7089, 0.9373), (3, 23, 0.4512, 0.3083), (23, 24, 0.8980, 0.7091), (24, 25, 0.8960, 0.7011),
    (6, 26, 0.2030, 0.1034), (26, 27, 0.2842, 0.1447), (27, 28, 1.0590, 0.9337), (28, 29, 0.8042, 0.7006),
    (29, 30, 0.5075, 0.2585), (30, 31, 0.9744, 0.9630), (31, 32, 0.3105, 0.3619), (32, 33, 0.3410, 0.5302),
]


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: Dict[str, float]


class IEEE33SingleMGEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, data_path: str, config: Optional[Mapping[str, Any]] = None, mg_id: int = 1):
        super().__init__()
        self.config = IEEE33EnvConfig.from_mapping(config)
        self.data = pd.read_csv(data_path)
        self._prepare_dataframe()

        self.nrows = min(int(self.config.nrows), len(self.data))
        self.data = self.data.iloc[: self.nrows].copy()
        self.num_microgrids = 5
        self.mg_id = int(mg_id)
        self.zone_index = self.mg_id - 1
        self.zone_name = f"MG{self.mg_id}"
        self.bus_partitions = DEFAULT_PARTITIONS
        self.device_map = DEFAULT_DEVICE_MAP
        self.buses = list(self.bus_partitions[self.zone_name])
        self.devices = dict(self.device_map[self.zone_name])

        self.simulation_steps = len(self.data)
        self.current_step = 0
        self.time_step = float(self.config.time_step)

        self.zone_load_weights = self._normalized_array(self.config.zone_load_weights)
        self.zone_pv_weights = self._normalized_array(self.config.zone_pv_weights)
        self.zone_wind_weights = self._normalized_array(self.config.zone_wind_weights)

        self.pv_capacity = self._config_array(self.config.pv_capacity)[self.zone_index]
        self.wind_capacity = self._config_array(self.config.wind_capacity)[self.zone_index]
        self.diesel_capacity = self._config_array(self.config.diesel_capacity)[self.zone_index]
        self.battery_capacity = self._config_array(self.config.battery_capacity)[self.zone_index]
        self.battery_max_power_ratio = self._config_array(self.config.battery_max_power_ratio)[self.zone_index]
        self.battery_soc_min = self._config_array(self.config.battery_soc_min)[self.zone_index]
        self.battery_soc_max = self._config_array(self.config.battery_soc_max)[self.zone_index]
        self.battery_efficiency = self._config_array(self.config.battery_efficiency)[self.zone_index]
        self.initial_soc = self._config_array(self.config.initial_soc)[self.zone_index]
        self.diesel_cost = self._config_array(self.config.diesel_cost)[self.zone_index]
        self.diesel_fuel_capacity = self._config_array(self.config.diesel_fuel_capacity)[self.zone_index]
        self.fuel_consumption_rate = self._config_array(self.config.fuel_consumption_rate)[self.zone_index]

        self.line_capacity_pv = self._config_array(self.config.line_capacity_pv)[self.zone_index]
        self.line_capacity_wind = self._config_array(self.config.line_capacity_wind)[self.zone_index]
        self.line_capacity_diesel = self._config_array(self.config.line_capacity_diesel)[self.zone_index]
        self.line_capacity_battery = self._config_array(self.config.line_capacity_battery)[self.zone_index]
        self.line_capacity_grid = self._config_array(self.config.line_capacity_grid)[self.zone_index]
        self.line_capacity_load = self._config_array(self.config.line_capacity_load)[self.zone_index]

        self._init_network_model()

        self.episode_horizon = int(min(max(8, self.config.episode_horizon), self.simulation_steps))
        self.random_episode_start = bool(self.config.random_episode_start)
        self.train_full_day = bool(self.config.train_full_day)

        self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_dim = 7 + len(self.network_buses) + len(self.branch_order) + 1 + 2
        self.observation_space = Box(
            low=np.full(self.observation_dim, -5.0, dtype=np.float32),
            high=np.full(self.observation_dim, 5.0, dtype=np.float32),
            dtype=np.float32,
        )

        self.operation_history: List[Dict[str, Any]] = []
        self.balance_errors: List[float] = []
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.prev_diesel_power = 0.0
        self.last_pf: Dict[str, Any] = {}
        self.cumulative_cost = 0.0
        self.battery_soc = float(self.initial_soc)
        self.diesel_fuel_level = 100.0
        self.start_step = 0
        self.end_step = self.episode_horizon
        self.episode_step = 0
        _ = self.reset(seed=None)

    @staticmethod
    def _normalized_array(values: Sequence[float]) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return arr / arr.sum()

    @staticmethod
    def _config_array(values: Sequence[float]) -> np.ndarray:
        return np.asarray(values, dtype=float)

    def _prepare_dataframe(self) -> None:
        if "time" in self.data.columns:
            self.data["time"] = pd.to_datetime(self.data["time"])
            self.data = self.data.sort_values("time").drop_duplicates(subset="time", keep="last").set_index("time")
        elif not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.date_range("2018-01-01", periods=len(self.data), freq="15min")

    def _init_network_model(self) -> None:
        z_base_ohm = (self.config.base_kv**2) / self.config.base_mva
        self.network_buses = list(range(1, 34))
        self.slack_bus = 1
        self.bus_p_base = np.array([IEEE33_BUS_LOADS_KW[bus][0] for bus in self.network_buses], dtype=float)
        self.bus_q_base = np.array([IEEE33_BUS_LOADS_KW[bus][1] for bus in self.network_buses], dtype=float)
        self.bus_load_weights_within_mg: Dict[str, np.ndarray] = {}
        for zone_name, buses in self.bus_partitions.items():
            weights = np.array([IEEE33_BUS_LOADS_KW[bus][0] for bus in buses], dtype=float)
            if weights.sum() <= 0:
                weights = np.ones(len(buses), dtype=float)
            self.bus_load_weights_within_mg[zone_name] = weights / weights.sum()

        self.branches: List[Dict[str, float]] = []
        self.children: Dict[int, List[int]] = {bus: [] for bus in self.network_buses}
        self.parent: Dict[int, Optional[int]] = {bus: None for bus in self.network_buses}
        self.branch_map: Dict[Tuple[int, int], Dict[str, float]] = {}
        for from_bus, to_bus, r_ohm, x_ohm in IEEE33_BRANCHES:
            branch = {
                "from_bus": int(from_bus),
                "to_bus": int(to_bus),
                "r_ohm": float(r_ohm),
                "x_ohm": float(x_ohm),
                "r_pu": float(r_ohm / z_base_ohm),
                "x_pu": float(x_ohm / z_base_ohm),
            }
            self.branches.append(branch)
            self.children[int(from_bus)].append(int(to_bus))
            self.parent[int(to_bus)] = int(from_bus)
            self.branch_map[(int(from_bus), int(to_bus))] = branch
        self.branch_order = [(branch["from_bus"], branch["to_bus"]) for branch in self.branches]
        self.post_order = list(reversed(self.branch_order))

    def _profile_split(self, total_value: float, weights: np.ndarray, hour_index: int, kind: str) -> np.ndarray:
        phase = np.array([0.0, 0.2, 0.45, 0.65, 0.85])
        if kind == "load":
            modulation = 1.0 + 0.08 * np.sin(2 * np.pi * (hour_index / 96.0 + phase))
        elif kind == "pv":
            modulation = np.clip(1.0 + 0.12 * np.cos(2 * np.pi * (hour_index / 96.0 + phase)), 0.7, 1.3)
        else:
            modulation = np.clip(1.0 + 0.15 * np.sin(2 * np.pi * (hour_index / 96.0 + phase + 0.1)), 0.6, 1.4)
        mixed_weights = weights * modulation
        return total_value * (mixed_weights / mixed_weights.sum())

    def _load_multiplier(self, timestamp: pd.Timestamp) -> float:
        quarter = (int(timestamp.hour) * 4 + int(timestamp.minute / 15)) / 96.0
        wave = 1.0 + self.config.load_wave_amplitude * np.sin(2 * np.pi * (quarter + 0.08))
        return max(0.5, wave)

    def _all_zone_inputs(self):
        row = self.data.iloc[self.current_step]
        timestamp = self.data.index[self.current_step]
        total_pv = float(row["solar_power"]) * 1000.0
        total_wind = float(row["wind_power"]) * 1000.0
        total_load = float(row["household_power"]) * 1000.0 * self.config.load_demand_scale
        total_load *= self._load_multiplier(timestamp)

        buy_price = float(row["EUR/kWh"]) if "EUR/kWh" in row else self.config.grid_buy_price
        sell_price = buy_price * (self.config.grid_sell_price / self.config.grid_buy_price) if self.config.grid_buy_price > 1e-6 else 0.0

        pv_all = np.minimum(self._profile_split(total_pv, self.zone_pv_weights, self.current_step, "pv"), self._config_array(self.config.pv_capacity))
        wind_all = np.minimum(self._profile_split(total_wind, self.zone_wind_weights, self.current_step, "wind"), self._config_array(self.config.wind_capacity))
        load_all = self._profile_split(total_load, self.zone_load_weights, self.current_step, "load")
        return pv_all, wind_all, load_all, buy_price, sell_price

    def _zone_inputs(self):
        pv_all, wind_all, load_all, buy_price, sell_price = self._all_zone_inputs()
        return float(pv_all[self.zone_index]), float(wind_all[self.zone_index]), float(load_all[self.zone_index]), float(buy_price), float(sell_price)

    @staticmethod
    def _sanitize_action(action: np.ndarray) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        return np.clip(arr[:3], -1.0, 1.0)

    def _build_observation(
        self,
        pv: float,
        wind: float,
        load: float,
        diesel_power: float,
        fuel_level_ratio: float,
        grid_power: float,
        buy_price: float,
        sell_price: float,
        balance_error: float,
        pf: Optional[Mapping[str, Any]],
    ) -> np.ndarray:
        max_load = max(self.pv_capacity + self.wind_capacity + self.diesel_capacity + self.line_capacity_grid, 1e-6)
        if pf is None:
            voltages = np.ones(len(self.network_buses), dtype=np.float32)
            line_state = np.zeros(len(self.branch_order), dtype=np.float32)
            loss_feature = np.zeros(1, dtype=np.float32)
        else:
            voltages = np.asarray(pf["bus_voltage_pu"], dtype=np.float32)
            line_state = np.array(
                [float(pf["line_loading_pu"].get(branch, 0.0)) for branch in self.branch_order],
                dtype=np.float32,
            )
            max_loss_ref = max(300.0, float(self.bus_p_base.sum()) * self.config.load_demand_scale)
            loss_feature = np.array(
                [np.clip(float(pf["total_loss_kw"]) / max_loss_ref, 0.0, 5.0)],
                dtype=np.float32,
            )

        observation = np.concatenate(
            [
                np.array(
                    [
                        pv / max(self.pv_capacity, 1e-6),
                        wind / max(self.wind_capacity, 1e-6),
                        load / max_load,
                        self.battery_soc,
                        diesel_power / max(self.diesel_capacity, 1e-6),
                        fuel_level_ratio,
                        grid_power / max(self.line_capacity_grid * self.config.line_safety_margin, 1e-6),
                    ],
                    dtype=np.float32,
                ),
                voltages.astype(np.float32),
                line_state.astype(np.float32),
                loss_feature,
                np.array(
                    [
                        np.clip(buy_price / 0.2, 0.0, 5.0),
                        np.clip(sell_price / 0.2, 0.0, 5.0),
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        return np.clip(observation, self.observation_space.low, self.observation_space.high)

    def _build_bus_injections(self, zone_snapshot: Mapping[str, Mapping[str, float]], current_actions: Mapping[str, float]):
        p_load_kw = np.zeros(33, dtype=float)
        q_load_kvar = np.zeros(33, dtype=float)
        p_gen_kw = np.zeros(33, dtype=float)

        for zone_name, buses in self.bus_partitions.items():
            zone = zone_snapshot[zone_name]
            load_weights = self.bus_load_weights_within_mg[zone_name]
            zone_load_bus = zone["load_kw"] * load_weights
            for offset, bus in enumerate(buses):
                idx = bus - 1
                p_load_kw[idx] += zone_load_bus[offset]
                base_p, base_q = IEEE33_BUS_LOADS_KW[bus]
                q_ratio = (base_q / base_p) if base_p > 1e-9 else 0.0
                q_load_kvar[idx] += zone_load_bus[offset] * q_ratio * self.config.load_q_ratio_scale

            devices = self.device_map[zone_name]
            p_gen_kw[devices["pv_bus"] - 1] += zone["pv_kw"]
            p_gen_kw[devices["wind_bus"] - 1] += zone["wind_kw"]

        devices = self.device_map[self.zone_name]
        p_gen_kw[devices["diesel_bus"] - 1] += max(0.0, current_actions.get("diesel_power_kw", 0.0))

        battery_power = current_actions.get("battery_power_kw", 0.0)
        if battery_power >= 0:
            p_gen_kw[devices["bess_bus"] - 1] += battery_power
        else:
            p_load_kw[devices["bess_bus"] - 1] += abs(battery_power)

        controlled_load_shedding = max(0.0, current_actions.get("load_shedding_kw", 0.0))
        controlled_curtailment = max(0.0, current_actions.get("curtailment_kw", 0.0))

        if controlled_load_shedding > 0:
            buses = self.bus_partitions[self.zone_name]
            weights = self.bus_load_weights_within_mg[self.zone_name]
            for offset, bus in enumerate(buses):
                p_load_kw[bus - 1] = max(0.0, p_load_kw[bus - 1] - controlled_load_shedding * weights[offset])
                base_p, base_q = IEEE33_BUS_LOADS_KW[bus]
                q_ratio = (base_q / base_p) if base_p > 1e-9 else 0.0
                q_load_kvar[bus - 1] = max(0.0, q_load_kvar[bus - 1] - controlled_load_shedding * weights[offset] * q_ratio * self.config.load_q_ratio_scale)

        if controlled_curtailment > 0:
            renewable_total = zone_snapshot[self.zone_name]["pv_kw"] + zone_snapshot[self.zone_name]["wind_kw"]
            if renewable_total > 1e-9:
                pv_cut = controlled_curtailment * zone_snapshot[self.zone_name]["pv_kw"] / renewable_total
                wind_cut = controlled_curtailment * zone_snapshot[self.zone_name]["wind_kw"] / renewable_total
                p_gen_kw[self.device_map[self.zone_name]["pv_bus"] - 1] = max(0.0, p_gen_kw[self.device_map[self.zone_name]["pv_bus"] - 1] - pv_cut)
                p_gen_kw[self.device_map[self.zone_name]["wind_bus"] - 1] = max(0.0, p_gen_kw[self.device_map[self.zone_name]["wind_bus"] - 1] - wind_cut)

        return {
            "p_load_kw": p_load_kw,
            "q_load_kvar": q_load_kvar,
            "p_gen_kw": p_gen_kw,
            "p_net_mw": (p_load_kw - p_gen_kw) / 1000.0,
            "q_net_mvar": q_load_kvar / 1000.0,
        }

    def _run_power_flow(self, p_net_mw: np.ndarray, q_net_mvar: np.ndarray):
        p_bus = np.asarray(p_net_mw, dtype=float) / self.config.base_mva
        q_bus = np.asarray(q_net_mvar, dtype=float) / self.config.base_mva
        voltage_sq = np.ones(33, dtype=float)
        p_flow: Dict[Tuple[int, int], float] = {}
        q_flow: Dict[Tuple[int, int], float] = {}
        loss_pu: Dict[Tuple[int, int], float] = {}
        line_loading: Dict[Tuple[int, int], float] = {}

        p_acc = {bus: float(p_bus[bus - 1]) for bus in self.network_buses}
        q_acc = {bus: float(q_bus[bus - 1]) for bus in self.network_buses}
        for from_bus, to_bus in self.post_order:
            branch = self.branch_map[(from_bus, to_bus)]
            p_down = p_acc[to_bus]
            q_down = q_acc[to_bus]
            v_from = max(voltage_sq[from_bus - 1], 0.90**2)
            current_sq = (p_down**2 + q_down**2) / max(v_from, 1e-9)
            loss = branch["r_pu"] * current_sq
            p_flow[(from_bus, to_bus)] = p_down + loss
            q_flow[(from_bus, to_bus)] = q_down + branch["x_pu"] * current_sq
            loss_pu[(from_bus, to_bus)] = loss
            p_acc[from_bus] += p_flow[(from_bus, to_bus)]
            q_acc[from_bus] += q_flow[(from_bus, to_bus)]

        for from_bus, to_bus in self.branch_order:
            branch = self.branch_map[(from_bus, to_bus)]
            vi_sq = max(voltage_sq[from_bus - 1], 0.85**2)
            pf = p_flow[(from_bus, to_bus)]
            qf = q_flow[(from_bus, to_bus)]
            drop = 2.0 * (branch["r_pu"] * pf + branch["x_pu"] * qf)
            correction = ((branch["r_pu"] ** 2 + branch["x_pu"] ** 2) * (pf**2 + qf**2)) / max(vi_sq, 1e-9)
            voltage_sq[to_bus - 1] = max(0.80**2, vi_sq - drop + correction)
            apparent_mva = np.sqrt((pf * self.config.base_mva) ** 2 + (qf * self.config.base_mva) ** 2)
            thermal_ref_mva = max(1.0, 0.8 * (self.bus_p_base.sum() / 1000.0))
            line_loading[(from_bus, to_bus)] = apparent_mva / thermal_ref_mva

        voltage = np.sqrt(voltage_sq)
        voltage_violation = np.maximum(0.0, self.config.voltage_min_pu - voltage) + np.maximum(0.0, voltage - self.config.voltage_max_pu)
        return {
            "bus_voltage_pu": voltage,
            "line_p_flow_mw": {key: p_flow[key] * self.config.base_mva for key in p_flow},
            "line_q_flow_mvar": {key: q_flow[key] * self.config.base_mva for key in q_flow},
            "line_loading_pu": line_loading,
            "line_loss_kw": {key: loss_pu[key] * self.config.base_mva * 1000.0 for key in loss_pu},
            "total_loss_kw": sum(loss_pu.values()) * self.config.base_mva * 1000.0,
            "slack_p_mw": p_acc[self.slack_bus] * self.config.base_mva,
            "slack_q_mvar": q_acc[self.slack_bus] * self.config.base_mva,
            "v_min_pu": float(voltage.min()),
            "v_max_pu": float(voltage.max()),
            "avg_voltage_dev_pu": float(np.mean(np.abs(voltage - 1.0))),
            "voltage_violation_sum": float(voltage_violation.sum()),
            "line_overload_sum": float(sum(max(0.0, loading - 1.0) for loading in line_loading.values())),
        }

    def _power_flow_from_dispatch(
        self,
        zone_snapshot: Mapping[str, Mapping[str, float]],
        diesel_power_kw: float,
        battery_power_kw: float,
        curtailment_kw: float = 0.0,
        load_shedding_kw: float = 0.0,
    ):
        bus_injections = self._build_bus_injections(
            zone_snapshot,
            {
                "battery_power_kw": float(battery_power_kw),
                "diesel_power_kw": float(diesel_power_kw),
                "curtailment_kw": float(curtailment_kw),
                "load_shedding_kw": float(load_shedding_kw),
            },
        )
        power_flow = self._run_power_flow(bus_injections["p_net_mw"], bus_injections["q_net_mvar"])
        return {"bus_inj": bus_injections, "pf": power_flow}

    def _build_action_bounds(self) -> Dict[str, float]:
        max_battery = min(self.battery_capacity * self.battery_max_power_ratio, self.line_capacity_battery * self.config.line_safety_margin)
        soc_discharge_max = max(0.0, (self.battery_soc - self.battery_soc_min) * self.battery_capacity * self.battery_efficiency) / max(self.time_step, 1e-9)
        soc_charge_max = max(0.0, (self.battery_soc_max - self.battery_soc) * self.battery_capacity) / max(self.battery_efficiency * self.time_step, 1e-9)
        battery_min = -min(max_battery, soc_charge_max)
        battery_max = min(max_battery, soc_discharge_max)

        diesel_max = min(self.diesel_capacity, self.line_capacity_diesel * self.config.line_safety_margin)
        fuel_limit = (self.diesel_fuel_level / 100.0) * self.diesel_fuel_capacity / max(self.time_step, 1e-9)
        diesel_max = min(diesel_max, fuel_limit)
        diesel_min = min(diesel_max, self.config.diesel_min_output_ratio * self.diesel_capacity) if diesel_max > 1e-9 else 0.0

        grid_max = self.line_capacity_grid * self.config.line_safety_margin
        return {
            "diesel_min": float(max(0.0, diesel_min)),
            "diesel_max": float(max(0.0, diesel_max)),
            "battery_min": float(battery_min),
            "battery_max": float(battery_max),
            "grid_min": -float(grid_max),
            "grid_max": float(grid_max),
        }

    def _raw_action_to_physical_target(self, raw_action: np.ndarray, bounds: Mapping[str, float]) -> np.ndarray:
        diesel_cmd, battery_cmd, grid_cmd = self._sanitize_action(raw_action)
        diesel_target = 0.5 * (float(diesel_cmd) + 1.0) * bounds["diesel_max"]
        battery_target = 0.0 if abs(float(battery_cmd)) < self.config.battery_action_deadband else float(battery_cmd) * max(abs(bounds["battery_min"]), abs(bounds["battery_max"]))
        grid_target = float(grid_cmd) * max(abs(bounds["grid_min"]), abs(bounds["grid_max"]))
        return np.array([diesel_target, battery_target, grid_target], dtype=float)

    @staticmethod
    def _physical_action_to_policy_action(diesel: float, battery: float, grid: float, bounds: Mapping[str, float]) -> np.ndarray:
        battery_scale = max(abs(bounds["battery_min"]), abs(bounds["battery_max"]), 1e-6)
        grid_scale = max(abs(bounds["grid_min"]), abs(bounds["grid_max"]), 1e-6)
        diesel_scale = max(bounds["diesel_max"], 1e-6)
        diesel_cmd = np.clip(2.0 * diesel / diesel_scale - 1.0, -1.0, 1.0) if diesel_scale > 1e-9 else -1.0
        battery_cmd = np.clip(battery / battery_scale, -1.0, 1.0)
        grid_cmd = np.clip(grid / grid_scale, -1.0, 1.0)
        return np.array([diesel_cmd, battery_cmd, grid_cmd], dtype=np.float32)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)

        max_start = max(0, self.simulation_steps - self.episode_horizon)
        self.start_step = int(np.random.randint(0, max_start + 1)) if self.random_episode_start and max_start > 0 else 0
        self.end_step = int(min(self.simulation_steps, self.start_step + self.episode_horizon))

        self.current_step = self.start_step
        self.episode_step = 0
        self.battery_soc = float(self.initial_soc)
        self.diesel_fuel_level = 100.0
        self.cumulative_cost = 0.0
        self.operation_history = []
        self.balance_errors = []
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.prev_diesel_power = 0.0

        pv, wind, load, buy_price, sell_price = self._zone_inputs()
        zone_pv, zone_wind, zone_load, _, _ = self._all_zone_inputs()
        zone_snapshot = {f"MG{idx + 1}": {"pv_kw": float(zone_pv[idx]), "wind_kw": float(zone_wind[idx]), "load_kw": float(zone_load[idx])} for idx in range(self.num_microgrids)}
        bus_injections = self._build_bus_injections(zone_snapshot, {"battery_power_kw": 0.0, "diesel_power_kw": 0.0, "curtailment_kw": 0.0, "load_shedding_kw": 0.0})
        self.last_pf = self._run_power_flow(bus_injections["p_net_mw"], bus_injections["q_net_mvar"])

        return self._build_observation(
            pv,
            wind,
            load,
            diesel_power=0.0,
            fuel_level_ratio=1.0,
            grid_power=0.0,
            buy_price=buy_price,
            sell_price=sell_price,
            balance_error=0.0,
            pf=self.last_pf,
        )

    def step(self, action: np.ndarray):
        raw_action = self._sanitize_action(action)
        zone_pv, zone_wind, zone_load, buy_price, sell_price = self._all_zone_inputs()
        zone_snapshot = {f"MG{idx + 1}": {"pv_kw": float(zone_pv[idx]), "wind_kw": float(zone_wind[idx]), "load_kw": float(zone_load[idx])} for idx in range(self.num_microgrids)}

        pv = zone_snapshot[self.zone_name]["pv_kw"]
        wind = zone_snapshot[self.zone_name]["wind_kw"]
        load = zone_snapshot[self.zone_name]["load_kw"]
        renewable = pv + wind

        bounds = self._build_action_bounds()
        target_action = self._raw_action_to_physical_target(raw_action, bounds)
        diesel_power = float(np.clip(target_action[0], bounds["diesel_min"], bounds["diesel_max"]))
        battery_power = float(np.clip(target_action[1], bounds["battery_min"], bounds["battery_max"]))
        grid_power = float(np.clip(target_action[2], bounds["grid_min"], bounds["grid_max"]))

        supply_gap = load - (renewable + battery_power + diesel_power + grid_power)
        load_shedding = min(load, supply_gap) if supply_gap > 0 else 0.0
        curtailment = min(renewable, -supply_gap) if supply_gap <= 0 else 0.0
        served_load = max(load - load_shedding, 0.0)
        balance_error = renewable + battery_power + diesel_power + grid_power - curtailment - served_load

        pf_eval = self._power_flow_from_dispatch(
            zone_snapshot,
            diesel_power_kw=diesel_power,
            battery_power_kw=battery_power,
            curtailment_kw=curtailment,
            load_shedding_kw=load_shedding,
        )
        power_flow = pf_eval["pf"]
        bus_injections = pf_eval["bus_inj"]
        self.last_pf = power_flow

        if battery_power >= 0:
            soc_delta = -(battery_power * self.time_step) / max(self.battery_capacity * self.battery_efficiency, 1e-6)
        else:
            soc_delta = (abs(battery_power) * self.time_step * self.battery_efficiency) / max(self.battery_capacity, 1e-6)
        self.battery_soc = float(np.clip(self.battery_soc + soc_delta, self.battery_soc_min, self.battery_soc_max))

        fuel_used = diesel_power * self.fuel_consumption_rate * self.time_step
        remaining_fuel = max(0.0, self.diesel_fuel_capacity * self.diesel_fuel_level / 100.0 - fuel_used)
        self.diesel_fuel_level = float(100.0 * remaining_fuel / max(self.diesel_fuel_capacity, 1e-6))
        fuel_level_ratio = self.diesel_fuel_level / 100.0

        diesel_cost = diesel_power * self.diesel_cost * self.time_step
        grid_import_cost = max(grid_power, 0.0) * buy_price * self.time_step
        grid_export_credit = max(-grid_power, 0.0) * sell_price * self.time_step
        reward_voltage_penalty = power_flow["voltage_violation_sum"] * self.config.reward_voltage_cost_coef
        reward_line_penalty = power_flow["line_overload_sum"] * self.config.reward_line_cost_coef
        economic_cost = diesel_cost + grid_import_cost - grid_export_credit
        reward = -float(self.config.reward_scale * economic_cost)
        constraint_cost = float(reward_voltage_penalty + reward_line_penalty)

        self.cumulative_cost += economic_cost
        self.balance_errors.append(abs(balance_error))
        policy_action = self._physical_action_to_policy_action(diesel_power, battery_power, grid_power, bounds)

        self.operation_history.append(
            {
                "timestamp": self.data.index[self.current_step],
                "mg": self.zone_name,
                "pv_power": pv,
                "wind_power": wind,
                "load_demand": load,
                "served_load_demand": served_load,
                "battery_power": battery_power,
                "diesel_power": diesel_power,
                "grid_power": grid_power,
                "battery_soc": self.battery_soc,
                "fuel_level": self.diesel_fuel_level,
                "buy_price": buy_price,
                "sell_price": sell_price,
                "curtailment": curtailment,
                "load_shedding": load_shedding,
                "power_balance_error": balance_error,
                "reward": reward,
                "economic_cost": economic_cost,
                "constraint_cost": constraint_cost,
                "network_total_loss_kw": power_flow["total_loss_kw"],
                "network_v_min_pu": power_flow["v_min_pu"],
                "network_v_max_pu": power_flow["v_max_pu"],
                "network_avg_voltage_dev_pu": power_flow["avg_voltage_dev_pu"],
                "network_voltage_violation_sum": power_flow["voltage_violation_sum"],
                "slack_p_mw": power_flow["slack_p_mw"],
                "slack_q_mvar": power_flow["slack_q_mvar"],
                "bus_voltage_pu": power_flow["bus_voltage_pu"].tolist(),
                "bus_p_load_kw": bus_injections["p_load_kw"].tolist(),
                "bus_q_load_kvar": bus_injections["q_load_kvar"].tolist(),
                "bus_p_gen_kw": bus_injections["p_gen_kw"].tolist(),
                "line_loading_pu": {f"{key[0]}->{key[1]}": float(value) for key, value in power_flow["line_loading_pu"].items()},
                "line_p_flow_mw": {f"{key[0]}->{key[1]}": float(value) for key, value in power_flow["line_p_flow_mw"].items()},
                "line_q_flow_mvar": {f"{key[0]}->{key[1]}": float(value) for key, value in power_flow["line_q_flow_mvar"].items()},
                "line_loss_kw": {f"{key[0]}->{key[1]}": float(value) for key, value in power_flow["line_loss_kw"].items()},
            }
        )

        self.prev_action = policy_action.astype(np.float32)
        self.prev_diesel_power = float(diesel_power)
        self.current_step += 1
        self.episode_step += 1
        done = self.current_step >= self.end_step or self.current_step >= self.simulation_steps - 1

        if done:
            next_pv, next_wind, next_load, next_buy_price, next_sell_price, next_pf = pv, wind, load, buy_price, sell_price, power_flow
        else:
            next_pv, next_wind, next_load, next_buy_price, next_sell_price = self._zone_inputs()
            zone_pv_2, zone_wind_2, zone_load_2, _, _ = self._all_zone_inputs()
            zone_snapshot_2 = {f"MG{idx + 1}": {"pv_kw": float(zone_pv_2[idx]), "wind_kw": float(zone_wind_2[idx]), "load_kw": float(zone_load_2[idx])} for idx in range(self.num_microgrids)}
            bus_injections_2 = self._build_bus_injections(zone_snapshot_2, {"battery_power_kw": 0.0, "diesel_power_kw": 0.0, "curtailment_kw": 0.0, "load_shedding_kw": 0.0})
            next_pf = self._run_power_flow(bus_injections_2["p_net_mw"], bus_injections_2["q_net_mvar"])

        observation = self._build_observation(
            next_pv,
            next_wind,
            next_load,
            diesel_power=diesel_power,
            fuel_level_ratio=fuel_level_ratio,
            grid_power=grid_power,
            buy_price=next_buy_price,
            sell_price=next_sell_price,
            balance_error=balance_error,
            pf=next_pf,
        )
        info = {
            "mg": self.zone_name,
            "cumulative_cost": float(self.cumulative_cost),
            "constraint_cost": constraint_cost,
            "economic_cost": float(economic_cost),
        }
        return observation, reward, done, info
