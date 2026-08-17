from typing import Dict as TypingDict, Any, List
import numpy as np
import pandas as pd

try:
    import gym
    from gym.spaces import Box
except ImportError:
    import gymnasium as gym
    from gymnasium.spaces import Box

REQUIRED_ENV_FIELDS = {
    "time_step",
    "load_demand_scale",
    "grid_buy_price",
    "grid_sell_price",
    "curtailment_penalty",
    "load_shedding_penalty",
    "capacity_violation_weight",
    "network_loss_penalty_coef",
    "voltage_violation_weight",
    "slack_import_cost_coef",
    "voltage_min_pu",
    "voltage_max_pu",
    "base_kv",
    "base_mva",
    "power_flow_max_iterations",
    "power_flow_tolerance",
    "power_flow_relaxation",
    "load_q_ratio_scale",
    "pv_capacity",
    "wind_capacity",
    "diesel_capacity",
    "battery_capacity",
    "battery_max_power_ratio",
    "battery_soc_min",
    "battery_soc_max",
    "battery_efficiency",
    "initial_soc",
    "diesel_cost",
    "diesel_fuel_capacity",
    "fuel_consumption_rate",
    "line_capacity_pv",
    "line_capacity_wind",
    "line_capacity_diesel",
    "line_capacity_battery",
    "line_capacity_grid",
    "line_capacity_load",
    "branch_capacity_mva",
    "line_safety_margin",
    "nrows",
    "zone_load_weights",
    "zone_pv_weights",
    "zone_wind_weights",
    "episode_horizon",
    "random_episode_start",
    "reward_offset",
    "reward_scale",
    "reward_min",
    "reward_max",
    "imbalance_penalty_coef",
    "action_ramp_penalty_coef",
    "battery_action_deadband",
    "diesel_ramp_penalty_coef",
    "grid_import_penalty_coef",
    "grid_export_penalty_coef",
    "soc_target",
    "soc_deviation_penalty_coef",
    "action_gap_penalty_coef",
    "price_normalization_reference",
}

DEFAULT_PARTITIONS = {
    "MG1": [1, 2, 3, 4, 5, 6],
    "MG2": [7, 8, 9, 10, 11, 12],
    "MG3": [13, 14, 15, 16, 17, 18],
    "MG4": [19, 20, 21, 22, 23, 24, 25],
    "MG5": [26, 27, 28, 29, 30, 31, 32, 33],
}

DEFAULT_DEVICE_MAP = {
    "MG1": {"diesel_bus": 4, "pv_bus": 5, "wind_bus": 6, "bess_bus": 3},
    "MG2": {"diesel_bus": 9, "pv_bus": 8, "wind_bus": 11, "bess_bus": 12},
    "MG3": {"diesel_bus": 16, "pv_bus": 14, "wind_bus": 13, "bess_bus": 15},
    "MG4": {"diesel_bus": 19, "pv_bus": 22, "wind_bus": 24, "bess_bus": 25},
    "MG5": {"diesel_bus": 27, "pv_bus": 30, "wind_bus": 33, "bess_bus": 32},
}

IEEE33_BUS_LOADS_KW = {
    1: (0.0, 0.0),
    2: (100.0, 60.0),
    3: (90.0, 40.0),
    4: (120.0, 80.0),
    5: (60.0, 30.0),
    6: (60.0, 20.0),
    7: (200.0, 100.0),
    8: (200.0, 100.0),
    9: (60.0, 20.0),
    10: (60.0, 20.0),
    11: (45.0, 30.0),
    12: (60.0, 35.0),
    13: (60.0, 35.0),
    14: (120.0, 80.0),
    15: (60.0, 10.0),
    16: (60.0, 20.0),
    17: (60.0, 20.0),
    18: (90.0, 40.0),
    19: (90.0, 40.0),
    20: (90.0, 40.0),
    21: (90.0, 40.0),
    22: (90.0, 40.0),
    23: (90.0, 50.0),
    24: (420.0, 200.0),
    25: (420.0, 200.0),
    26: (60.0, 25.0),
    27: (60.0, 25.0),
    28: (60.0, 20.0),
    29: (120.0, 70.0),
    30: (200.0, 600.0),
    31: (150.0, 70.0),
    32: (210.0, 100.0),
    33: (60.0, 40.0),
}

IEEE33_BRANCHES = [
    (1, 2, 0.0922, 0.0470),
    (2, 3, 0.4930, 0.2511),
    (3, 4, 0.3660, 0.1864),
    (4, 5, 0.3811, 0.1941),
    (5, 6, 0.8190, 0.7070),
    (6, 7, 0.1872, 0.6188),
    (7, 8, 0.7114, 0.2351),
    (8, 9, 1.0300, 0.7400),
    (9, 10, 1.0440, 0.7400),
    (10, 11, 0.1966, 0.0650),
    (11, 12, 0.3744, 0.1238),
    (12, 13, 1.4680, 1.1550),
    (13, 14, 0.5416, 0.7129),
    (14, 15, 0.5910, 0.5260),
    (15, 16, 0.7463, 0.5450),
    (16, 17, 1.2890, 1.7210),
    (17, 18, 0.7320, 0.5740),
    (2, 19, 0.1640, 0.1565),
    (19, 20, 1.5042, 1.3554),
    (20, 21, 0.4095, 0.4784),
    (21, 22, 0.7089, 0.9373),
    (3, 23, 0.4512, 0.3083),
    (23, 24, 0.8980, 0.7091),
    (24, 25, 0.8960, 0.7011),
    (6, 26, 0.2030, 0.1034),
    (26, 27, 0.2842, 0.1447),
    (27, 28, 1.0590, 0.9337),
    (28, 29, 0.8042, 0.7006),
    (29, 30, 0.5075, 0.2585),
    (30, 31, 0.9744, 0.9630),
    (31, 32, 0.3105, 0.3619),
    (32, 33, 0.3410, 0.5302),
]


def build_ieee33_config(values: TypingDict[str, Any]) -> TypingDict[str, Any]:
    missing = sorted(REQUIRED_ENV_FIELDS.difference(values))
    if missing:
        raise ValueError(f"Missing environment fields: {', '.join(missing)}")
    cfg = dict(values)
    cfg["bus_partitions"] = dict(DEFAULT_PARTITIONS)
    cfg["device_map"] = dict(DEFAULT_DEVICE_MAP)
    return cfg


class IEEE33SingleMGEnv(gym.Env):
    def __init__(
        self,
        data_path: str,
        config: TypingDict[str, Any],
        mg_id: int,
    ):
        super().__init__()
        self.config = build_ieee33_config(config)
        self.data = pd.read_csv(data_path)
        required_columns = {
            "household_power",
            "solar_power",
            "wind_power",
            "EUR/kWh",
        }
        missing_columns = sorted(required_columns.difference(self.data.columns))
        if missing_columns:
            raise ValueError(f"Missing dataset columns: {', '.join(missing_columns)}")
        self._prepare_dataframe()

        self.nrows = int(self.config["nrows"])
        self.data = self.data.iloc[: min(self.nrows, len(self.data))].copy()

        self.num_microgrids = len(DEFAULT_PARTITIONS)
        self.mg_id = int(mg_id)
        if not 1 <= self.mg_id <= self.num_microgrids:
            raise ValueError(
                f"mg_id must be in [1, {self.num_microgrids}], got {self.mg_id}"
            )
        self.idx = self.mg_id - 1
        self.zone_name = f"MG{self.mg_id}"
        self.bus_partitions = self.config["bus_partitions"]
        self.device_map = self.config["device_map"]
        self.buses = list(self.bus_partitions[self.zone_name])
        self.devices = dict(self.device_map[self.zone_name])

        self.simulation_steps = len(self.data)
        self.current_step = 0
        self.time_step = float(self.config["time_step"])

        self.zone_load_weights = np.asarray(
            self.config["zone_load_weights"], dtype=float
        )
        self.zone_pv_weights = np.asarray(self.config["zone_pv_weights"], dtype=float)
        self.zone_wind_weights = np.asarray(
            self.config["zone_wind_weights"], dtype=float
        )
        self.zone_load_weights /= self.zone_load_weights.sum()
        self.zone_pv_weights /= self.zone_pv_weights.sum()
        self.zone_wind_weights /= self.zone_wind_weights.sum()

        def arr(key: str) -> np.ndarray:
            values = np.asarray(self.config[key], dtype=float)
            if values.shape != (self.num_microgrids,):
                raise ValueError(
                    f"{key} must contain {self.num_microgrids} microgrid values"
                )
            return values

        self.pv_capacity = arr("pv_capacity")[self.idx]
        self.wind_capacity = arr("wind_capacity")[self.idx]
        self.diesel_capacity = arr("diesel_capacity")[self.idx]
        self.battery_capacity = arr("battery_capacity")[self.idx]
        self.battery_max_power_ratio = arr("battery_max_power_ratio")[self.idx]
        self.battery_soc_min = arr("battery_soc_min")[self.idx]
        self.battery_soc_max = arr("battery_soc_max")[self.idx]
        self.battery_efficiency = arr("battery_efficiency")[self.idx]
        self.initial_soc = arr("initial_soc")[self.idx]
        self.diesel_cost = arr("diesel_cost")[self.idx]
        self.diesel_fuel_capacity = arr("diesel_fuel_capacity")[self.idx]
        self.fuel_consumption_rate = arr("fuel_consumption_rate")[self.idx]

        self.line_capacity_pv = arr("line_capacity_pv")[self.idx]
        self.line_capacity_wind = arr("line_capacity_wind")[self.idx]
        self.line_capacity_diesel = arr("line_capacity_diesel")[self.idx]
        self.line_capacity_battery = arr("line_capacity_battery")[self.idx]
        self.line_capacity_grid = arr("line_capacity_grid")[self.idx]
        self.line_capacity_load = arr("line_capacity_load")[self.idx]
        self.line_safety_margin = float(self.config["line_safety_margin"])
        self.branch_capacity_mva = float(self.config["branch_capacity_mva"])

        self.grid_buy_price = float(self.config["grid_buy_price"])
        self.grid_sell_price = float(self.config["grid_sell_price"])

        self.load_demand_scale = float(self.config["load_demand_scale"])
        self.curtailment_penalty = float(self.config["curtailment_penalty"])
        self.load_shedding_penalty = float(self.config["load_shedding_penalty"])
        self.capacity_violation_weight = float(self.config["capacity_violation_weight"])
        self.network_loss_penalty_coef = float(self.config["network_loss_penalty_coef"])
        self.voltage_violation_weight = float(self.config["voltage_violation_weight"])
        self.slack_import_cost_coef = float(self.config["slack_import_cost_coef"])
        self.voltage_min_pu = float(self.config["voltage_min_pu"])
        self.voltage_max_pu = float(self.config["voltage_max_pu"])
        self.base_kv = float(self.config["base_kv"])
        self.base_mva = float(self.config["base_mva"])
        self.power_flow_max_iterations = int(self.config["power_flow_max_iterations"])
        self.power_flow_tolerance = float(self.config["power_flow_tolerance"])
        self.power_flow_relaxation = float(self.config["power_flow_relaxation"])
        if self.power_flow_max_iterations <= 0:
            raise ValueError("power_flow_max_iterations must be positive")
        if self.power_flow_tolerance <= 0.0:
            raise ValueError("power_flow_tolerance must be positive")
        if not 0.0 < self.power_flow_relaxation <= 1.0:
            raise ValueError("power_flow_relaxation must be in (0, 1]")
        self.load_q_ratio_scale = float(self.config["load_q_ratio_scale"])
        self.action_gap_penalty_coef = float(self.config["action_gap_penalty_coef"])

        self._init_network_model()

        self.episode_horizon = int(self.config["episode_horizon"])
        self.random_episode_start = bool(self.config["random_episode_start"])
        self.reward_offset = float(self.config["reward_offset"])
        self.reward_scale = float(self.config["reward_scale"])
        self.reward_min = float(self.config["reward_min"])
        self.reward_max = float(self.config["reward_max"])
        self.imbalance_penalty_coef = float(self.config["imbalance_penalty_coef"])
        self.action_ramp_penalty_coef = float(self.config["action_ramp_penalty_coef"])
        self.battery_action_deadband = float(self.config["battery_action_deadband"])
        self.diesel_ramp_penalty_coef = float(self.config["diesel_ramp_penalty_coef"])
        self.grid_import_penalty_coef = float(self.config["grid_import_penalty_coef"])
        self.grid_export_penalty_coef = float(self.config["grid_export_penalty_coef"])
        self.soc_target = float(self.config["soc_target"])
        self.soc_deviation_penalty_coef = float(
            self.config["soc_deviation_penalty_coef"]
        )
        self.price_normalization_reference = float(
            self.config["price_normalization_reference"]
        )

        self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(75,), dtype=np.float32
        )
        _ = self.reset(seed=None)

    def _init_network_model(self) -> None:
        z_base_ohm = (self.base_kv**2) / self.base_mva
        self.network_buses = list(range(1, 34))
        self.slack_bus = 1
        self.bus_p_base = np.array(
            [IEEE33_BUS_LOADS_KW[b][0] for b in self.network_buses], dtype=float
        )
        self.bus_q_base = np.array(
            [IEEE33_BUS_LOADS_KW[b][1] for b in self.network_buses], dtype=float
        )
        self.bus_load_weights_within_mg: TypingDict[str, np.ndarray] = {}
        for zone_name, buses in self.bus_partitions.items():
            w = np.array([IEEE33_BUS_LOADS_KW[b][0] for b in buses], dtype=float)
            if w.sum() <= 0:
                w = np.ones(len(buses), dtype=float)
            self.bus_load_weights_within_mg[zone_name] = w / w.sum()

        self.branches = []
        self.children: TypingDict[int, List[int]] = {b: [] for b in self.network_buses}
        self.parent: TypingDict[int, int | None] = {b: None for b in self.network_buses}
        self.branch_map = {}
        for fbus, tbus, r_ohm, x_ohm in IEEE33_BRANCHES:
            branch = {
                "from_bus": int(fbus),
                "to_bus": int(tbus),
                "r_ohm": float(r_ohm),
                "x_ohm": float(x_ohm),
                "r_pu": float(r_ohm / z_base_ohm),
                "x_pu": float(x_ohm / z_base_ohm),
            }
            self.branches.append(branch)
            self.children[int(fbus)].append(int(tbus))
            self.parent[int(tbus)] = int(fbus)
            self.branch_map[(int(fbus), int(tbus))] = branch
        self.branch_order = [(b["from_bus"], b["to_bus"]) for b in self.branches]
        self.post_order = list(reversed(self.branch_order))

    def _prepare_dataframe(self):
        if "time" in self.data.columns:
            self.data["time"] = pd.to_datetime(self.data["time"])
            self.data = (
                self.data.sort_values("time")
                .drop_duplicates(subset="time", keep="last")
                .set_index("time")
            )
        elif not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.date_range(
                "2018-01-01", periods=len(self.data), freq="15min"
            )

    def _profile_split(self, total_value: float, weights: np.ndarray) -> np.ndarray:
        return total_value * weights

    def _all_zone_inputs(self):
        row = self.data.iloc[self.current_step]
        total_pv = float(row["solar_power"]) * 1000.0
        total_wind = float(row["wind_power"]) * 1000.0
        total_load = float(row["household_power"]) * 1000.0 * self.load_demand_scale

        price = float(row["EUR/kWh"]) if "EUR/kWh" in row else self.grid_buy_price

        pv_all = self._profile_split(total_pv, self.zone_pv_weights)
        wind_all = self._profile_split(total_wind, self.zone_wind_weights)
        load_all = self._profile_split(total_load, self.zone_load_weights)

        pv_all = np.minimum(pv_all, self.config["pv_capacity"])
        wind_all = np.minimum(wind_all, self.config["wind_capacity"])
        return pv_all, wind_all, load_all, price

    def _zone_inputs(self):
        pv_all, wind_all, load_all, price = self._all_zone_inputs()
        return (
            float(pv_all[self.idx]),
            float(wind_all[self.idx]),
            float(load_all[self.idx]),
            float(price),
        )

    def _check_action(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size != 3:
            raise ValueError(f"Action must contain three values, got {a.size}")
        return np.clip(a, -1.0, 1.0)

    def _obs(
        self,
        pv,
        wind,
        load,
        diesel_power,
        grid_power,
        buy_price,
        sell_price,
        pf,
    ):
        max_load = max(
            self.pv_capacity
            + self.wind_capacity
            + self.diesel_capacity
            + self.line_capacity_grid,
            1e-6,
        )
        voltage_state = np.asarray(pf["bus_voltage_pu"], dtype=np.float32)
        line_state = np.asarray(
            [pf["line_loading_pu"][branch] for branch in self.branch_order],
            dtype=np.float32,
        )
        obs = np.concatenate(
            [
                np.array(
                    [
                        pv / max(self.pv_capacity, 1e-6),
                        wind / max(self.wind_capacity, 1e-6),
                        min(load / max_load, 2.0),
                        self.battery_soc,
                        diesel_power / max(self.diesel_capacity, 1e-6),
                        self.diesel_fuel_level / 100.0,
                        grid_power / max(self.line_capacity_grid, 1e-6),
                    ],
                    dtype=np.float32,
                ),
                voltage_state,
                line_state,
                np.array(
                    [
                        pf["total_loss_kw"] / max(float(self.bus_p_base.sum()), 1e-6),
                        buy_price / self.price_normalization_reference,
                        sell_price / self.price_normalization_reference,
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        return obs

    def _build_bus_injections(self, zone_snapshot, current_actions):
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
                q_load_kvar[idx] += (
                    zone_load_bus[offset] * q_ratio * self.load_q_ratio_scale
                )

            devices = self.device_map[zone_name]
            p_gen_kw[devices["pv_bus"] - 1] += zone["pv_kw"]
            p_gen_kw[devices["wind_bus"] - 1] += zone["wind_kw"]

        devices = self.device_map[self.zone_name]
        p_gen_kw[devices["diesel_bus"] - 1] += max(
            0.0, current_actions.get("diesel_power_kw", 0.0)
        )

        batt_p = current_actions.get("battery_power_kw", 0.0)
        if batt_p >= 0:
            p_gen_kw[devices["bess_bus"] - 1] += batt_p
        else:
            p_load_kw[devices["bess_bus"] - 1] += abs(batt_p)

        controlled_load_shedding = max(
            0.0, current_actions.get("load_shedding_kw", 0.0)
        )
        controlled_curtail = max(0.0, current_actions.get("curtailment_kw", 0.0))

        if controlled_load_shedding > 0:
            buses = self.bus_partitions[self.zone_name]
            weights = self.bus_load_weights_within_mg[self.zone_name]
            for offset, bus in enumerate(buses):
                p_load_kw[bus - 1] = max(
                    0.0, p_load_kw[bus - 1] - controlled_load_shedding * weights[offset]
                )
                base_p, base_q = IEEE33_BUS_LOADS_KW[bus]
                q_ratio = (base_q / base_p) if base_p > 1e-9 else 0.0
                q_load_kvar[bus - 1] = max(
                    0.0,
                    q_load_kvar[bus - 1]
                    - controlled_load_shedding
                    * weights[offset]
                    * q_ratio
                    * self.load_q_ratio_scale,
                )
        if controlled_curtail > 0:
            ren_total = (
                zone_snapshot[self.zone_name]["pv_kw"]
                + zone_snapshot[self.zone_name]["wind_kw"]
            )
            if ren_total > 1e-9:
                pv_cut = (
                    controlled_curtail
                    * zone_snapshot[self.zone_name]["pv_kw"]
                    / ren_total
                )
                wind_cut = (
                    controlled_curtail
                    * zone_snapshot[self.zone_name]["wind_kw"]
                    / ren_total
                )
                p_gen_kw[self.device_map[self.zone_name]["pv_bus"] - 1] = max(
                    0.0,
                    p_gen_kw[self.device_map[self.zone_name]["pv_bus"] - 1] - pv_cut,
                )
                p_gen_kw[self.device_map[self.zone_name]["wind_bus"] - 1] = max(
                    0.0,
                    p_gen_kw[self.device_map[self.zone_name]["wind_bus"] - 1]
                    - wind_cut,
                )

        p_net_mw = (p_load_kw - p_gen_kw) / 1000.0
        q_net_mvar = q_load_kvar / 1000.0
        return {
            "p_load_kw": p_load_kw,
            "q_load_kvar": q_load_kvar,
            "p_gen_kw": p_gen_kw,
            "p_net_mw": p_net_mw,
            "q_net_mvar": q_net_mvar,
        }

    def _run_power_flow(self, p_net_mw, q_net_mvar):
        p_bus = np.asarray(p_net_mw, dtype=float) / self.base_mva
        q_bus = np.asarray(q_net_mvar, dtype=float) / self.base_mva
        v_sq = np.ones(33, dtype=float)
        line_loading = {}

        def backward_sweep(voltage_sq):
            active = {bus: float(p_bus[bus - 1]) for bus in self.network_buses}
            reactive = {bus: float(q_bus[bus - 1]) for bus in self.network_buses}
            active_flow, reactive_flow, active_loss = {}, {}, {}
            for from_bus, to_bus in self.post_order:
                branch = self.branch_map[(from_bus, to_bus)]
                p_down = active[to_bus]
                q_down = reactive[to_bus]
                current_sq = (p_down**2 + q_down**2) / max(
                    voltage_sq[from_bus - 1], 1e-9
                )
                p_loss = branch["r_pu"] * current_sq
                active_flow[(from_bus, to_bus)] = p_down + p_loss
                reactive_flow[(from_bus, to_bus)] = q_down + branch["x_pu"] * current_sq
                active_loss[(from_bus, to_bus)] = p_loss
                active[from_bus] += active_flow[(from_bus, to_bus)]
                reactive[from_bus] += reactive_flow[(from_bus, to_bus)]
            return active, reactive, active_flow, reactive_flow, active_loss

        for _ in range(self.power_flow_max_iterations):
            p_acc, q_acc, p_flow, q_flow, loss_pu = backward_sweep(v_sq)
            next_v_sq = np.ones(33, dtype=float)
            for from_bus, to_bus in self.branch_order:
                branch = self.branch_map[(from_bus, to_bus)]
                from_voltage_sq = max(next_v_sq[from_bus - 1], 1e-9)
                active_flow = p_flow[(from_bus, to_bus)]
                reactive_flow = q_flow[(from_bus, to_bus)]
                drop = 2.0 * (
                    branch["r_pu"] * active_flow + branch["x_pu"] * reactive_flow
                )
                correction = (
                    (branch["r_pu"] ** 2 + branch["x_pu"] ** 2)
                    * (active_flow**2 + reactive_flow**2)
                    / from_voltage_sq
                )
                next_v_sq[to_bus - 1] = max(1e-9, from_voltage_sq - drop + correction)
            relaxed = (
                self.power_flow_relaxation * next_v_sq
                + (1.0 - self.power_flow_relaxation) * v_sq
            )
            relaxed[0] = 1.0
            if float(np.max(np.abs(relaxed - v_sq))) <= self.power_flow_tolerance:
                v_sq = relaxed
                break
            v_sq = relaxed

        p_acc, q_acc, p_flow, q_flow, loss_pu = backward_sweep(v_sq)
        for branch in self.branch_order:
            apparent_power = np.hypot(
                p_flow[branch] * self.base_mva,
                q_flow[branch] * self.base_mva,
            )
            line_loading[branch] = apparent_power / self.branch_capacity_mva

        v = np.sqrt(v_sq)
        total_loss_kw = sum(loss_pu.values()) * self.base_mva * 1000.0
        slack_p_mw = p_acc[self.slack_bus] * self.base_mva
        slack_q_mvar = q_acc[self.slack_bus] * self.base_mva
        voltage_violation = np.maximum(0.0, self.voltage_min_pu - v) + np.maximum(
            0.0, v - self.voltage_max_pu
        )

        return {
            "bus_voltage_pu": v,
            "line_p_flow_mw": {k: p_flow[k] * self.base_mva for k in p_flow},
            "line_q_flow_mvar": {k: q_flow[k] * self.base_mva for k in q_flow},
            "line_loading_pu": line_loading,
            "line_loss_kw": {k: loss_pu[k] * self.base_mva * 1000.0 for k in loss_pu},
            "total_loss_kw": total_loss_kw,
            "slack_p_mw": slack_p_mw,
            "slack_q_mvar": slack_q_mvar,
            "v_min_pu": float(v.min()),
            "v_max_pu": float(v.max()),
            "avg_voltage_dev_pu": float(np.mean(np.abs(v - 1.0))),
            "voltage_violation_sum": float(voltage_violation.sum()),
        }

    def _internal_loss_kw(self, pf: TypingDict[str, Any]) -> float:
        bus_set = set(self.buses)
        return float(
            sum(
                loss
                for (from_bus, to_bus), loss in pf["line_loss_kw"].items()
                if from_bus in bus_set and to_bus in bus_set
            )
        )

    def _power_flow_from_dispatch(
        self,
        zone_snapshot,
        diesel_power_kw,
        battery_power_kw,
        curtailment_kw=0.0,
        load_shedding_kw=0.0,
    ):
        bus_inj = self._build_bus_injections(
            zone_snapshot,
            {
                "battery_power_kw": float(battery_power_kw),
                "diesel_power_kw": float(diesel_power_kw),
                "curtailment_kw": float(curtailment_kw),
                "load_shedding_kw": float(load_shedding_kw),
            },
        )
        pf = self._run_power_flow(bus_inj["p_net_mw"], bus_inj["q_net_mvar"])
        return {"bus_inj": bus_inj, "pf": pf}

    def _build_action_bounds(self):
        max_batt = self.battery_capacity * self.battery_max_power_ratio
        soc_discharge_max = max(
            0.0,
            (self.battery_soc - self.battery_soc_min)
            * self.battery_capacity
            * self.battery_efficiency,
        ) / max(self.time_step, 1e-9)
        soc_charge_max = max(
            0.0, (self.battery_soc_max - self.battery_soc) * self.battery_capacity
        ) / max(self.battery_efficiency * self.time_step, 1e-9)
        batt_min = -min(max_batt, soc_charge_max)
        batt_max = min(max_batt, soc_discharge_max)

        diesel_max = self.diesel_capacity
        fuel_lim = (
            (self.diesel_fuel_level / 100.0)
            * self.diesel_fuel_capacity
            / max(self.time_step, 1e-9)
        )
        diesel_max = min(diesel_max, fuel_lim)

        grid_max = self.line_capacity_grid
        return {
            "diesel_min": 0.0,
            "diesel_max": max(0.0, diesel_max),
            "battery_min": float(batt_min),
            "battery_max": float(batt_max),
            "grid_min": -float(grid_max),
            "grid_max": float(grid_max),
        }

    def _raw_action_to_target_physical(self, raw_action, bounds):
        diesel_cmd, batt_cmd, grid_cmd = self._check_action(raw_action)
        batt_target = (
            0.0
            if abs(float(batt_cmd)) < self.battery_action_deadband
            else float(batt_cmd)
            * max(abs(bounds["battery_min"]), abs(bounds["battery_max"]))
        )
        diesel_target = 0.5 * (float(diesel_cmd) + 1.0) * bounds["diesel_max"]
        grid_target = float(grid_cmd) * max(
            abs(bounds["grid_min"]), abs(bounds["grid_max"])
        )
        return np.array([diesel_target, batt_target, grid_target], dtype=float)

    def _physical_action_to_policy_action(self, diesel, batt, grid, bounds):
        batt_scale = max(abs(bounds["battery_min"]), abs(bounds["battery_max"]), 1e-6)
        grid_scale = max(abs(bounds["grid_min"]), abs(bounds["grid_max"]), 1e-6)
        diesel_scale = max(bounds["diesel_max"], 1e-6)
        batt_cmd = np.clip(batt / batt_scale, -1.0, 1.0)
        diesel_cmd = (
            np.clip(2.0 * diesel / diesel_scale - 1.0, -1.0, 1.0)
            if diesel_scale > 1e-9
            else -1.0
        )
        grid_cmd = np.clip(grid / grid_scale, -1.0, 1.0)
        return np.array([diesel_cmd, batt_cmd, grid_cmd], dtype=np.float32)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        horizon = int(min(self.episode_horizon, self.simulation_steps))
        if horizon <= 0:
            raise ValueError("episode_horizon must be positive")
        max_start = max(0, self.simulation_steps - horizon)
        self.start_step = (
            int(np.random.randint(0, max_start + 1))
            if self.random_episode_start and max_start > 0
            else 0
        )
        self.end_step = int(min(self.simulation_steps, self.start_step + horizon))

        self.current_step = int(self.start_step)
        self.episode_step = 0
        self.battery_soc = float(self.initial_soc)
        self.diesel_fuel_level = 100.0
        self.cumulative_cost = 0.0
        self.operation_history = []
        self.balance_errors = []
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.prev_diesel_power = 0.0

        pv, wind, load, price = self._zone_inputs()
        zone_pv, zone_wind, zone_load, _ = self._all_zone_inputs()
        zone_snapshot = {
            f"MG{i+1}": {
                "pv_kw": float(zone_pv[i]),
                "wind_kw": float(zone_wind[i]),
                "load_kw": float(zone_load[i]),
            }
            for i in range(self.num_microgrids)
        }
        bus_inj = self._build_bus_injections(
            zone_snapshot,
            {
                "battery_power_kw": 0.0,
                "diesel_power_kw": 0.0,
                "curtailment_kw": 0.0,
                "load_shedding_kw": 0.0,
            },
        )
        self.last_pf = self._run_power_flow(bus_inj["p_net_mw"], bus_inj["q_net_mvar"])

        sell_price = price * (self.grid_sell_price / self.grid_buy_price)
        return self._obs(pv, wind, load, 0.0, 0.0, price, sell_price, self.last_pf)

    def step(self, action):
        raw_action = self._check_action(action)
        zone_pv, zone_wind, zone_load, price = self._all_zone_inputs()
        zone_snapshot = {
            f"MG{i+1}": {
                "pv_kw": float(zone_pv[i]),
                "wind_kw": float(zone_wind[i]),
                "load_kw": float(zone_load[i]),
            }
            for i in range(self.num_microgrids)
        }

        pv = zone_snapshot[self.zone_name]["pv_kw"]
        wind = zone_snapshot[self.zone_name]["wind_kw"]
        load = zone_snapshot[self.zone_name]["load_kw"]
        renewable = pv + wind
        rhs = load - renewable

        bounds = self._build_action_bounds()
        target_action = self._raw_action_to_target_physical(raw_action, bounds)

        rhs_feasible = float(
            np.clip(
                rhs,
                bounds["diesel_min"] + bounds["battery_min"] + bounds["grid_min"],
                bounds["diesel_max"] + bounds["battery_max"] + bounds["grid_max"],
            )
        )
        diesel_target = float(
            np.clip(target_action[0], bounds["diesel_min"], bounds["diesel_max"])
        )
        batt_power = float(
            np.clip(target_action[1], bounds["battery_min"], bounds["battery_max"])
        )
        grid_target = float(
            np.clip(target_action[2], bounds["grid_min"], bounds["grid_max"])
        )

        residual = rhs_feasible - batt_power
        diesel_min_feasible = max(bounds["diesel_min"], residual - bounds["grid_max"])
        diesel_max_feasible = min(bounds["diesel_max"], residual - bounds["grid_min"])

        if diesel_min_feasible <= diesel_max_feasible:
            diesel_power = float(
                np.clip(diesel_target, diesel_min_feasible, diesel_max_feasible)
            )
            grid_power = float(
                np.clip(residual - diesel_power, bounds["grid_min"], bounds["grid_max"])
            )
        else:
            diesel_power = diesel_target
            grid_power = grid_target

        preliminary = self._power_flow_from_dispatch(
            zone_snapshot, diesel_power, batt_power
        )
        grid_power = float(
            np.clip(
                grid_power + self._internal_loss_kw(preliminary["pf"]),
                bounds["grid_min"],
                bounds["grid_max"],
            )
        )

        supply_gap = (
            load
            + self._internal_loss_kw(preliminary["pf"])
            - (renewable + batt_power + diesel_power + grid_power)
        )
        load_shedding = min(load, supply_gap) if supply_gap > 0 else 0.0
        curtail = min(renewable, -supply_gap) if supply_gap <= 0 else 0.0
        served_load = max(load - load_shedding, 0.0)
        final_evaluation = self._power_flow_from_dispatch(
            zone_snapshot,
            diesel_power,
            batt_power,
            curtailment_kw=curtail,
            load_shedding_kw=load_shedding,
        )
        pf, bus_inj = final_evaluation["pf"], final_evaluation["bus_inj"]
        self.last_pf = pf
        internal_loss_kw = self._internal_loss_kw(pf)
        final_balance_error = (
            renewable
            + batt_power
            + diesel_power
            + grid_power
            - curtail
            - served_load
            - internal_loss_kw
        )

        dsoc = (
            -(batt_power * self.time_step)
            / max(self.battery_capacity * self.battery_efficiency, 1e-6)
            if batt_power >= 0
            else (abs(batt_power) * self.time_step * self.battery_efficiency)
            / max(self.battery_capacity, 1e-6)
        )
        self.battery_soc = float(
            np.clip(self.battery_soc + dsoc, self.battery_soc_min, self.battery_soc_max)
        )

        fuel_used = diesel_power * self.fuel_consumption_rate * self.time_step
        self.diesel_fuel_level = float(
            100.0
            * max(
                0.0,
                self.diesel_fuel_capacity * self.diesel_fuel_level / 100.0 - fuel_used,
            )
            / max(self.diesel_fuel_capacity, 1e-6)
        )

        dispatch_capacity_excess = sum(
            max(0.0, abs(v) - cap)
            for v, cap in [
                (pv, self.line_capacity_pv * self.line_safety_margin),
                (wind, self.line_capacity_wind * self.line_safety_margin),
                (diesel_power, self.line_capacity_diesel * self.line_safety_margin),
                (batt_power, self.line_capacity_battery * self.line_safety_margin),
                (grid_power, self.line_capacity_grid * self.line_safety_margin),
                (served_load, self.line_capacity_load * self.line_safety_margin),
            ]
        )
        branch_capacity_excess = 1000.0 * sum(
            max(
                0.0,
                np.hypot(
                    pf["line_p_flow_mw"][branch],
                    pf["line_q_flow_mvar"][branch],
                )
                - self.line_safety_margin * self.branch_capacity_mva,
            )
            for branch in self.branch_order
        )
        capacity_excess = dispatch_capacity_excess + branch_capacity_excess
        line_penalty = self.capacity_violation_weight * capacity_excess

        current_sell_price = (
            price * (self.grid_sell_price / self.grid_buy_price)
            if self.grid_buy_price > 1e-6
            else 0.0
        )

        diesel_cost = diesel_power * self.diesel_cost * self.time_step
        grid_import_cost = max(grid_power, 0.0) * price * self.time_step
        grid_export_credit = max(-grid_power, 0.0) * current_sell_price * self.time_step

        voltage_penalty = pf["voltage_violation_sum"] * self.voltage_violation_weight
        network_loss_penalty = (
            pf["total_loss_kw"] * self.network_loss_penalty_coef * self.time_step
        )
        slack_import_penalty = (
            max(0.0, pf["slack_p_mw"]) * self.slack_import_cost_coef * self.time_step
        )
        imbalance_penalty = (
            abs(final_balance_error) * self.imbalance_penalty_coef * self.time_step
        )

        executed_policy_action = self._physical_action_to_policy_action(
            diesel_power, batt_power, grid_power, bounds
        )
        ramp_penalty = self.action_ramp_penalty_coef * float(
            np.sum(np.abs(np.asarray(executed_policy_action) - self.prev_action))
        )
        diesel_ramp_penalty = (
            self.diesel_ramp_penalty_coef
            * abs(diesel_power - self.prev_diesel_power)
            * self.time_step
        )
        grid_usage_penalty = (
            max(grid_power, 0.0) * self.grid_import_penalty_coef * self.time_step
            + max(-grid_power, 0.0) * self.grid_export_penalty_coef * self.time_step
        )
        soc_deviation_penalty = self.soc_deviation_penalty_coef * abs(
            self.battery_soc - self.soc_target
        )
        action_gap_penalty = self.action_gap_penalty_coef * float(
            np.sum(
                np.abs(
                    np.asarray(target_action)
                    - np.array([diesel_power, batt_power, grid_power])
                )
            )
        )

        economic_cost = (
            diesel_cost
            + grid_import_cost
            - grid_export_credit
            + curtail * self.curtailment_penalty * self.time_step
            + load_shedding * self.load_shedding_penalty * self.time_step
            + network_loss_penalty
            + slack_import_penalty
            + imbalance_penalty
            + ramp_penalty
            + diesel_ramp_penalty
            + grid_usage_penalty
            + soc_deviation_penalty
            + action_gap_penalty
        )
        reward = float(
            np.clip(
                self.reward_offset - self.reward_scale * economic_cost,
                self.reward_min,
                self.reward_max,
            )
        )
        constraint_cost = float(voltage_penalty + line_penalty)
        self.cumulative_cost += economic_cost

        self.balance_errors.append(abs(final_balance_error))

        self.operation_history.append(
            {
                "timestamp": self.data.index[self.current_step],
                "mg": self.zone_name,
                "pv_power": pv,
                "wind_power": wind,
                "load_demand": load,
                "served_load_demand": served_load,
                "battery_power": batt_power,
                "diesel_power": diesel_power,
                "grid_power": grid_power,
                "battery_soc": self.battery_soc,
                "fuel_level": self.diesel_fuel_level,
                "retail_price": price,
                "buy_price": price,
                "sell_price": current_sell_price,
                "curtailment": curtail,
                "load_shedding": load_shedding,
                "power_balance_error": final_balance_error,
                "reward": reward,
                "cost": economic_cost,
                "economic_cost": economic_cost,
                "constraint_cost": constraint_cost,
                "voltage_penalty": voltage_penalty,
                "line_penalty": line_penalty,
                "dispatch_capacity_excess_kw": dispatch_capacity_excess,
                "branch_capacity_excess_kw": branch_capacity_excess,
                "internal_network_loss_kw": internal_loss_kw,
                "network_total_loss_kw": pf["total_loss_kw"],
                "network_v_min_pu": pf["v_min_pu"],
                "network_v_max_pu": pf["v_max_pu"],
                "network_avg_voltage_dev_pu": pf["avg_voltage_dev_pu"],
                "network_voltage_violation_sum": pf["voltage_violation_sum"],
                "slack_p_mw": pf["slack_p_mw"],
                "slack_q_mvar": pf["slack_q_mvar"],
                "bus_voltage_pu": pf["bus_voltage_pu"].tolist(),
                "bus_p_load_kw": bus_inj["p_load_kw"].tolist(),
                "bus_q_load_kvar": bus_inj["q_load_kvar"].tolist(),
                "bus_p_gen_kw": bus_inj["p_gen_kw"].tolist(),
                "line_loading_pu": {
                    f"{k[0]}->{k[1]}": float(v)
                    for k, v in pf["line_loading_pu"].items()
                },
                "line_p_flow_mw": {
                    f"{k[0]}->{k[1]}": float(v) for k, v in pf["line_p_flow_mw"].items()
                },
                "line_q_flow_mvar": {
                    f"{k[0]}->{k[1]}": float(v)
                    for k, v in pf["line_q_flow_mvar"].items()
                },
                "line_loss_kw": {
                    f"{k[0]}->{k[1]}": float(v) for k, v in pf["line_loss_kw"].items()
                },
            }
        )

        self.prev_action = executed_policy_action.astype(np.float32)
        self.prev_diesel_power = float(diesel_power)
        self.current_step += 1
        self.episode_step += 1
        done = (
            self.current_step >= self.end_step
            or self.current_step >= self.simulation_steps - 1
        )

        if done:
            next_pv, next_wind, next_load, next_price = pv, wind, load, price
        else:
            next_pv, next_wind, next_load, next_price = self._zone_inputs()

        info = {
            "mg": self.zone_name,
            "cumulative_cost": self.cumulative_cost,
            "constraint_cost": constraint_cost,
            "economic_cost": economic_cost,
            "net_pen": constraint_cost,
        }
        next_sell_price = next_price * (self.grid_sell_price / self.grid_buy_price)
        obs = self._obs(
            next_pv,
            next_wind,
            next_load,
            diesel_power,
            grid_power,
            next_price,
            next_sell_price,
            pf,
        )
        return obs, reward, done, info
