import json
from pathlib import Path

import numpy as np

from src.ieee33_mmg_env import IEEE33SingleMGEnv


ROOT = Path(__file__).resolve().parents[1]


def make_environment(mg_id):
    config = json.loads(
        (ROOT / "configs" / "environment.paper.json").read_text(encoding="utf-8")
    )
    return IEEE33SingleMGEnv(
        str(ROOT / "tests" / "fixtures" / "environment_sample.csv"), config, mg_id
    )


def test_all_microgrids_share_paper_spaces_and_cost_interface():
    for mg_id in range(1, 6):
        environment = make_environment(mg_id)
        observation = environment.reset()
        next_observation, reward, done, info = environment.step(
            np.zeros(3, dtype=np.float32)
        )
        assert observation.shape == (75,)
        assert next_observation.shape == (75,)
        assert environment.action_space.shape == (3,)
        assert np.isfinite(observation).all()
        assert np.isfinite(next_observation).all()
        assert np.isfinite(reward)
        assert isinstance(done, bool)
        assert info["constraint_cost"] >= 0.0
        assert np.isfinite(info["economic_cost"])


def test_action_order_is_diesel_battery_grid():
    environment = make_environment(1)
    bounds = environment._build_action_bounds()
    target = environment._raw_action_to_target_physical(
        np.array([1.0, -1.0, 0.0], dtype=np.float32), bounds
    )
    assert target[0] == bounds["diesel_max"]
    assert target[1] == bounds["battery_min"]
    assert target[2] == 0.0
