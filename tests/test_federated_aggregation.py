import torch

from baselines.federated import aggregate_bundles
from src.ring_federated_qcmdp_training import RingFederatedAggregator


def test_qcmdp_aggregation_is_order_invariant():
    aggregator = RingFederatedAggregator(0.0)
    previous = {"q_weights": torch.zeros(2), "weight": torch.zeros(2)}
    first = {
        "q_weights": torch.tensor([3.0, -2.0]),
        "weight": torch.tensor([1.0, 3.0]),
    }
    second = {
        "q_weights": torch.tensor([-3.0, 2.0]),
        "weight": torch.tensor([3.0, 1.0]),
    }
    forward = aggregator.aggregate_state_dicts(previous, [first, second])
    reverse = aggregator.aggregate_state_dicts(previous, [second, first])
    for key in forward:
        assert torch.allclose(forward[key], reverse[key])


def test_baseline_aggregation_is_equal_weight_and_order_invariant():
    previous = {"actor": {"weight": torch.zeros(1)}}
    first = {"actor": {"weight": torch.tensor([1.0])}}
    second = {"actor": {"weight": torch.tensor([3.0])}}
    forward = aggregate_bundles(previous, [first, second], 0.0)
    reverse = aggregate_bundles(previous, [second, first], 0.0)
    assert torch.allclose(forward["actor"]["weight"], torch.tensor([2.0]))
    assert torch.allclose(forward["actor"]["weight"], reverse["actor"]["weight"])
