import pytest
import torch

from counterdrive.metrics import collision_metrics, trajectory_metrics


def test_metrics_known_values() -> None:
    target = torch.zeros(1, 2, 2)
    prediction = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])
    metrics = trajectory_metrics(prediction, target)
    assert metrics["ade"] == pytest.approx(2.5)
    assert metrics["fde"] == pytest.approx(0.0)
    collision = collision_metrics(torch.tensor([-5.0, 5.0]), torch.tensor([0.0, 1.0]))
    assert collision["collision_accuracy"] == 1.0
    assert collision["collision_f1"] == pytest.approx(1.0)
    assert collision["collision_auroc"] == pytest.approx(1.0)
    assert collision["collision_average_precision"] == pytest.approx(1.0)
    assert collision["collision_best_f1"] == pytest.approx(1.0)
    assert 0.0 < collision["collision_best_threshold"] < 1.0
