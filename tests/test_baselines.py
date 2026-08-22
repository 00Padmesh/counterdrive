import torch

from counterdrive.baselines import constant_velocity_prediction


def test_constant_velocity_prediction() -> None:
    past = torch.tensor([[[-2.0, -4.0], [-1.0, -2.0], [0.0, 0.0]]])
    prediction = constant_velocity_prediction(past, future_steps=3)
    expected = torch.tensor([[[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]])
    torch.testing.assert_close(prediction, expected)
