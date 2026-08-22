import torch

from counterdrive.config import ModelConfig
from counterdrive.engine import compute_loss
from counterdrive.model import CounterDriveModel


def test_model_forward_and_backward() -> None:
    config = ModelConfig(
        pretrained=False,
        freeze_vision=True,
        latent_dim=32,
        transformer_layers=1,
        transformer_heads=4,
        dropout=0.0,
    )
    model = CounterDriveModel(config, future_steps=3)
    frames = torch.rand(2, 2, 3, 64, 64)
    actions = torch.rand(2, 3, 3)
    batch = {"future_trajectory": torch.rand(2, 3, 2), "collision": torch.tensor([0.0, 1.0])}
    outputs = model(frames, actions)
    assert outputs["trajectory"].shape == (2, 3, 2)
    assert outputs["collision_logits"].shape == (2,)
    loss, parts = compute_loss(outputs, batch)
    loss.backward()
    assert parts["loss"] > 0
    assert model.trajectory_head[-1].weight.grad is not None


def test_action_agnostic_baseline_ignores_actions() -> None:
    config = ModelConfig(
        pretrained=False,
        freeze_vision=True,
        latent_dim=32,
        transformer_layers=1,
        transformer_heads=4,
        dropout=0.0,
        action_conditioned=False,
    )
    model = CounterDriveModel(config, future_steps=3).eval()
    frames = torch.rand(1, 2, 3, 64, 64)
    first = model(frames, torch.zeros(1, 3, 3))["trajectory"]
    second = model(frames, torch.ones(1, 3, 3))["trajectory"]
    assert torch.allclose(first, second)
