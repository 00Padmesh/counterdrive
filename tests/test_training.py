import importlib

import torch
from torch import nn

from counterdrive.config import Config

train_module = importlib.import_module("counterdrive.train")


class TinyWorldModel(nn.Module):
    def __init__(self, _model_config, future_steps: int):
        super().__init__()
        self.future_steps = future_steps
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, frames: torch.Tensor, actions: torch.Tensor):
        batch_size = frames.shape[0]
        trajectory = self.scale * actions[..., :2]
        collision_logits = self.scale.repeat(batch_size)
        latents = self.scale.repeat(batch_size, self.future_steps, 2)
        return {
            "trajectory": trajectory,
            "collision_logits": collision_logits,
            "future_latents": latents,
        }


def test_training_checkpoint_and_resume(tmp_path, monkeypatch) -> None:
    batch = {
        "frames": torch.zeros(2, 2, 3, 8, 8),
        "actions": torch.ones(2, 2, 3),
        "future_trajectory": torch.ones(2, 2, 2),
        "collision": torch.tensor([0.0, 1.0]),
    }
    monkeypatch.setattr(train_module, "CounterDriveModel", TinyWorldModel)
    monkeypatch.setattr(train_module, "build_dataloaders", lambda _config: ([batch], [batch]))
    config = Config(device="cpu")
    config.data.future_steps = 2
    config.training.epochs = 1
    config.training.mixed_precision = False
    config.training.checkpoint_dir = str(tmp_path)

    best_path = train_module.train(config)
    last_path = tmp_path / "last.pt"
    assert best_path.exists()
    assert last_path.exists()
    assert (tmp_path / "history.json").exists()
    assert (tmp_path / "run_metadata.json").exists()

    config.training.epochs = 2
    train_module.train(config, resume=str(last_path))
    checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 2
    assert len(checkpoint["history"]) == 2

