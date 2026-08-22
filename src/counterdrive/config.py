from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml


@dataclass
class DataConfig:
    backend: str = "synthetic"
    root: str = "data/nuscenes"
    version: str = "v1.0-mini"
    sequence_length: int = 4
    future_steps: int = 6
    image_size: int = 96
    train_samples: int = 256
    val_samples: int = 64
    num_workers: int = 0


@dataclass
class ModelConfig:
    pretrained: bool = True
    freeze_vision: bool = True
    unfreeze_layer4: bool = False
    latent_dim: int = 128
    action_dim: int = 3
    transformer_layers: int = 2
    transformer_heads: int = 4
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    trajectory_loss_weight: float = 1.0
    collision_loss_weight: float = 0.5
    latent_loss_weight: float = 0.1
    checkpoint_dir: str = "checkpoints"


@dataclass
class Config:
    seed: int = 42
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @property
    def resolved_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


def load_config(path: str | Path) -> Config:
    with Path(path).open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    return Config(
        seed=raw.get("seed", 42),
        device=raw.get("device", "auto"),
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
    )

