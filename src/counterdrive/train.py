from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from counterdrive.config import Config, load_config
from counterdrive.data import build_dataloaders
from counterdrive.engine import compute_loss, evaluate_model, move_batch
from counterdrive.model import CounterDriveModel


def train(config: Config) -> Path:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = config.resolved_device
    train_loader, val_loader = build_dataloaders(config)
    model = CounterDriveModel(config.model, config.data.future_steps).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    best_ade = float("inf")
    for epoch in range(1, config.training.epochs + 1):
        model.train()
        running_loss = 0.0
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["frames"], batch["actions"])
            loss, _ = compute_loss(
                outputs, batch, config.training.trajectory_loss_weight,
                config.training.collision_loss_weight, config.training.latent_loss_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
        metrics = evaluate_model(model, val_loader, device)
        epoch_result = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_loader),
            **metrics,
        }
        print(json.dumps(epoch_result))
        if metrics["ade"] < best_ade:
            best_ade = metrics["ade"]
            torch.save({"model": model.state_dict(), "config": config}, best_path)
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    args = parser.parse_args()
    print(f"Saved best checkpoint to {train(load_config(args.config))}")


if __name__ == "__main__":
    main()
