from __future__ import annotations

import argparse
import json
import platform
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from counterdrive.config import Config, load_config
from counterdrive.data import build_dataloaders
from counterdrive.engine import compute_loss, evaluate_model, move_batch
from counterdrive.model import CounterDriveModel


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=True),
        encoding="utf-8",
    )


def create_grad_scaler(enabled: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def build_checkpoint(
    model: CounterDriveModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: Any,
    config: Config,
    epoch: int,
    best_metric: float,
    history: list[dict[str, float]],
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": asdict(config),
        "epoch": epoch,
        "best_metric": best_metric,
        "history": history,
    }


def train(config: Config, resume: str | None = None) -> Path:
    seed_everything(config.seed)
    device = config.resolved_device
    amp_enabled = config.training.mixed_precision and device.startswith("cuda")
    train_loader, val_loader = build_dataloaders(config)
    model = CounterDriveModel(config.model, config.data.future_steps).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.training.scheduler_factor,
        patience=config.training.scheduler_patience,
    )
    scaler = create_grad_scaler(amp_enabled)
    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    history_path = checkpoint_dir / "history.json"
    metadata_path = checkpoint_dir / "run_metadata.json"

    start_epoch, epochs_without_improvement = 1, 0
    best_metric = float("inf")
    history: list[dict[str, float]] = []
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint.get("scaler", {}))
            start_epoch = int(checkpoint["epoch"]) + 1
            best_metric = float(checkpoint["best_metric"])
            history = checkpoint.get("history", [])

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "config": asdict(config),
        "resumed_from": resume,
    }
    save_json(metadata_path, metadata)

    for epoch in range(start_epoch, config.training.epochs + 1):
        model.train()
        running_loss = 0.0
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.split(":")[0],
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                outputs = model(batch["frames"], batch["actions"])
                loss, _ = compute_loss(
                    outputs,
                    batch,
                    config.training.trajectory_loss_weight,
                    config.training.collision_loss_weight,
                    config.training.latent_loss_weight,
                    config.training.collision_positive_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.training.gradient_clip_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        metrics = evaluate_model(model, val_loader, device)
        monitored = float(metrics[config.training.monitor])
        scheduler.step(monitored)
        epoch_result = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_loader),
            "learning_rate": optimizer.param_groups[0]["lr"],
            **metrics,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, allow_nan=True))
        improved = monitored < best_metric
        if improved:
            best_metric = monitored
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        checkpoint = build_checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            config,
            epoch,
            best_metric,
            history,
        )
        torch.save(checkpoint, last_path)
        if improved:
            torch.save(checkpoint, best_path)
        save_json(history_path, history)
        if epochs_without_improvement >= config.training.early_stopping_patience:
            print(f"Early stopping after epoch {epoch}")
            break
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    best_path = train(load_config(args.config), resume=args.resume)
    print(f"Saved best checkpoint to {best_path}")


if __name__ == "__main__":
    main()
