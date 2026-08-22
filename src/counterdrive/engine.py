from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from counterdrive.metrics import collision_metrics, trajectory_metrics


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    trajectory_weight: float = 1.0,
    collision_weight: float = 0.5,
    latent_weight: float = 0.1,
    collision_positive_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    trajectory_loss = nn.functional.smooth_l1_loss(
        outputs["trajectory"], batch["future_trajectory"]
    )
    collision_loss = nn.functional.binary_cross_entropy_with_logits(
        outputs["collision_logits"],
        batch["collision"],
        pos_weight=outputs["collision_logits"].new_tensor(collision_positive_weight),
    )
    latents = outputs["future_latents"]
    if latents.shape[1] > 1:
        latent_loss = (latents[:, 1:] - latents[:, :-1]).square().mean()
    else:
        latent_loss = latents.new_tensor(0.0)
    total = (
        trajectory_weight * trajectory_loss
        + collision_weight * collision_loss
        + latent_weight * latent_loss
    )
    parts = {
        "loss": total.item(),
        "trajectory_loss": trajectory_loss.item(),
        "collision_loss": collision_loss.item(),
        "latent_loss": latent_loss.item(),
    }
    return total, parts


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: str,
) -> dict[str, float]:
    model.eval()
    trajectories, targets, logits, collisions = [], [], [], []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        outputs = model(
            batch["frames"],
            batch["actions"],
            batch.get("past_trajectory"),
        )
        trajectories.append(outputs["trajectory"].cpu())
        targets.append(batch["future_trajectory"].cpu())
        logits.append(outputs["collision_logits"].cpu())
        collisions.append(batch["collision"].cpu())
    result = trajectory_metrics(torch.cat(trajectories), torch.cat(targets))
    result.update(collision_metrics(torch.cat(logits), torch.cat(collisions)))
    return result
