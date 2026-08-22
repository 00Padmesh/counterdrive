from __future__ import annotations

import argparse
import json

import torch

from counterdrive.config import load_config
from counterdrive.data import build_dataloaders
from counterdrive.metrics import collision_metrics, trajectory_metrics


def constant_velocity_prediction(
    past_trajectory: torch.Tensor,
    future_steps: int,
) -> torch.Tensor:
    if past_trajectory.shape[1] < 2:
        raise ValueError("Constant-velocity baseline requires at least two past steps")
    velocity = past_trajectory[:, -1] - past_trajectory[:, -2]
    steps = torch.arange(
        1,
        future_steps + 1,
        dtype=past_trajectory.dtype,
        device=past_trajectory.device,
    )
    return velocity.unsqueeze(1) * steps.view(1, -1, 1)


def evaluate_baselines(config_path: str) -> dict[str, dict[str, float]]:
    config = load_config(config_path)
    if config.data.backend != "nuscenes":
        raise ValueError("Real-data baselines require data.backend: nuscenes")
    _, validation_loader = build_dataloaders(config)
    targets, stationary_predictions, velocity_predictions, collisions = [], [], [], []
    for batch in validation_loader:
        target = batch["future_trajectory"]
        targets.append(target)
        collisions.append(batch["collision"])
        stationary_predictions.append(torch.zeros_like(target))
        velocity_predictions.append(
            constant_velocity_prediction(batch["past_trajectory"], target.shape[1])
        )
    target = torch.cat(targets)
    collision_target = torch.cat(collisions)
    return {
        "stationary": trajectory_metrics(
            torch.cat(stationary_predictions),
            target,
        ),
        "constant_velocity": trajectory_metrics(
            torch.cat(velocity_predictions),
            target,
        ),
        "always_no_risk": collision_metrics(
            torch.full_like(collision_target, -20.0),
            collision_target,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nuscenes_mini.yaml")
    args = parser.parse_args()
    print(json.dumps(evaluate_baselines(args.config), indent=2))


if __name__ == "__main__":
    main()
