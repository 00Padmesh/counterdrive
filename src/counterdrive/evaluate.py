from __future__ import annotations

import argparse
import json

import torch

from counterdrive.config import load_config
from counterdrive.data import build_dataloaders
from counterdrive.engine import evaluate_model
from counterdrive.model import CounterDriveModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()
    config = load_config(args.config)
    model = CounterDriveModel(config.model, config.data.future_steps).to(config.resolved_device)
    checkpoint = torch.load(args.checkpoint, map_location=config.resolved_device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    _, val_loader = build_dataloaders(config)
    print(json.dumps(evaluate_model(model, val_loader, config.resolved_device), indent=2))


if __name__ == "__main__":
    main()
