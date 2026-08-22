from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from counterdrive.config import Config, load_config
from counterdrive.data import SyntheticDrivingDataset, build_dataloaders
from counterdrive.engine import evaluate_model
from counterdrive.model import CounterDriveModel

SCENARIOS: dict[str, tuple[float, float, float]] = {
    "hard_brake": (0.0, 0.0, 1.0),
    "maintain": (0.0, 0.45, 0.0),
    "accelerate": (0.0, 1.0, 0.0),
    "turn_left": (-0.65, 0.35, 0.0),
    "turn_right": (0.65, 0.35, 0.0),
}


def load_model(config: Config, checkpoint_path: str) -> CounterDriveModel:
    model = CounterDriveModel(config.model, config.data.future_steps)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=config.resolved_device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(config.resolved_device).eval()
    return model


@torch.inference_mode()
def predict_scenarios(
    model: CounterDriveModel,
    frames: torch.Tensor,
    future_steps: int,
    device: str,
) -> dict[str, dict[str, object]]:
    frames = frames.unsqueeze(0).to(device)
    results: dict[str, dict[str, object]] = {}
    for name, action in SCENARIOS.items():
        actions = torch.tensor(action, dtype=torch.float32)
        actions = actions.repeat(future_steps, 1).unsqueeze(0).to(device)
        outputs = model(frames, actions)
        results[name] = {
            "action": list(action),
            "trajectory": outputs["trajectory"][0].cpu().tolist(),
            "collision_probability": torch.sigmoid(
                outputs["collision_logits"][0]
            ).item(),
        }
    return results


def render_trajectories(
    results: dict[str, dict[str, object]],
    output_path: Path,
) -> None:
    width, height = 800, 600
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    origin = np.array([width // 2, height - 60])
    colors = {
        "hard_brake": (40, 40, 220),
        "maintain": (80, 160, 80),
        "accelerate": (220, 120, 20),
        "turn_left": (180, 70, 180),
        "turn_right": (180, 160, 30),
    }
    cv2.line(canvas, (origin[0], 40), tuple(origin), (190, 190, 190), 2)
    cv2.putText(
        canvas,
        "CounterDrive action-conditioned futures",
        (25, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (30, 30, 30),
        2,
    )
    scale = 42.0
    for row, (name, result) in enumerate(results.items()):
        trajectory = np.asarray(result["trajectory"], dtype=np.float32)
        points = np.column_stack(
            (origin[0] + trajectory[:, 0] * scale, origin[1] - trajectory[:, 1] * scale)
        ).astype(np.int32)
        color = colors[name]
        cv2.polylines(canvas, [points], False, color, 3)
        for point in points:
            cv2.circle(canvas, tuple(point), 4, color, -1)
        risk = float(result["collision_probability"])
        label = f"{name}: risk={risk:.3f}"
        cv2.putText(
            canvas,
            label,
            (25, 65 + row * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"Unable to write visualization to {output_path}")


def action_sensitivity(results: dict[str, dict[str, object]]) -> dict[str, float]:
    final_positions = np.asarray(
        [result["trajectory"][-1] for result in results.values()],
        dtype=np.float32,
    )
    pairwise_distances = []
    for first in range(len(final_positions)):
        for second in range(first + 1, len(final_positions)):
            pairwise_distances.append(
                np.linalg.norm(final_positions[first] - final_positions[second])
            )
    risks = np.asarray(
        [result["collision_probability"] for result in results.values()],
        dtype=np.float32,
    )
    return {
        "mean_pairwise_final_displacement": float(np.mean(pairwise_distances)),
        "collision_probability_range": float(risks.max() - risks.min()),
    }


def run_experiment(
    config: Config,
    checkpoint_path: str,
    output_dir: Path,
    baseline_checkpoint: str | None = None,
) -> dict[str, object]:
    dataset = SyntheticDrivingDataset(
        samples=1,
        sequence_length=config.data.sequence_length,
        future_steps=config.data.future_steps,
        image_size=config.data.image_size,
        seed=config.seed + 200_000,
        collision_fraction=1.0,
    )
    sample = dataset[0]
    model = load_model(config, checkpoint_path)
    scenarios = predict_scenarios(
        model,
        sample["frames"],
        config.data.future_steps,
        config.resolved_device,
    )
    _, validation_loader = build_dataloaders(config)
    report: dict[str, object] = {
        "conditioned_metrics": evaluate_model(
            model, validation_loader, config.resolved_device
        ),
        "action_sensitivity": action_sensitivity(scenarios),
        "scenarios": scenarios,
    }
    if baseline_checkpoint:
        baseline_config = load_config_from(config, action_conditioned=False)
        baseline = load_model(baseline_config, baseline_checkpoint)
        report["action_agnostic_metrics"] = evaluate_model(
            baseline,
            validation_loader,
            baseline_config.resolved_device,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    render_trajectories(scenarios, output_dir / "counterfactual_trajectories.png")
    (output_dir / "counterfactual_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    return report


def load_config_from(config: Config, action_conditioned: bool) -> Config:
    config.model.action_conditioned = action_conditioned
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline-checkpoint", default=None)
    parser.add_argument("--output-dir", default="artifacts/counterfactual")
    args = parser.parse_args()
    config = load_config(args.config)
    report = run_experiment(
        config,
        args.checkpoint,
        Path(args.output_dir),
        args.baseline_checkpoint,
    )
    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
