from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from counterdrive.config import Config, load_config
from counterdrive.data import NuScenesSequenceDataset


def build_nuscenes_dataset(config: Config) -> NuScenesSequenceDataset:
    data = config.data
    if data.backend != "nuscenes":
        raise ValueError("The audit command requires data.backend: nuscenes")
    return NuScenesSequenceDataset(
        data.root,
        data.version,
        data.sequence_length,
        data.future_steps,
        data.image_size,
        data.cache_dir,
        data.use_cache,
    )


def render_sample(sample: dict, width: int = 720, height: int = 280) -> np.ndarray:
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    frame = sample["frames"][-1].permute(1, 2, 0).numpy()
    frame = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    camera_width = width // 2
    frame = cv2.resize(frame, (camera_width, height))
    canvas[:, :camera_width] = frame

    trajectory = sample["future_trajectory"].numpy()
    origin = np.asarray([camera_width + camera_width // 2, height - 30])
    scale = min(20.0, 200.0 / max(float(np.abs(trajectory).max()), 1.0))
    points = np.column_stack(
        (origin[0] + trajectory[:, 0] * scale, origin[1] - trajectory[:, 1] * scale)
    ).astype(np.int32)
    cv2.line(canvas, (origin[0], 20), tuple(origin), (180, 180, 180), 1)
    cv2.polylines(canvas, [points], False, (30, 150, 30), 3)
    for point in points:
        cv2.circle(canvas, tuple(point), 4, (30, 150, 30), -1)
    actions = sample["actions"].numpy()
    label = (
        f"collision={int(sample['collision'].item())}  "
        f"steer=[{actions[:, 0].min():.2f}, {actions[:, 0].max():.2f}]"
    )
    cv2.putText(
        canvas,
        label,
        (camera_width + 8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (30, 30, 30),
        1,
    )
    return canvas


def audit_dataset(
    dataset: NuScenesSequenceDataset,
    output_dir: Path,
    max_samples: int = 100,
    preview_samples: int = 8,
) -> dict[str, object]:
    sample_count = min(max_samples, len(dataset))
    collisions, trajectories, actions = [], [], []
    previews = []
    for index in range(sample_count):
        sample = dataset[index]
        collisions.append(float(sample["collision"].item()))
        trajectories.append(sample["future_trajectory"].numpy())
        actions.append(sample["actions"].numpy())
        if len(previews) < preview_samples:
            previews.append(render_sample(sample))

    trajectory_array = np.concatenate(trajectories)
    action_array = np.concatenate(actions)
    summary: dict[str, object] = {
        "dataset_windows": len(dataset),
        "scene_count": len(set(dataset.scene_tokens)),
        "audited_samples": sample_count,
        "collision_rate": float(np.mean(collisions)),
        "trajectory": {
            "lateral_min": float(trajectory_array[:, 0].min()),
            "lateral_max": float(trajectory_array[:, 0].max()),
            "longitudinal_min": float(trajectory_array[:, 1].min()),
            "longitudinal_max": float(trajectory_array[:, 1].max()),
        },
        "actions": {
            "steering_min": float(action_array[:, 0].min()),
            "steering_max": float(action_array[:, 0].max()),
            "throttle_mean": float(action_array[:, 1].mean()),
            "brake_mean": float(action_array[:, 2].mean()),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "nuscenes_audit.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    if previews:
        preview = np.concatenate(previews, axis=0)
        if not cv2.imwrite(str(output_dir / "nuscenes_audit.png"), preview):
            raise OSError("Unable to write nuScenes audit preview")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nuscenes_mini.yaml")
    parser.add_argument("--output-dir", default="artifacts/nuscenes_audit")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--preview-samples", type=int, default=8)
    args = parser.parse_args()
    config = load_config(args.config)
    summary = audit_dataset(
        build_nuscenes_dataset(config),
        Path(args.output_dir),
        args.max_samples,
        args.preview_samples,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
