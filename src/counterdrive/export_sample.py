from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from counterdrive.config import Config, load_config
from counterdrive.data import NuScenesSequenceDataset, SyntheticDrivingDataset


def build_dataset(config: Config):
    data = config.data
    if data.backend == "nuscenes":
        return NuScenesSequenceDataset(
            data.root,
            data.version,
            data.sequence_length,
            data.future_steps,
            data.image_size,
            data.cache_dir,
            data.use_cache,
        )
    if data.backend == "synthetic":
        return SyntheticDrivingDataset(
            max(data.train_samples, 1),
            data.sequence_length,
            data.future_steps,
            data.image_size,
            config.seed,
            data.collision_fraction,
        )
    raise ValueError(f"Unsupported data backend: {data.backend}")


def select_index(length: int, index: int, random_sample: bool, seed: int) -> int:
    if length < 1:
        raise ValueError("Dataset contains no exportable windows")
    if random_sample:
        return int(np.random.default_rng(seed).integers(0, length))
    if not 0 <= index < length:
        raise IndexError(f"Sample index {index} is outside 0..{length - 1}")
    return index


def export_sample(config: Config, output_dir: Path, index: int) -> Path:
    dataset = build_dataset(config)
    sample = dataset[index]
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_names = []
    for frame_index, frame in enumerate(sample["frames"], start=1):
        rgb = frame.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        frame_name = f"frame_{frame_index:02d}.jpg"
        if not cv2.imwrite(str(output_dir / frame_name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Unable to write {frame_name}")
        frame_names.append(frame_name)
    past = sample.get("past_trajectory")
    if past is None:
        past = torch.zeros((config.data.sequence_length, 2), dtype=torch.float32)
    manifest = {
        "name": f"{config.data.backend} sample {index}",
        "backend": config.data.backend,
        "dataset_index": index,
        "frames": frame_names,
        "past_trajectory": past.tolist(),
        "sequence_length": config.data.sequence_length,
        "future_steps": config.data.future_steps,
    }
    manifest_path = output_dir / "scene.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nuscenes_mini.yaml")
    parser.add_argument("--output-dir", default="frontend/public/examples/nuscenes_scene")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--random", action="store_true", dest="random_sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = build_dataset(config)
    index = select_index(len(dataset), args.index, args.random_sample, args.seed)
    manifest = export_sample(config, Path(args.output_dir), index)
    print(f"Exported sample {index} to {manifest}")


if __name__ == "__main__":
    main()
