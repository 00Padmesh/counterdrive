from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from counterdrive.config import Config


@dataclass(frozen=True)
class DrivingSample:
    frames: torch.Tensor
    actions: torch.Tensor
    future_trajectory: torch.Tensor
    collision: torch.Tensor


class SyntheticDrivingDataset(Dataset[dict[str, torch.Tensor]]):
    """Deterministic toy scenes whose future depends on steering, throttle, and brake."""

    def __init__(
        self,
        samples: int,
        sequence_length: int,
        future_steps: int,
        image_size: int,
        seed: int,
    ):
        self.samples = samples
        self.sequence_length = sequence_length
        self.future_steps = future_steps
        self.image_size = image_size
        self.seed = seed

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + index)
        steering = float(rng.uniform(-0.7, 0.7))
        throttle = float(rng.uniform(0.0, 1.0))
        brake = float(rng.uniform(0.0, 0.7))
        speed = max(0.15, throttle - 0.65 * brake + 0.25)
        obstacle_x = float(rng.uniform(-1.0, 1.0))
        obstacle_y = float(rng.uniform(2.5, 8.0))

        frames = []
        for step in range(self.sequence_length):
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            image[:] = (35, 35, 35)
            horizon = self.image_size // 3
            cv2.rectangle(image, (0, horizon), (self.image_size, self.image_size), (55, 55, 55), -1)
            center = self.image_size // 2
            cv2.line(
                image,
                (center - 15, self.image_size),
                (center - 5, horizon),
                (220, 220, 220),
                2,
            )
            cv2.line(
                image,
                (center + 15, self.image_size),
                (center + 5, horizon),
                (220, 220, 220),
                2,
            )
            scale = 1.0 / max(obstacle_y - step * 0.15 * speed, 0.5)
            px = int(center + obstacle_x * self.image_size * 0.22 * scale)
            py = int(horizon + self.image_size * 0.6 * scale)
            obstacle_center = (
                np.clip(px, 2, self.image_size - 3),
                np.clip(py, 2, self.image_size - 3),
            )
            cv2.circle(image, obstacle_center, 3, (0, 0, 255), -1)
            frames.append(torch.from_numpy(image).permute(2, 0, 1).float() / 255.0)

        times = torch.arange(1, self.future_steps + 1, dtype=torch.float32)
        longitudinal = times * speed
        lateral = 0.18 * steering * times.square()
        trajectory = torch.stack((lateral, longitudinal), dim=-1)
        squared_distance = (
            (lateral - obstacle_x).square()
            + (longitudinal - obstacle_y).square()
        )
        min_distance = torch.sqrt(squared_distance).min()
        collision = (min_distance < 0.9).float()
        action = torch.tensor([steering, throttle, brake], dtype=torch.float32)
        actions = action.repeat(self.future_steps, 1)
        return {
            "frames": torch.stack(frames),
            "actions": actions,
            "future_trajectory": trajectory,
            "collision": collision,
        }


class NuScenesSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Minimal nuScenes-mini front-camera adapter with ego-motion trajectory labels."""

    def __init__(
        self,
        root: str,
        version: str,
        sequence_length: int,
        future_steps: int,
        image_size: int,
    ):
        try:
            from nuscenes.nuscenes import NuScenes
        except ImportError as exc:
            raise ImportError("Install CounterDrive with the 'nuscenes' extra") from exc
        self.nusc = NuScenes(version=version, dataroot=root, verbose=False)
        self.sequence_length = sequence_length
        self.future_steps = future_steps
        self.image_size = image_size
        self.tokens = [scene["first_sample_token"] for scene in self.nusc.scene]

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.nusc.get("sample", self.tokens[index])
        frames, positions = [], []
        current = sample
        for _ in range(self.sequence_length + self.future_steps):
            cam = self.nusc.get("sample_data", current["data"]["CAM_FRONT"])
            pose = self.nusc.get("ego_pose", cam["ego_pose_token"])
            positions.append(pose["translation"][:2])
            if len(frames) < self.sequence_length:
                image = cv2.imread(str(Path(self.nusc.dataroot) / cam["filename"]))
                image = cv2.resize(image, (self.image_size, self.image_size))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                frames.append(torch.from_numpy(image).permute(2, 0, 1).float() / 255.0)
            if not current["next"]:
                break
            current = self.nusc.get("sample", current["next"])
        if len(positions) < self.sequence_length + self.future_steps:
            return self[(index + 1) % len(self)]
        origin = np.asarray(positions[self.sequence_length - 1], dtype=np.float32)
        future = np.asarray(positions[self.sequence_length :], dtype=np.float32) - origin
        actions = np.zeros((self.future_steps, 3), dtype=np.float32)
        position_deltas = np.diff(np.asarray(positions), axis=0)
        speeds = np.linalg.norm(position_deltas, axis=1)
        actions[:, 1] = speeds[-self.future_steps :]
        return {
            "frames": torch.stack(frames),
            "actions": torch.from_numpy(actions),
            "future_trajectory": torch.from_numpy(future),
            "collision": torch.tensor(0.0),
        }


def build_dataloaders(config: Config) -> tuple[DataLoader, DataLoader]:
    data = config.data
    if data.backend == "nuscenes":
        dataset: Dataset = NuScenesSequenceDataset(
            data.root,
            data.version,
            data.sequence_length,
            data.future_steps,
            data.image_size,
        )
        train_size = max(1, int(0.8 * len(dataset)))
        val_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(config.seed)
        train_set, val_set = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=generator,
        )
    elif data.backend == "synthetic":
        train_set = SyntheticDrivingDataset(
            data.train_samples,
            data.sequence_length,
            data.future_steps,
            data.image_size,
            config.seed,
        )
        val_set = SyntheticDrivingDataset(
            data.val_samples,
            data.sequence_length,
            data.future_steps,
            data.image_size,
            config.seed + 100_000,
        )
    else:
        raise ValueError(f"Unsupported data backend: {data.backend}")
    kwargs = {"batch_size": config.training.batch_size, "num_workers": data.num_workers}
    train_loader = DataLoader(train_set, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **kwargs)
    return train_loader, val_loader
