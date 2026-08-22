from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from counterdrive.config import Config

ACTION_SCENARIOS: dict[str, tuple[float, float, float]] = {
    "hard_brake": (0.0, 0.0, 1.0),
    "maintain": (0.0, 0.45, 0.0),
    "accelerate": (0.0, 1.0, 0.0),
    "turn_left": (-0.65, 0.35, 0.0),
    "turn_right": (0.65, 0.35, 0.0),
}


@dataclass(frozen=True)
class DrivingSample:
    frames: torch.Tensor
    actions: torch.Tensor
    future_trajectory: torch.Tensor
    collision: torch.Tensor
    obstacle_position: torch.Tensor | None = None


def quaternion_yaw(rotation: list[float]) -> float:
    """Return planar yaw from a nuScenes [w, x, y, z] quaternion."""
    w, x, y, z = rotation
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def global_to_ego(
    global_positions: np.ndarray,
    ego_origin: np.ndarray,
    ego_yaw: float,
) -> np.ndarray:
    offsets = global_positions - ego_origin
    cosine, sine = np.cos(ego_yaw), np.sin(ego_yaw)
    global_from_ego = np.asarray([[cosine, -sine], [sine, cosine]])
    return offsets @ global_from_ego


def derive_actions(
    positions: np.ndarray,
    yaws: np.ndarray,
    timestamps: np.ndarray,
    history_index: int,
    future_steps: int,
) -> np.ndarray:
    deltas = np.diff(positions, axis=0)
    elapsed = np.diff(timestamps).clip(min=1e-3)
    speeds = np.linalg.norm(deltas, axis=1) / elapsed
    accelerations = np.diff(speeds, prepend=speeds[0]) / elapsed
    yaw_changes = np.diff(yaws)
    yaw_deltas = np.arctan2(np.sin(yaw_changes), np.cos(yaw_changes))
    indices = np.arange(history_index, history_index + future_steps)
    actions = np.zeros((future_steps, 3), dtype=np.float32)
    actions[:, 0] = np.clip(yaw_deltas[indices] / 0.5, -1.0, 1.0)
    actions[:, 1] = np.clip(accelerations[indices] / 3.0, 0.0, 1.0)
    actions[:, 2] = np.clip(-accelerations[indices] / 5.0, 0.0, 1.0)
    return actions


def trajectory_for_action(
    action: tuple[float, float, float],
    future_steps: int,
) -> torch.Tensor:
    steering, throttle, brake = action
    speed = max(0.15, throttle - 0.65 * brake + 0.25)
    times = torch.arange(1, future_steps + 1, dtype=torch.float32)
    longitudinal = times * speed
    lateral = 0.18 * steering * times.square()
    return torch.stack((lateral, longitudinal), dim=-1)


class SyntheticDrivingDataset(Dataset[dict[str, torch.Tensor]]):
    """Deterministic toy scenes whose future depends on steering, throttle, and brake."""

    def __init__(
        self,
        samples: int,
        sequence_length: int,
        future_steps: int,
        image_size: int,
        seed: int,
        collision_fraction: float = 0.35,
    ):
        self.samples = samples
        self.sequence_length = sequence_length
        self.future_steps = future_steps
        self.image_size = image_size
        self.seed = seed
        self.collision_fraction = collision_fraction

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        scenario_values = list(ACTION_SCENARIOS.values())
        scenario_count = len(scenario_values)
        scene_index, action_index = divmod(index, scenario_count)
        rng = np.random.default_rng(self.seed + scene_index)
        history_speed = float(rng.uniform(0.25, 0.8))
        has_reachable_obstacle = rng.random() < self.collision_fraction
        if has_reachable_obstacle:
            target_action_index = int(rng.integers(1, scenario_count))
            target_trajectory = trajectory_for_action(
                scenario_values[target_action_index],
                self.future_steps,
            )
            earliest_step = max(1, self.future_steps // 2)
            collision_step = int(rng.integers(earliest_step, self.future_steps))
            obstacle_x, obstacle_y = target_trajectory[collision_step].tolist()
        else:
            obstacle_x = float(rng.choice([-1.0, 1.0]) * rng.uniform(8.0, 10.0))
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
            scale = 1.0 / max(obstacle_y - step * 0.15 * history_speed, 0.5)
            px = int(center + obstacle_x * self.image_size * 0.22 * scale)
            py = int(horizon + self.image_size * 0.6 * scale)
            obstacle_center = (
                np.clip(px, 2, self.image_size - 3),
                np.clip(py, 2, self.image_size - 3),
            )
            cv2.circle(image, obstacle_center, 3, (0, 0, 255), -1)
            frames.append(torch.from_numpy(image).permute(2, 0, 1).float() / 255.0)

        action = scenario_values[action_index]
        trajectory = trajectory_for_action(action, self.future_steps)
        squared_distance = (
            (trajectory[:, 0] - obstacle_x).square()
            + (trajectory[:, 1] - obstacle_y).square()
        )
        min_distance = torch.sqrt(squared_distance).min()
        collision = (min_distance < 0.9).float()
        action_tensor = torch.tensor(action, dtype=torch.float32)
        actions = action_tensor.repeat(self.future_steps, 1)
        return {
            "frames": torch.stack(frames),
            "actions": actions,
            "future_trajectory": trajectory,
            "collision": collision,
            "obstacle_position": torch.tensor(
                [obstacle_x, obstacle_y], dtype=torch.float32
            ),
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
        frames, positions, yaws, timestamps, future_records = [], [], [], [], []
        current = sample
        total_steps = self.sequence_length + self.future_steps
        for step in range(total_steps):
            cam = self.nusc.get("sample_data", current["data"]["CAM_FRONT"])
            pose = self.nusc.get("ego_pose", cam["ego_pose_token"])
            positions.append(pose["translation"][:2])
            yaws.append(quaternion_yaw(pose["rotation"]))
            timestamps.append(current["timestamp"] / 1_000_000.0)
            if step >= self.sequence_length:
                future_records.append(current)
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
        positions_array = np.asarray(positions, dtype=np.float32)
        history_index = self.sequence_length - 1
        origin = positions_array[history_index]
        future = global_to_ego(
            positions_array[self.sequence_length :],
            origin,
            yaws[history_index],
        ).astype(np.float32)
        actions = derive_actions(
            positions_array,
            np.asarray(yaws),
            np.asarray(timestamps),
            history_index,
            self.future_steps,
        )
        collision = self._has_collision(future_records, positions_array[self.sequence_length :])
        return {
            "frames": torch.stack(frames),
            "actions": torch.from_numpy(actions),
            "future_trajectory": torch.from_numpy(future),
            "collision": torch.tensor(float(collision)),
        }

    def _has_collision(
        self,
        future_records: list[dict],
        ego_positions: np.ndarray,
    ) -> bool:
        for record, ego_position in zip(future_records, ego_positions, strict=True):
            for annotation_token in record["anns"]:
                annotation = self.nusc.get("sample_annotation", annotation_token)
                object_position = np.asarray(annotation["translation"][:2])
                width, length = annotation["size"][:2]
                object_radius = 0.5 * np.hypot(width, length)
                if np.linalg.norm(object_position - ego_position) < object_radius + 1.2:
                    return True
        return False


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
            data.collision_fraction,
        )
        val_set = SyntheticDrivingDataset(
            data.val_samples,
            data.sequence_length,
            data.future_steps,
            data.image_size,
            config.seed + 100_000,
            data.collision_fraction,
        )
    else:
        raise ValueError(f"Unsupported data backend: {data.backend}")
    kwargs = {"batch_size": config.training.batch_size, "num_workers": data.num_workers}
    train_loader = DataLoader(train_set, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **kwargs)
    return train_loader, val_loader
