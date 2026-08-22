import numpy as np
import pytest
import torch

from counterdrive.data import (
    SyntheticDrivingDataset,
    derive_actions,
    global_to_ego,
    quaternion_yaw,
)


def test_synthetic_dataset_shapes_and_determinism() -> None:
    dataset = SyntheticDrivingDataset(4, sequence_length=3, future_steps=5, image_size=64, seed=7)
    first = dataset[0]
    repeated = dataset[0]
    assert first["frames"].shape == (3, 3, 64, 64)
    assert first["actions"].shape == (5, 3)
    assert first["future_trajectory"].shape == (5, 2)
    assert first["collision"].ndim == 0
    assert torch.equal(first["frames"], repeated["frames"])


def test_synthetic_dataset_balances_collision_labels() -> None:
    dataset = SyntheticDrivingDataset(
        100,
        sequence_length=2,
        future_steps=4,
        image_size=32,
        seed=11,
        collision_fraction=0.4,
    )
    positives = sum(int(dataset[index]["collision"].item()) for index in range(100))
    assert positives == 40


def test_nuscenes_coordinate_and_action_helpers() -> None:
    yaw = quaternion_yaw([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    assert yaw == pytest.approx(np.pi / 2)
    local = global_to_ego(
        np.asarray([[10.0, 11.0]], dtype=np.float32),
        np.asarray([10.0, 10.0], dtype=np.float32),
        yaw,
    )
    assert local[0] == pytest.approx([1.0, 0.0], abs=1e-6)

    positions = np.asarray([[0.0, 0.0], [0.0, 1.0], [0.1, 3.0], [0.3, 6.0]])
    yaws = np.asarray([0.0, 0.0, 0.1, 0.2])
    timestamps = np.asarray([0.0, 1.0, 2.0, 3.0])
    actions = derive_actions(positions, yaws, timestamps, history_index=1, future_steps=2)
    assert actions.shape == (2, 3)
    assert np.all(actions[:, 0] > 0)
    assert np.all(actions[:, 1] > 0)
    assert np.all(actions[:, 2] == 0)
