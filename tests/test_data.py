import numpy as np
import pytest
import torch

from counterdrive.data import (
    SyntheticDrivingDataset,
    derive_actions,
    global_to_ego,
    oriented_proximity_risk,
    quaternion_yaw,
    scene_split_indices,
    valid_window_starts,
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
        collision_fraction=1.0,
    )
    positives = sum(int(dataset[index]["collision"].item()) for index in range(100))
    assert 20 <= positives <= 80
    for scene_start in range(0, 100, 5):
        scene_samples = [dataset[scene_start + offset] for offset in range(5)]
        reference_frames = scene_samples[0]["frames"]
        assert all(torch.equal(sample["frames"], reference_frames) for sample in scene_samples)
        unique_actions = {
            tuple(sample["actions"][0].tolist()) for sample in scene_samples
        }
        assert len(unique_actions) == 5
        labels = {int(sample["collision"].item()) for sample in scene_samples}
        assert labels == {0, 1}


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


def test_nuscenes_windows_and_scene_split_do_not_leak() -> None:
    assert valid_window_starts(["a", "b", "c", "d"], 3) == ["a", "b"]
    assert valid_window_starts(["a", "b"], 3) == []
    scene_tokens = ["scene-a"] * 4 + ["scene-b"] * 3 + ["scene-c"] * 2
    train_indices, validation_indices = scene_split_indices(scene_tokens, 0.34, seed=7)
    train_scenes = {scene_tokens[index] for index in train_indices}
    validation_scenes = {scene_tokens[index] for index in validation_indices}
    assert train_scenes.isdisjoint(validation_scenes)
    assert train_indices and validation_indices


def test_oriented_proximity_avoids_large_radius_false_positive() -> None:
    ego = np.asarray([0.0, 0.0])
    rotation = [1.0, 0.0, 0.0, 0.0]
    car_size = [1.8, 4.2, 1.6]
    assert oriented_proximity_risk(
        ego,
        np.asarray([5.0, 0.0]),
        rotation,
        car_size,
    )
    assert not oriented_proximity_risk(
        ego,
        np.asarray([0.0, 5.0]),
        rotation,
        car_size,
    )
