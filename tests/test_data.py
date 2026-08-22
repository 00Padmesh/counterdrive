import torch

from counterdrive.data import SyntheticDrivingDataset


def test_synthetic_dataset_shapes_and_determinism() -> None:
    dataset = SyntheticDrivingDataset(4, sequence_length=3, future_steps=5, image_size=64, seed=7)
    first = dataset[0]
    repeated = dataset[0]
    assert first["frames"].shape == (3, 3, 64, 64)
    assert first["actions"].shape == (5, 3)
    assert first["future_trajectory"].shape == (5, 2)
    assert first["collision"].ndim == 0
    assert torch.equal(first["frames"], repeated["frames"])

