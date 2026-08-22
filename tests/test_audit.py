from pathlib import Path

import cv2

from counterdrive.audit import evenly_spaced_indices, render_sample
from counterdrive.data import SyntheticDrivingDataset


def test_render_real_data_audit_panel(tmp_path: Path) -> None:
    sample = SyntheticDrivingDataset(
        samples=5,
        sequence_length=3,
        future_steps=4,
        image_size=64,
        seed=3,
        collision_fraction=1.0,
    )[0]
    panel = render_sample(sample)
    output = tmp_path / "audit.png"
    assert cv2.imwrite(str(output), panel)
    assert panel.shape == (280, 720, 3)


def test_audit_indices_cover_entire_dataset() -> None:
    indices = evenly_spaced_indices(314, 8)
    assert indices[0] == 0
    assert indices[-1] == 313
    assert len(indices) == len(set(indices)) == 8
