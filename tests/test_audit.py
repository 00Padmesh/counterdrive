from pathlib import Path

import cv2

from counterdrive.audit import render_sample
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

