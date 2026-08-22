from pathlib import Path

import cv2
import pytest

from counterdrive.counterfactual import (
    SCENARIOS,
    action_sensitivity,
    render_trajectories,
)


def test_render_counterfactual_trajectories(tmp_path: Path) -> None:
    results = {
        name: {
            "trajectory": [[0.0, 0.5], [action[0], 1.0]],
            "collision_probability": 0.8 if index % 2 else 0.2,
            "ground_truth_collision": index % 2,
        }
        for index, (name, action) in enumerate(SCENARIOS.items())
    }
    output = tmp_path / "trajectories.png"
    render_trajectories(results, output)
    image = cv2.imread(str(output))
    assert image is not None
    assert image.shape == (600, 800, 3)
    sensitivity = action_sensitivity(results)
    assert sensitivity["mean_pairwise_final_displacement"] > 0
    assert sensitivity["collision_probability_range"] == pytest.approx(0.6)
    assert sensitivity["counterfactual_collision_accuracy"] == 1.0
    assert sensitivity["positive_negative_risk_gap"] == pytest.approx(0.6)
