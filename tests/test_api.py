import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient

from counterdrive.api import create_app


def test_predict_endpoint(tmp_path) -> None:
    config = tmp_path / "test.yaml"
    config.write_text("""
device: cpu
data:
  sequence_length: 2
  future_steps: 3
  image_size: 64
model:
  pretrained: false
  freeze_vision: true
  latent_dim: 32
  transformer_layers: 1
  transformer_heads: 4
  dropout: 0.0
""", encoding="utf-8")
    ok, encoded = cv2.imencode(".jpg", np.zeros((64, 64, 3), dtype=np.uint8))
    assert ok
    frame = base64.b64encode(encoded.tobytes()).decode()
    app = create_app(str(config))
    with TestClient(app) as client:
        response = client.post("/predict", json={
            "frames": [frame, frame],
            "actions": [{"steering": 0.0, "throttle": 0.4, "brake": 0.0}] * 3,
        })
    assert response.status_code == 200
    assert len(response.json()["trajectory"]) == 3
    assert 0.0 <= response.json()["collision_probability"] <= 1.0
    assert response.json()["risk_label"] in {"low", "elevated"}


def test_metadata_and_counterfactual_endpoints(tmp_path) -> None:
    config = tmp_path / "test.yaml"
    config.write_text("""
device: cpu
data:
  sequence_length: 2
  future_steps: 3
  image_size: 64
model:
  pretrained: false
  freeze_vision: true
  latent_dim: 32
  transformer_layers: 1
  transformer_heads: 4
  dropout: 0.0
""", encoding="utf-8")
    ok, encoded = cv2.imencode(".jpg", np.zeros((64, 64, 3), dtype=np.uint8))
    assert ok
    frame = base64.b64encode(encoded.tobytes()).decode()
    app = create_app(str(config), risk_threshold=0.25)
    with TestClient(app) as client:
        metadata = client.get("/metadata")
        response = client.post("/counterfactual", json={"frames": [frame, frame]})
    assert metadata.status_code == 200
    assert metadata.json()["risk_threshold"] == 0.25
    assert response.status_code == 200
    assert set(response.json()["scenarios"]) == {
        "hard_brake",
        "maintain",
        "accelerate",
        "turn_left",
        "turn_right",
    }
