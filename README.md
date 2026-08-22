# CounterDrive

CounterDrive is a student-scale, action-conditioned driving world model. Given recent front-camera frames and a proposed sequence of controls (`steering`, `throttle`, `brake`), it rolls forward a learned latent state and predicts:

- future ego positions `(x, y)`
- the probability of a collision during the horizon
- future latent states for later world-model objectives

The MVP is deliberately small enough for a laptop or a modest GPU. It uses a frozen pretrained ResNet-18 initially, temporal and dynamics Transformers, and compact prediction heads. A deterministic synthetic driving generator makes the entire pipeline runnable without nuScenes credentials.

## Architecture

```text
past RGB frames -> ResNet-18 -> temporal Transformer -> current latent state
                                                          +
future controls -> action MLP -> step-conditioned embeddings + -> dynamics Transformer
                                                               |-> trajectory head
                                                               |-> collision head
```

This is a predictive representation model, not a safety-ready driving policy. Its collision labels and synthetic images are useful for pipeline development, not real-world validation.

## Setup

Python 3.10+ is required. From the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
```

On macOS/Linux, activation is `source .venv/bin/activate`. PyTorch users who need a specific CUDA build should install it using the official PyTorch selector before installing CounterDrive.

## Train and evaluate

The default configuration uses synthetic data and downloads pretrained ResNet-18 weights on first use:

```bash
counterdrive-train --config configs/mvp.yaml
counterdrive-evaluate --config configs/mvp.yaml --checkpoint checkpoints/best.pt
```

For a fast CPU smoke test, reduce `train_samples`, `val_samples`, and `epochs`, and set `model.pretrained: false` if offline. Metrics are average displacement error (ADE), final displacement error (FDE), and collision accuracy/precision/recall.

## Run the API

```bash
counterdrive-api --config configs/mvp.yaml --checkpoint checkpoints/best.pt
```

Open `http://127.0.0.1:8000/docs`. `POST /predict` accepts exactly `sequence_length` base64-encoded JPEG/PNG frames and exactly `future_steps` actions. Omitting `--checkpoint` starts a randomly initialized model, which is useful only for integration testing.

Example request shape:

```json
{
  "frames": ["<base64 image>", "<base64 image>", "<base64 image>", "<base64 image>"],
  "actions": [
    {"steering": 0.0, "throttle": 0.3, "brake": 0.0},
    {"steering": 0.0, "throttle": 0.3, "brake": 0.0},
    {"steering": 0.1, "throttle": 0.2, "brake": 0.0},
    {"steering": 0.1, "throttle": 0.0, "brake": 0.2},
    {"steering": 0.0, "throttle": 0.0, "brake": 0.5},
    {"steering": 0.0, "throttle": 0.0, "brake": 0.5}
  ]
}
```

## nuScenes-mini

1. Download and unpack nuScenes mini so `data/nuscenes` contains its maps, samples, sweeps, and `v1.0-mini` metadata.
2. Install the optional adapter: `python -m pip install -e ".[dev,nuscenes]"`.
3. Change `data.backend` in the YAML to `nuscenes` and set `data.root`.

The initial adapter reads `CAM_FRONT`, derives future ego displacement from poses, and uses speed as the throttle proxy. nuScenes does not directly provide intervention/control or collision labels, so collision defaults to zero in this first adapter. That limitation should be addressed before treating nuScenes results as meaningful collision-risk evaluation.

## Quality checks

```bash
ruff check .
pytest
```

## Repository layout

```text
configs/                 experiment settings
src/counterdrive/
  data.py                synthetic and nuScenes-compatible sequences
  model.py               encoder, temporal state, dynamics, heads
  engine.py              losses and evaluation loop
  metrics.py             ADE/FDE and collision metrics
  train.py               training/checkpoint CLI
  evaluate.py            evaluation CLI
  api.py                 FastAPI inference service
tests/                   data, model, metrics, and API tests
```

## Next milestone

The highest-value next step is **data realism and counterfactual supervision**, not a frontend. Add proper ego-frame coordinate transforms, CAN-bus/control signals, object tracks, and time-to-collision labels; then compare action-conditioned rollouts against an action-agnostic baseline. After that, add latent future-frame consistency or a lightweight decoder so predictions can be visually audited.

