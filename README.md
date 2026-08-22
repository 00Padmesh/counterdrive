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

For a fast CPU smoke test, reduce `train_samples`, `val_samples`, and `epochs`,
and set `model.pretrained: false` if offline. Evaluation reports ADE, FDE,
collision accuracy, precision, recall, F1, AUROC, and average precision. The
synthetic generator reuses each observed scene across five alternative future
actions and recomputes collision labels from each proposed trajectory.

### Colab and persistent checkpoints

Mount Google Drive before using the Phase 2 configuration:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Train with mixed precision while writing outputs directly to Drive:

```bash
counterdrive-train --config configs/colab_phase2.yaml
```

Every epoch updates `last.pt`; validation improvements update `best.pt`. Resume an
interrupted run with:

```bash
counterdrive-train \
  --config configs/colab_phase2.yaml \
  --resume /content/drive/MyDrive/CounterDrive/checkpoints/phase2_counterfactual/last.pt
```

The same directory receives `history.json` and `run_metadata.json`, including the
resolved device, versions, configuration, and resume source.

### Baseline and counterfactual experiment

Train the action-agnostic baseline with the same data and model capacity:

```bash
counterdrive-train --config configs/colab_baseline.yaml
```

Compare both models and visualize five action scenarios:

```bash
counterdrive-counterfactual \
  --config configs/colab_phase2.yaml \
  --checkpoint /content/drive/MyDrive/CounterDrive/checkpoints/phase2_counterfactual/best.pt \
  --baseline-checkpoint /content/drive/MyDrive/CounterDrive/checkpoints/baseline_counterfactual/best.pt \
  --output-dir /content/drive/MyDrive/CounterDrive/artifacts/counterfactual
```

The report includes ground-truth outcomes, counterfactual collision accuracy,
and the predicted risk gap between unsafe and safe actions. This writes
`counterfactual_report.json` and `counterfactual_trajectories.png`.

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

1. Download and unpack nuScenes mini so the configured root contains its maps,
   samples, sweeps, and `v1.0-mini` metadata.
2. Install the optional adapter: `python -m pip install -e ".[dev,nuscenes]"`.
3. Update `data.root` and `data.cache_dir` in `configs/nuscenes_mini.yaml`.

Expected Drive layout for the included Colab configuration:

```text
MyDrive/CounterDrive/data/nuscenes/
├── maps/
├── samples/
├── sweeps/
└── v1.0-mini/
```

Before training, audit the derived samples:

```bash
counterdrive-audit \
  --config configs/nuscenes_mini.yaml \
  --output-dir /content/drive/MyDrive/CounterDrive/artifacts/nuscenes_audit
```

This produces `nuscenes_audit.json` with label/control ranges and
`nuscenes_audit.png` with camera/trajectory panels. Review both before training.

The adapter reads `CAM_FRONT`, converts future global poses into the current ego
coordinate frame, and derives steering plus throttle/brake proxies from yaw rate and
acceleration. It creates approximate proximity-risk labels by testing the future ego
footprint against oriented, safety-expanded object footprints. These are not observed
collisions: nuScenes does not provide interventions or vehicle control commands, so
the proxy must not be treated as safety validation.

All valid temporal windows are indexed. Splits are made by scene rather than by
individual frame, preventing adjacent windows from leaking between training and
validation. Processed tensors are cached using a versioned key so later epochs do not
redecode images or recompute poses.

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
  metrics.py             ADE/FDE and extended collision metrics
  train.py               training/checkpoint CLI
  evaluate.py            evaluation CLI
  api.py                 FastAPI inference service
  counterfactual.py      action comparisons and trajectory visualization
  audit.py               nuScenes label audit and visual QA
tests/                   data, model, metrics, and API tests
```

## Next milestone

The highest-value next step after Phase 2 is a **real nuScenes-mini experiment and
label audit**, not a frontend. Inspect derived controls and risk labels, add explicit
time-to-collision/object-track targets, and compare the conditioned model against the
included action-agnostic baseline. After that, add latent future-frame consistency or
a lightweight decoder so rollouts can be visually audited.
