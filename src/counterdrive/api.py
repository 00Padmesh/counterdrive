from __future__ import annotations

import argparse
import base64
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from counterdrive.config import Config, load_config
from counterdrive.data import ACTION_SCENARIOS
from counterdrive.model import CounterDriveModel


class Action(BaseModel):
    steering: float = Field(ge=-1.0, le=1.0)
    throttle: float = Field(ge=0.0, le=1.0)
    brake: float = Field(ge=0.0, le=1.0)


class SceneRequest(BaseModel):
    frames: list[str] = Field(description="Base64-encoded JPEG or PNG frames, oldest first")
    past_trajectory: list[list[float]] | None = Field(
        default=None,
        description="Optional observed ego positions as [lateral, longitudinal]",
    )

    @field_validator("frames")
    @classmethod
    def frames_not_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("must not be empty")
        return value


class PredictionRequest(SceneRequest):
    actions: list[Action]

    @field_validator("actions")
    @classmethod
    def actions_not_empty(cls, value: list[Action]) -> list[Action]:
        if not value:
            raise ValueError("must not be empty")
        return value


class PredictionResponse(BaseModel):
    trajectory: list[list[float]]
    collision_probability: float
    risk_label: Literal["elevated", "low"]
    risk_threshold: float


class CounterfactualResponse(BaseModel):
    scenarios: dict[str, PredictionResponse]


def decode_frame(encoded: str, image_size: int) -> torch.Tensor:
    if encoded.startswith("data:"):
        encoded = encoded.partition(",")[2]
    try:
        payload = base64.b64decode(encoded, validate=True)
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail="Invalid base64 frame") from exc
    if image is None:
        raise HTTPException(status_code=422, detail="Frame is not a valid image")
    image = cv2.cvtColor(cv2.resize(image, (image_size, image_size)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0


def _load_model(config: Config, checkpoint_path: str | None) -> CounterDriveModel:
    model = CounterDriveModel(config.model, config.data.future_steps).to(config.resolved_device)
    if checkpoint_path:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=config.resolved_device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
    return model.eval()


def _validate_scene(request: SceneRequest, config: Config) -> torch.Tensor | None:
    if len(request.frames) != config.data.sequence_length:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {config.data.sequence_length} frames",
        )
    if request.past_trajectory is None:
        if config.model.use_kinematic_residual:
            raise HTTPException(status_code=422, detail="This model requires past_trajectory")
        return None
    if len(request.past_trajectory) != config.data.sequence_length:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {config.data.sequence_length} past positions",
        )
    if any(len(position) != 2 for position in request.past_trajectory):
        raise HTTPException(
            status_code=422,
            detail="Each past position must contain two coordinates",
        )
    return torch.tensor(request.past_trajectory, dtype=torch.float32).unsqueeze(0)


def create_app(
    config_path: str = "configs/mvp.yaml",
    checkpoint_path: str | None = None,
    *,
    trajectory_checkpoint: str | None = None,
    risk_checkpoint: str | None = None,
    risk_threshold: float = 0.5,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    config = load_config(config_path)
    trajectory_checkpoint = trajectory_checkpoint or checkpoint_path
    risk_checkpoint = risk_checkpoint or trajectory_checkpoint

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        trajectory_model = _load_model(config, trajectory_checkpoint)
        risk_model = (
            trajectory_model
            if risk_checkpoint == trajectory_checkpoint
            else _load_model(config, risk_checkpoint)
        )
        app.state.trajectory_model = trajectory_model
        app.state.risk_model = risk_model
        yield

    app = FastAPI(
        title="CounterDrive API",
        description="Action-conditioned driving trajectory and proximity-risk inference.",
        version="0.2.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins
        or ["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "device": config.resolved_device, "version": "0.2.0"}

    @app.get("/metadata")
    def metadata() -> dict[str, object]:
        return {
            "name": "CounterDrive",
            "version": "0.2.0",
            "sequence_length": config.data.sequence_length,
            "future_steps": config.data.future_steps,
            "image_size": config.data.image_size,
            "action_scenarios": list(ACTION_SCENARIOS),
            "risk_threshold": risk_threshold,
            "trajectory_checkpoint_loaded": trajectory_checkpoint is not None,
            "risk_checkpoint_loaded": risk_checkpoint is not None,
            "uses_kinematic_residual": config.model.use_kinematic_residual,
            "disclaimer": (
                "Research prototype. Proximity-risk predictions are not safety validation "
                "or driving advice."
            ),
        }

    def run_prediction(request: SceneRequest, actions: list[Action]) -> PredictionResponse:
        if len(actions) != config.data.future_steps:
            raise HTTPException(
                status_code=422,
                detail=f"Expected {config.data.future_steps} actions",
            )
        past_trajectory = _validate_scene(request, config)
        frames = torch.stack(
            [decode_frame(frame, config.data.image_size) for frame in request.frames]
        ).unsqueeze(0)
        action_values = [
            [action.steering, action.throttle, action.brake] for action in actions
        ]
        action_tensor = torch.tensor(action_values, dtype=torch.float32).unsqueeze(0)
        device = config.resolved_device
        device_past = past_trajectory.to(device) if past_trajectory is not None else None
        with torch.inference_mode():
            trajectory_outputs = app.state.trajectory_model(
                frames.to(device), action_tensor.to(device), device_past
            )
            risk_outputs = (
                trajectory_outputs
                if app.state.risk_model is app.state.trajectory_model
                else app.state.risk_model(frames.to(device), action_tensor.to(device), device_past)
            )
        probability = torch.sigmoid(risk_outputs["collision_logits"])[0].item()
        return PredictionResponse(
            trajectory=trajectory_outputs["trajectory"][0].cpu().tolist(),
            collision_probability=probability,
            risk_label="elevated" if probability >= risk_threshold else "low",
            risk_threshold=risk_threshold,
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        return run_prediction(request, request.actions)

    @app.post("/counterfactual", response_model=CounterfactualResponse)
    def counterfactual(request: SceneRequest) -> CounterfactualResponse:
        scenarios = {}
        for name, values in ACTION_SCENARIOS.items():
            action = Action(steering=values[0], throttle=values[1], brake=values[2])
            scenarios[name] = run_prediction(request, [action] * config.data.future_steps)
        return CounterfactualResponse(scenarios=scenarios)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--checkpoint", default=None, help="Legacy shared checkpoint")
    parser.add_argument("--trajectory-checkpoint", default=None)
    parser.add_argument("--risk-checkpoint", default=None)
    parser.add_argument("--risk-threshold", type=float, default=0.5)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    for path in (args.checkpoint, args.trajectory_checkpoint, args.risk_checkpoint):
        if path and not Path(path).exists():
            raise FileNotFoundError(path)
    if not 0.0 <= args.risk_threshold <= 1.0:
        raise ValueError("--risk-threshold must be between 0 and 1")
    app = create_app(
        args.config,
        args.checkpoint,
        trajectory_checkpoint=args.trajectory_checkpoint,
        risk_checkpoint=args.risk_checkpoint,
        risk_threshold=args.risk_threshold,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
