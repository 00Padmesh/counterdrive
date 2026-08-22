from __future__ import annotations

import argparse
import base64
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from counterdrive.config import Config, load_config
from counterdrive.model import CounterDriveModel


class Action(BaseModel):
    steering: float = Field(ge=-1.0, le=1.0)
    throttle: float = Field(ge=0.0, le=1.0)
    brake: float = Field(ge=0.0, le=1.0)


class PredictionRequest(BaseModel):
    frames: list[str] = Field(description="Base64-encoded JPEG or PNG frames, oldest first")
    actions: list[Action]

    @field_validator("frames", "actions")
    @classmethod
    def not_empty(cls, value: list) -> list:
        if not value:
            raise ValueError("must not be empty")
        return value


class PredictionResponse(BaseModel):
    trajectory: list[list[float]]
    collision_probability: float


def decode_frame(encoded: str, image_size: int) -> torch.Tensor:
    try:
        payload = base64.b64decode(encoded, validate=True)
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail="Invalid base64 frame") from exc
    if image is None:
        raise HTTPException(status_code=422, detail="Frame is not a valid image")
    image = cv2.cvtColor(cv2.resize(image, (image_size, image_size)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0


def create_app(
    config_path: str = "configs/mvp.yaml",
    checkpoint_path: str | None = None,
) -> FastAPI:
    config: Config = load_config(config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        model = CounterDriveModel(config.model, config.data.future_steps).to(config.resolved_device)
        if checkpoint_path:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=config.resolved_device,
                weights_only=False,
            )
            model.load_state_dict(checkpoint["model"])
        model.eval()
        app.state.model = model
        yield

    app = FastAPI(title="CounterDrive API", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "device": config.resolved_device}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        if len(request.frames) != config.data.sequence_length:
            detail = f"Expected {config.data.sequence_length} frames"
            raise HTTPException(status_code=422, detail=detail)
        if len(request.actions) != config.data.future_steps:
            detail = f"Expected {config.data.future_steps} actions"
            raise HTTPException(status_code=422, detail=detail)
        decoded_frames = [
            decode_frame(frame, config.data.image_size) for frame in request.frames
        ]
        frames = torch.stack(decoded_frames).unsqueeze(0)
        action_values = [
            [action.steering, action.throttle, action.brake]
            for action in request.actions
        ]
        actions = torch.tensor(action_values).unsqueeze(0)
        with torch.inference_mode():
            outputs = app.state.model(
                frames.to(config.resolved_device),
                actions.to(config.resolved_device),
            )
        return PredictionResponse(
            trajectory=outputs["trajectory"][0].cpu().tolist(),
            collision_probability=torch.sigmoid(outputs["collision_logits"])[0].item(),
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.checkpoint and not Path(args.checkpoint).exists():
        raise FileNotFoundError(args.checkpoint)
    uvicorn.run(create_app(args.config, args.checkpoint), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
