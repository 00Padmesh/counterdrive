from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from counterdrive.config import ModelConfig


class FrameEncoder(nn.Module):
    def __init__(self, latent_dim: int, pretrained: bool, freeze: bool, unfreeze_layer4: bool):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Linear(feature_dim, latent_dim)
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.frozen = freeze
        if freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            if unfreeze_layer4:
                for parameter in self.backbone.layer4.parameters():
                    parameter.requires_grad = True

    def train(self, mode: bool = True) -> FrameEncoder:
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
            if any(parameter.requires_grad for parameter in self.backbone.layer4.parameters()):
                self.backbone.layer4.train(mode)
        return self

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = frames.shape
        images = frames.reshape(batch * time, channels, height, width)
        images = (images - self.image_mean) / self.image_std
        features = self.backbone(images)
        return self.projection(features).reshape(batch, time, -1)


class CounterDriveModel(nn.Module):
    def __init__(self, config: ModelConfig, future_steps: int):
        super().__init__()
        dim = config.latent_dim
        self.future_steps = future_steps
        self.action_conditioned = config.action_conditioned
        self.frame_encoder = FrameEncoder(
            dim,
            config.pretrained,
            config.freeze_vision,
            config.unfreeze_layer4,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=config.transformer_heads,
            dim_feedforward=dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, config.transformer_layers)
        self.action_encoder = nn.Sequential(
            nn.Linear(config.action_dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        dynamics_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=config.transformer_heads,
            dim_feedforward=dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.dynamics = nn.TransformerEncoder(dynamics_layer, config.transformer_layers)
        self.step_embedding = nn.Parameter(torch.randn(1, future_steps, dim) * 0.02)
        self.trajectory_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 2),
        )
        self.collision_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, frames: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        history = self.temporal_encoder(self.frame_encoder(frames))
        current_state = history[:, -1:, :]
        if not self.action_conditioned:
            actions = torch.zeros_like(actions)
        dynamics_input = current_state + self.action_encoder(actions) + self.step_embedding
        future_latents = self.dynamics(dynamics_input)
        trajectory = self.trajectory_head(future_latents)
        collision_logits = self.collision_head(future_latents.mean(dim=1)).squeeze(-1)
        return {
            "trajectory": trajectory,
            "collision_logits": collision_logits,
            "future_latents": future_latents,
        }
