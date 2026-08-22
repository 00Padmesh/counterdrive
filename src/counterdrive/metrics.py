from __future__ import annotations

import torch


def trajectory_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    distances = torch.linalg.vector_norm(prediction - target, dim=-1)
    return {"ade": distances.mean().item(), "fde": distances[:, -1].mean().item()}


def collision_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction = (torch.sigmoid(logits) >= 0.5).float()
    target = target.float()
    accuracy = (prediction == target).float().mean()
    true_positive = ((prediction == 1) & (target == 1)).sum().float()
    false_positive = ((prediction == 1) & (target == 0)).sum().float()
    false_negative = ((prediction == 0) & (target == 1)).sum().float()
    precision = true_positive / (true_positive + false_positive).clamp_min(1)
    recall = true_positive / (true_positive + false_negative).clamp_min(1)
    return {
        "collision_accuracy": accuracy.item(),
        "collision_precision": precision.item(),
        "collision_recall": recall.item(),
    }
