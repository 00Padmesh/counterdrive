from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


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
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    labels = target.detach().cpu().numpy()
    has_both_classes = len(set(labels.tolist())) == 2
    auroc = float(roc_auc_score(labels, probabilities)) if has_both_classes else float("nan")
    average_precision = (
        float(average_precision_score(labels, probabilities))
        if has_both_classes
        else float("nan")
    )
    best_f1 = float("nan")
    best_threshold = float("nan")
    if has_both_classes:
        curve_precision, curve_recall, thresholds = precision_recall_curve(
            labels,
            probabilities,
        )
        curve_f1 = (
            2
            * curve_precision[:-1]
            * curve_recall[:-1]
            / np.clip(curve_precision[:-1] + curve_recall[:-1], 1e-8, None)
        )
        best_index = int(np.argmax(curve_f1))
        best_f1 = float(curve_f1[best_index])
        best_threshold = float(thresholds[best_index])
    return {
        "collision_accuracy": accuracy.item(),
        "collision_precision": precision.item(),
        "collision_recall": recall.item(),
        "collision_f1": f1.item(),
        "collision_auroc": auroc,
        "collision_average_precision": average_precision,
        "collision_best_f1": best_f1,
        "collision_best_threshold": best_threshold,
    }
