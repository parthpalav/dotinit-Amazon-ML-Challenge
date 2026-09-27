"""Differentiable Soft-Macro F0.5 Loss for Entity Resolution.

Directly aligns neural network and ranker optimization with the competition objective:
Entity-Macro F0.5 with Singleton Preservation.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftMacroF05Loss(nn.Module):
    """Differentiable approximation of Entity-Macro F0.5 loss.

    Supports:
    1. Grouped anchor-level evaluation (each S1 anchor has a variable number of candidates).
    2. Precision-weighted beta=0.5 (4x penalty on false positives).
    3. Singleton penalty barrier for anchors with zero ground-truth matches (T=0).
    """

    def __init__(
        self,
        beta: float = 0.5,
        eps: float = 1e-7,
        singleton_penalty_weight: float = 1.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.beta = beta
        self.beta_sq = beta ** 2
        self.eps = eps
        self.singleton_penalty_weight = singleton_penalty_weight
        self.temperature = temperature

    def forward(
        self,
        logits_or_probs: torch.Tensor,
        labels: torch.Tensor,
        anchor_indices: torch.Tensor,
        is_logits: bool = True,
    ) -> torch.Tensor:
        """Compute Soft Macro F0.5 loss across entity groups.

        Args:
            logits_or_probs: (N,) tensor of candidate pair predictions.
            labels: (N,) binary ground-truth target labels (0 or 1).
            anchor_indices: (N,) contiguous integer IDs grouping candidates by anchor.
            is_logits: True if input is unconstrained logits; False if probabilities in [0, 1].

        Returns:
            Scalar loss tensor (minimizing 1 - SoftMacroF0.5 + singleton_penalty).
        """
        if is_logits:
            probs = torch.sigmoid(logits_or_probs / self.temperature)
        else:
            probs = torch.clamp(logits_or_probs, self.eps, 1.0 - self.eps)

        labels = labels.float()
        
        # Identify unique anchor groups
        unique_anchors, inverse_indices = torch.unique(anchor_indices, return_inverse=True)
        num_anchors = unique_anchors.size(0)

        # Aggregate soft True Positives and soft Predicted Positives per anchor
        # TP_soft = sum(p_i * y_i)
        # Pred_soft = sum(p_i)
        # True_count = sum(y_i)
        tp_soft = torch.zeros(num_anchors, device=probs.device, dtype=probs.dtype)
        pred_soft = torch.zeros(num_anchors, device=probs.device, dtype=probs.dtype)
        true_counts = torch.zeros(num_anchors, device=probs.device, dtype=probs.dtype)

        tp_soft.scatter_add_(0, inverse_indices, probs * labels)
        pred_soft.scatter_add_(0, inverse_indices, probs)
        true_counts.scatter_add_(0, inverse_indices, labels)

        # Separate singletons (|T| == 0) from multi-match entities (|T| > 0)
        is_singleton = (true_counts == 0)
        is_matched = ~is_singleton

        losses = []

        # 1. Matched entities (|T| > 0)
        if is_matched.any():
            tp = tp_soft[is_matched]
            pred = pred_soft[is_matched]
            true = true_counts[is_matched]

            soft_prec = (tp + self.eps) / (pred + self.eps)
            soft_rec = (tp + self.eps) / (true + self.eps)

            # F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)
            numerator = (1.0 + self.beta_sq) * soft_prec * soft_rec
            denominator = (self.beta_sq * soft_prec) + soft_rec + self.eps
            soft_f05 = numerator / denominator

            loss_matched = (1.0 - soft_f05).mean()
            losses.append(loss_matched)

        # 2. Singleton entities (|T| == 0)
        # For singletons, any predicted positive is a false merge that drops score to 0.0.
        # We apply an exponential soft-barrier penalty on pred_soft.
        if is_singleton.any():
            singleton_preds = pred_soft[is_singleton]
            # Soft penalty: when singleton_preds -> 0, loss -> 0
            loss_singleton = torch.log1p(torch.exp(singleton_preds) - 1.0 + self.eps).mean()
            losses.append(self.singleton_penalty_weight * loss_singleton)

        if not losses:
            return torch.tensor(0.0, device=probs.device, requires_grad=True)

        return sum(losses) / len(losses)
