"""Losses for imbalanced segmentation.

Plain cross-entropy weights every pixel independently, so with a huge
background class it's easy for gradients to stay dominated by "get the
background right" even after inverse-frequency weighting. Tversky loss
instead works on a per-class overlap ratio (like Dice/IoU), and exposes
separate penalties for false positives (alpha) and false negatives (beta) --
setting beta > alpha directly tells the optimizer that missing a real
class_10/12 pixel is worse than a spurious extra one, which is a more
principled way to chase recall than post-hoc logit biasing or loss-weight
multipliers alone.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, ignore_index: int | None = None, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor, class_ids) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        num_classes = logits.shape[1]
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

        losses = []
        for c in class_ids:
            p, t = probs[:, c], target_one_hot[:, c]
            tp = (p * t).sum()
            fp = (p * (1 - t)).sum()
            fn = ((1 - p) * t).sum()
            tversky_index = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
            losses.append(1 - tversky_index)
        return torch.stack(losses).mean()


class CombinedLoss(nn.Module):
    """ce_weight * CrossEntropy + tversky_weight * TverskyLoss(target classes only)."""

    def __init__(self, class_weights: torch.Tensor, num_classes: int,
                 ce_weight: float = 0.5, tversky_weight: float = 0.5,
                 tversky_alpha: float = 0.3, tversky_beta: float = 0.7):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.ce_weight = ce_weight
        self.tversky_weight = tversky_weight
        self.target_class_ids = list(range(1, num_classes))  # skip background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        if self.ce_weight:
            loss = loss + self.ce_weight * self.ce(logits, target)
        if self.tversky_weight:
            loss = loss + self.tversky_weight * self.tversky(logits, target, self.target_class_ids)
        return loss
