import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingBCELoss(nn.Module):
    def __init__(self, smoothing: float = 0.08):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        smoothed_targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, smoothed_targets)
