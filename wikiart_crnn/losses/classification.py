import torch
import torch.nn.functional as F


def bias_aware_ce(logits: torch.Tensor, targets: torch.Tensor,
                  class_weights: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets, weight=class_weights)
