import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.T = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B = features.shape[0]
        features = F.normalize(features, dim=-1)
        sim = torch.matmul(features, features.T) / self.T

        labels   = labels.unsqueeze(1)
        pos_mask = (labels == labels.T).float()
        pos_mask.fill_diagonal_(0)   # safe: pos_mask is freshly created, not in graph

        # Mask the diagonal out-of-place using an identity mask
        # instead of fill_diagonal_(0) on exp_sim which is in the autograd graph
        diag_mask = 1.0 - torch.eye(B, device=features.device)
        exp_sim   = torch.exp(sim) * diag_mask   # zero diagonal without in-place op

        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        n_pos = pos_mask.sum(dim=1).clamp(min=1)
        loss  = -(pos_mask * log_prob).sum(dim=1) / n_pos
        return loss.mean()
