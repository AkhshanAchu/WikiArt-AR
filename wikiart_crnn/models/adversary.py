import torch
import torch.nn as nn
from ..config import NUM_STYLES


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


class StyleAdversary(nn.Module):
    def __init__(self, embed_dim: int = 256, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(),
            nn.Linear(64, NUM_STYLES),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_rev = GradientReversal.apply(z, self.alpha)
        return self.fc(z_rev)
