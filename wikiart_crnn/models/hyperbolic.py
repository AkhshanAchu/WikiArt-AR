import torch
import torch.nn as nn
import geoopt


class HyperbolicProjection(nn.Module):
    def __init__(self, embed_dim: int = 256, curv: float = 1.0):
        super().__init__()
        self.ball = geoopt.PoincareBall(c=curv)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        projected = self.proj(z)
        # .clone() ensures expmap0 works on a fresh tensor,
        # avoiding in-place modification issues in some geoopt versions
        return self.ball.expmap0(projected.clone())
