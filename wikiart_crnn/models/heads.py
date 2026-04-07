import torch
import torch.nn as nn
from ..config import NUM_STYLES, NUM_GENRES, NUM_ARTISTS


class HierarchicalHead(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.style_fc = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, NUM_STYLES),
        )
        self.genre_fc = nn.Sequential(
            nn.Linear(embed_dim + NUM_STYLES, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, NUM_GENRES),
        )
        self.artist_fc = nn.Sequential(
            nn.Linear(embed_dim + NUM_GENRES, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, NUM_ARTISTS),
        )

    def forward(self, z: torch.Tensor):
        style_logits  = self.style_fc(z)
        genre_logits  = self.genre_fc(torch.cat([z, style_logits.detach()], dim=-1))
        artist_logits = self.artist_fc(torch.cat([z, genre_logits.detach()], dim=-1))
        return style_logits, genre_logits, artist_logits
