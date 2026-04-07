import torch
import torch.nn as nn
from ..config import USE_MAMBA, Mamba


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class SequentialArtEncoder(nn.Module):
    def __init__(self, in_channels: int = 1280, hidden: int = 256, gru_layers: int = 1):
        super().__init__()

        self.proj = nn.Linear(in_channels, hidden)

        self.bigru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden // 2,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if gru_layers > 1 else 0.0,
        )

        if USE_MAMBA:
            self.global_enc = Mamba(
                d_model=hidden,
                d_state=16,
                d_conv=4,
                expand=2,
            )
            self._use_mamba = True
        else:
            self.global_enc = MultiHeadSelfAttention(hidden=hidden, num_heads=8)
            self._use_mamba = False

        self.norm = nn.LayerNorm(hidden)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat_map.shape
        x = feat_map.flatten(2).permute(0, 2, 1)
        x = self.proj(x)

        gru_out, _ = self.bigru(x)

        if self._use_mamba:
            global_out = self.global_enc(gru_out)
        else:
            global_out = self.global_enc(gru_out)

        out = self.norm(gru_out + global_out)
        out = self.pool(out.permute(0, 2, 1))
        return out.squeeze(-1)
