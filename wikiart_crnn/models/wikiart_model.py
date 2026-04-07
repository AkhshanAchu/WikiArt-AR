import torch
import torch.nn as nn
from .backbone import ArtBackbone
from .encoder import SequentialArtEncoder
from .heads import HierarchicalHead
from .adversary import StyleAdversary
from .hyperbolic import HyperbolicProjection
from ..config import EMBED_DIM


class WikiArtCRNNMamba(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone  = ArtBackbone(pretrained=pretrained)
        self.seq_enc   = SequentialArtEncoder(
            in_channels=self.backbone.out_channels,
            hidden=EMBED_DIM,
        )
        self.heads     = HierarchicalHead(embed_dim=EMBED_DIM)
        self.adversary = StyleAdversary(embed_dim=EMBED_DIM)
        self.hyp_proj  = HyperbolicProjection(embed_dim=EMBED_DIM)
        self.log_var   = nn.Parameter(torch.zeros(3))

    def forward(self, x: torch.Tensor) -> dict:
        feat_map = self.backbone(x)
        z = self.seq_enc(feat_map)

        style_logits, genre_logits, artist_logits = self.heads(z)
        adv_logits = self.adversary(z)
        hyp_emb    = self.hyp_proj(z)

        return {
            "embedding"     : z,
            "hyp_embedding" : hyp_emb,
            "style"         : style_logits,
            "genre"         : genre_logits,
            "artist"        : artist_logits,
            "adversarial"   : adv_logits,
        }

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.backbone(x)
        z = self.seq_enc(feat_map)
        return self.hyp_proj(z)
