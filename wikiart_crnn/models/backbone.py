import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class ArtBackbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        base = (
            EfficientNet.from_pretrained("efficientnet-b0")
            if pretrained
            else EfficientNet.from_name("efficientnet-b0")
        )
        self.features   = base.extract_features
        self._swish     = base._swish
        self._conv_stem = base._conv_stem
        self._bn0       = base._bn0
        self._blocks    = base._blocks
        self._conv_head = base._conv_head
        self._bn1       = base._bn1
        self.out_channels = 1280

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
