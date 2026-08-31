"""
Hybrid CNN+GRU model for temporal object detection.

Input : (B, T, 3, H, W)  — T consecutive frames
Output: (cls_preds, reg_preds)  — predictions for the last frame

Architecture (Section 3.7.4 of the thesis):
    1. MobileNetV3-Small backbone encodes each frame independently → feature vector
    2. Bidirectional GRU models temporal evolution across T frames
    3. Shared detection head predicts boxes and classes from GRU output
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.ops import FeaturePyramidNetwork

from models.cnn_baseline.loss import NUM_ANCHORS, NUM_FPN_LEVELS


class CNNEncoder(nn.Module):
    """Per-frame CNN feature extractor (shared weights across time steps)."""

    def __init__(self, pretrained: bool = True, fpn_channels: int = 96):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights).features
        # MobileNetV3-Small: [:4]=24ch stride8, [4:8]=48ch stride16, [8:12]=96ch stride32
        # features[12] es Conv(96→576) de expansión final — excluir
        self.layer0 = backbone[:2]    # stride 4,  out: 16 ch
        self.layer1 = backbone[2:4]   # stride 8,  out: 24 ch
        self.layer2 = backbone[4:8]   # stride 16, out: 48 ch
        self.layer3 = backbone[8:12]  # stride 32, out: 96 ch
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[16, 24, 48, 96],
            out_channels=fpn_channels,
        )
        self.fpn_channels = fpn_channels

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        f0 = self.layer0(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return self.fpn({"0": f0, "1": f1, "2": f2, "3": f3})


class CNNGRUDetector(nn.Module):
    """
    For each FPN scale independently:
        - Flatten spatial dims → sequence of T feature maps per scale
        - Run bidirectional GRU over time
        - Detection head on the last hidden state, reshaped to spatial grid
    """

    def __init__(
        self,
        nc: int = 10,
        T: int = 4,
        fpn_channels: int = 96,
        gru_hidden: int = 256,
        gru_layers: int = 1,
        bidirectional: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()
        self.nc = nc
        self.T = T
        self.num_anchors = NUM_ANCHORS   # ver models/cnn_baseline/loss.py
        self.fpn_channels = fpn_channels
        self.gru_hidden = gru_hidden
        self.bi = 2 if bidirectional else 1

        self.encoder = CNNEncoder(pretrained=pretrained, fpn_channels=fpn_channels)

        # One GRU per FPN scale
        self.grus = nn.ModuleDict({
            str(i): nn.GRU(
                input_size=fpn_channels,
                hidden_size=gru_hidden,
                num_layers=gru_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
            for i in range(NUM_FPN_LEVELS)
        })

        head_in = gru_hidden * self.bi
        self.cls_heads = nn.ModuleDict({
            str(i): nn.Conv2d(head_in, self.num_anchors * nc, 1)
            for i in range(NUM_FPN_LEVELS)
        })
        self.reg_heads = nn.ModuleDict({
            str(i): nn.Conv2d(head_in, self.num_anchors * 4, 1)
            for i in range(NUM_FPN_LEVELS)
        })

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, 3, H, W)
        """
        B, T, C, H, W = x.shape
        assert T == self.T

        # Encode all frames: (B*T, 3, H, W) → FPN feature maps
        frames_flat = x.view(B * T, C, H, W)
        fpn_all = self.encoder(frames_flat)

        cls_preds, reg_preds = [], []
        for i, key in enumerate(sorted(fpn_all)):
            feat = fpn_all[key]                        # (B*T, fpn_ch, fH, fW)
            _, fC, fH, fW = feat.shape

            # Temporal reshape: (B, T, fpn_ch, fH, fW) → (B, fH*fW, T, fpn_ch)
            feat_bt = feat.view(B, T, fC, fH, fW)
            feat_sp = feat_bt.permute(0, 3, 4, 1, 2).reshape(B * fH * fW, T, fC)

            gru_out, _ = self.grus[str(i)](feat_sp)    # (B*fH*fW, T, gru_h*bi)
            last = gru_out[:, -1, :]                   # take last time step

            # Reshape back to spatial: (B, gru_h*bi, fH, fW)
            last_spatial = last.view(B, fH, fW, -1).permute(0, 3, 1, 2)

            cls_preds.append(self.cls_heads[str(i)](last_spatial))
            reg_preds.append(self.reg_heads[str(i)](last_spatial))

        return cls_preds, reg_preds


def build_model(
    nc: int = 10,
    T: int = 4,
    gru_hidden: int = 256,
    pretrained: bool = True,
) -> CNNGRUDetector:
    return CNNGRUDetector(nc=nc, T=T, gru_hidden=gru_hidden, pretrained=pretrained)
