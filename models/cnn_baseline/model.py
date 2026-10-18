"""
CNN propia: backbone MobileNetV3-Small + cuello FPN simple + head con anchors.

Es el detector más ligero posible que aún sigue el esquema backbone → cuello → head.
"""

import math

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.ops import FeaturePyramidNetwork

from models.cnn_baseline.loss import FOCAL_PRIOR, NUM_ANCHORS


class CNNBaseline(nn.Module):
    """
    Detector de una etapa (estilo RetinaNet):
        backbone: MobileNetV3-Small (preentrenado en ImageNet)
        neck    : FPN de 4 niveles sobre los strides 4, 8, 16, 32 (P2-P5)
        head    : torre compartida de 4 convs + GroupNorm por rama, con el bias
                  de clasificación inicializado al prior de la focal loss
    """

    def __init__(self, nc: int = 10, fpn_channels: int = 96, pretrained: bool = True,
                 head_convs: int = 4):
        super().__init__()
        self.nc = nc
        # Fuente única de verdad: la configuración de anchors vive en loss.py.
        # Duplicarla aquí desincronizaría el head respecto al matcher.
        self.num_anchors = NUM_ANCHORS

        # ── Backbone ────────────────────────────────────────────────────────
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights).features
        # Mapa de canales/stride de MobileNetV3-Small features (torchvision):
        #  [0]  Conv(3→16,  s2)
        #  [1]  IR(16→16,  s2)
        #  [2]  IR(16→24,  s2)  stride acumulado 8
        #  [3]  IR(24→24,  s1)  ← aquí termina layer1: 24 ch, stride 8
        #  [4]  IR(24→40,  s2)  stride acumulado 16
        #  [5]  IR(40→40,  s1)
        #  [6]  IR(40→40,  s1)
        #  [7]  IR(40→48,  s1)  ← aquí termina layer2: 48 ch, stride 16
        #  [8]  IR(48→48,  s1)
        #  [9]  IR(48→96,  s2)  stride acumulado 32
        #  [10] IR(96→96,  s1)
        #  [11] IR(96→96,  s1)  ← aquí termina layer3: 96 ch, stride 32
        #  [12] Conv(96→576, s1) — expansión final; NO incluir (cambiaría canales de FPN)
        # P2 (stride 4) se conserva: es el nivel donde viven la mayoría de los
        # objetos de BDD100K.
        self.layer0 = backbone[:2]    # stride 4,  out: 16 ch
        self.layer1 = backbone[2:4]   # stride 8,  out: 24 ch
        self.layer2 = backbone[4:8]   # stride 16, out: 48 ch
        self.layer3 = backbone[8:12]  # stride 32, out: 96 ch

        # ── Neck (FPN) ───────────────────────────────────────────────────────
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[16, 24, 48, 96],
            out_channels=fpn_channels,
        )

        # ── Head ─────────────────────────────────────────────────────────────
        # Torre de `head_convs` convoluciones con GroupNorm, siguiendo RetinaNet.
        # GroupNorm y no BatchNorm porque el head es compartido entre niveles FPN
        # de tamaños muy distintos: las estadísticas de BN serían inconsistentes.
        head_ch = fpn_channels
        self.cls_head = self._build_tower(head_ch, self.num_anchors * nc, head_convs)
        self.reg_head = self._build_tower(head_ch, self.num_anchors * 4, head_convs)
        self._init_weights()

    @staticmethod
    def _build_tower(ch: int, out_ch: int, n_convs: int) -> nn.Sequential:
        layers = []
        for _ in range(n_convs):
            layers += [
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.GroupNorm(32, ch),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Conv2d(ch, out_ch, 1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        """Inicialización estándar de RetinaNet (Lin et al. 2017, §4.1).

        La clave es el bias de la última capa de clasificación: se fija a
        -log((1-π)/π) con π=0.01, de forma que la red arranque prediciendo
        probabilidad 0.01 para cada clase en cada anchor. Sin esto la red parte
        prediciendo 0.5 sobre cientos de miles de anchors casi todos negativos,
        la pérdida inicial se dispara (medido: 10777) y las primeras épocas se
        gastan en corregir la inicialización en vez de en aprender.
        """
        for tower in (self.cls_head, self.reg_head):
            for m in tower.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
        prior_bias = -math.log((1.0 - FOCAL_PRIOR) / FOCAL_PRIOR)
        nn.init.constant_(self.cls_head[-1].bias, prior_bias)

    def forward(self, x: torch.Tensor):
        f0 = self.layer0(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)

        fpn_out = self.fpn({"0": f0, "1": f1, "2": f2, "3": f3})

        cls_preds, reg_preds = [], []
        for key in sorted(fpn_out):
            feat = fpn_out[key]
            cls_preds.append(self.cls_head(feat))
            reg_preds.append(self.reg_head(feat))

        return cls_preds, reg_preds


def build_model(nc: int = 10, pretrained: bool = True) -> CNNBaseline:
    return CNNBaseline(nc=nc, pretrained=pretrained)
