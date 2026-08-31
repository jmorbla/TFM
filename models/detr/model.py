"""
DETR via HuggingFace Transformers for BDD100K (10 classes).

Convención de índices
---------------------
Con ``num_labels=nc`` e ``ignore_mismatched_sizes=True`` el head se
reinicializa con forma (nc+1,): clases [0…nc-1] y la clase "no-object"
al final (índice nc). El dataset BDD100K ya usa índices 0-indexed [0, nc-1],
por lo que NO se aplica ningún desplazamiento a las etiquetas.

Durante la inferencia, ``post_process_object_detection`` elimina la clase
no-object (prob[..., :-1].max(-1)) y devuelve índices 0-indexed [0, nc-1],
coherentes con el ground truth del dataset.
"""

import torch
import torch.nn as nn
from transformers import DetrConfig, DetrForObjectDetection


def build_model(nc: int = 10, pretrained: bool = True) -> nn.Module:
    model_id = "facebook/detr-resnet-50"
    if pretrained:
        # Load COCO-pretrained weights, then resize the classifier head
        model = DetrForObjectDetection.from_pretrained(
            model_id,
            num_labels=nc,
            ignore_mismatched_sizes=True,
        )
    else:
        config = DetrConfig(num_labels=nc)
        model = DetrForObjectDetection(config)

    return model


class DETRWrapper(nn.Module):
    """
    Thin wrapper so the training loop can call model(pixel_values, targets)
    the same way as torchvision detectors.

    HuggingFace DETR expects:
        pixel_values: (B, 3, H, W) — already normalized
        labels: list of dicts with 'class_labels' and 'boxes' (cx,cy,w,h normalized)
    """

    def __init__(self, nc: int = 10, pretrained: bool = True):
        super().__init__()
        self.model = build_model(nc, pretrained)

    def forward(self, images: torch.Tensor, targets: list[dict] | None = None):
        if self.training and targets is None:
            raise ValueError("targets required during training")

        if self.training:
            hf_labels = self._to_hf_labels(images, targets)
            out = self.model(pixel_values=images, labels=hf_labels)
            return {"loss": out.loss, "loss_dict": out.loss_dict}
        else:
            out = self.model(pixel_values=images)
            return out

    @staticmethod
    def _to_hf_labels(images: torch.Tensor, targets: list[dict]) -> list[dict]:
        _, _, H, W = images.shape
        hf = []
        for t in targets:
            boxes = t["boxes"]  # (N, 4) xyxy absolute
            # → cxcywh normalized
            cx = (boxes[:, 0] + boxes[:, 2]) / 2 / W
            cy = (boxes[:, 1] + boxes[:, 3]) / 2 / H
            w = (boxes[:, 2] - boxes[:, 0]) / W
            h = (boxes[:, 3] - boxes[:, 1]) / H
            hf.append({
                "class_labels": t["labels"],
                "boxes": torch.stack([cx, cy, w, h], dim=1),
            })
        return hf
