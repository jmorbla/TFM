"""
SSD300-VGG16 de torchvision, adaptado a BDD100K (10 clases).
"""

import torchvision
from torchvision.models.detection import SSD300_VGG16_Weights, ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead


def build_model(nc: int = 10, pretrained: bool = True) -> torchvision.models.detection.ssd.SSD:
    weights = SSD300_VGG16_Weights.COCO_V1 if pretrained else None
    model = ssd300_vgg16(weights=weights)

    # Sustituye el head de clasificación por uno de nc clases (+ fondo)
    in_channels = [512, 1024, 512, 256, 256, 256]
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=nc + 1,
    )

    return model
