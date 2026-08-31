"""
Faster R-CNN with ResNet-50-FPN backbone from torchvision.
Adapted for BDD100K (10 classes).
"""

import torchvision
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(nc: int = 10, pretrained: bool = True) -> torchvision.models.detection.FasterRCNN:
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)

    # Replace the classification head for nc classes (+ background)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, nc + 1)

    return model
