"""
Shared image transforms that also adjust bounding boxes.
Wraps torchvision functional ops so they work with our (img, target) convention.

Notes on normalize=False:
    Torchvision detectors (Faster R-CNN, SSD) apply their own normalization
    and resizing internally via GeneralizedRCNNTransform / SSDTransform.
    For those models pass normalize=False (and optionally img_size=None) so
    we don't double-normalize.
"""

import random

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class ToTensor:
    def __call__(self, img, target):
        return F.to_tensor(img), target


class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img, target):
        return F.normalize(img, self.mean, self.std), target


class Resize:
    """
    Two modes:
      square=False (default): resize shorter edge to `size`; cap longer edge at
                              `max_size` if given.  Boxes scaled proportionally.
      square=True           : resize to exactly (size × size), stretching if needed.
                              Boxes scaled independently in x and y.
    """

    def __init__(self, size: int, max_size: int | None = None, square: bool = False):
        self.size = size
        self.max_size = max_size
        self.square = square

    def __call__(self, img, target):
        W, H = img.size

        if self.square:
            img = F.resize(img, [self.size, self.size], InterpolationMode.BILINEAR)
            sx = self.size / W
            sy = self.size / H
            if target is not None and len(target["boxes"]):
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] *= sx
                boxes[:, [1, 3]] *= sy
                target["boxes"] = boxes
        else:
            scale = self.size / min(H, W)
            if self.max_size and scale * max(H, W) > self.max_size:
                scale = self.max_size / max(H, W)
            new_H, new_W = int(H * scale), int(W * scale)
            img = F.resize(img, [new_H, new_W], InterpolationMode.BILINEAR)
            sx = sy = scale
            if target is not None and len(target["boxes"]):
                target["boxes"] = target["boxes"] * scale

        # Escala aplicada, para poder devolver las cajas a píxeles NATIVOS en la
        # evaluación. Sin esto, el AP por tamaño de objeto mediría umbrales
        # distintos en cada modelo (32 px a 300 px de entrada no es lo mismo que
        # 32 px a 800) y la comparación entre resoluciones sería falsa.
        if target is not None:
            target["scale"] = torch.tensor([sx, sy], dtype=torch.float32)

        return img, target


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            W, _ = img.size
            img = F.hflip(img)
            if target is not None and len(target["boxes"]):
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return img, target


def build_transforms(
    img_size: int | None,
    train: bool,
    max_size: int | None = None,
    square: bool = False,
    normalize: bool = True,
):
    """
    Build the (img, target) transform pipeline.

    Args:
        img_size : target size for the shorter edge (or square side).
                   Pass None to skip resizing (e.g. for torchvision detectors).
        train    : if True, adds RandomHorizontalFlip.
        max_size : cap longer edge (only used when square=False).
        square   : if True, resize to (img_size × img_size). Used for SSD-300.
        normalize: if False, skip ImageNet normalisation (torchvision detectors
                   apply their own normalization internally).
    """
    ops = []
    if img_size is not None:
        ops.append(Resize(img_size, max_size=max_size, square=square))
    if train:
        ops.append(RandomHorizontalFlip(0.5))
    ops.append(ToTensor())
    if normalize:
        ops.append(Normalize())
    return Compose(ops)
