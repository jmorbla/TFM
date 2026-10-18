"""
Transforms de imagen compartidos, que también ajustan las cajas.
Envuelven las operaciones funcionales de torchvision para la convención (img, target).

Sobre normalize=False:
    Los detectores de torchvision (Faster R-CNN, SSD) normalizan y redimensionan
    internamente (GeneralizedRCNNTransform / SSDTransform). Para esos modelos se
    pasa normalize=False (y opcionalmente img_size=None) para no normalizar dos veces.
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
    Dos modos:
      square=False (por defecto): el lado corto pasa a `size`; el largo se limita a
                                  `max_size` si se indica. Cajas escaladas proporcionalmente.
      square=True               : redimensiona a (size × size) exactos, deformando si
                                  hace falta. Cajas escaladas por separado en x e y.
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
    Construye el pipeline de transforms sobre (img, target).

    Argumentos:
        img_size : tamaño del lado corto (o del lado del cuadrado).
                   None omite el redimensionado (p. ej. detectores de torchvision).
        train    : si es True, añade RandomHorizontalFlip.
        max_size : límite del lado largo (solo con square=False).
        square   : si es True, redimensiona a (img_size × img_size). Lo usa SSD-300.
        normalize: si es False, omite la normalización de ImageNet (los detectores
                   de torchvision normalizan internamente).
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
