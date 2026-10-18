"""
Clase Dataset de PyTorch para BDD100K.

- BDD100KDetection  : imagen → cajas (la usan todos los modelos salvo YOLO11)
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class BDD100KDetection(Dataset):
    """Dataset de detección de un solo fotograma, con etiquetas en formato YOLO."""

    def __init__(self, images_dir: str, labels_dir: str, transforms=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms

        # Índice: solo las imágenes que tienen fichero de etiquetas
        self.samples = [
            p.stem for p in self.labels_dir.glob("*.txt")
            if (self.images_dir / (p.stem + ".jpg")).exists()
        ]
        self.samples.sort()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem = self.samples[idx]
        img = Image.open(self.images_dir / f"{stem}.jpg").convert("RGB")
        W, H = img.size

        boxes, labels = [], []
        for line in (self.labels_dir / f"{stem}.txt").read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
            x1 = (cx - w / 2) * W
            y1 = (cy - h / 2) * H
            x2 = (cx + w / 2) * W
            y2 = (cy + h / 2) * H
            boxes.append([x1, y1, x2, y2])
            labels.append(cls)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target
