"""
PyTorch Dataset classes for BDD100K.

- BDD100KDetection  : standard image → boxes (used by Faster R-CNN, SSD, DETR, CNN baseline)
- BDD100KSequence   : T consecutive frames → boxes for last frame (used by CNN+GRU)
"""

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class BDD100KDetection(Dataset):
    """Single-frame detection dataset in YOLO-label format."""

    def __init__(self, images_dir: str, labels_dir: str, transforms=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms

        # Build index: only images that have a corresponding label file
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


class BDD100KSequence(Dataset):
    """
    Returns T consecutive frames from the same BDD100K video clip.
    Labels are taken from the last frame of the sequence (the prediction target).

    Requires images organised as:  images_dir/<video_name>/<frame>.jpg
    and labels as:                 labels_dir/<video_name>/<frame>.txt
    (video-level structure, not the flat per-image structure).

    Set flat=True if you're using the flat layout and want to simulate
    sequences by treating temporally-adjacent filenames as consecutive frames.
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        T: int = 4,
        stride: int = 2,
        transforms=None,
        flat: bool = True,
        seed: int = 42,
        require_same_video: bool = False,
    ):
        """
        Parameters
        ----------
        require_same_video : bool
            Si True, agrupa por prefijo de vídeo (BDD100K: <videoid>-<frame>.jpg)
            y solo crea secuencias dentro del mismo vídeo.
            Si False (por defecto), trata todas las imágenes ordenadas como una
            sola secuencia global.

            ADVERTENCIA METODOLÓGICA
            ------------------------
            BDD100K-100k (el split de detección) contiene UN keyframe por vídeo:
            medido sobre el subset de 8 000 imágenes hay 7 263 prefijos distintos
            y ninguno supera los 7 frames, mientras que una secuencia necesita
            T*stride = 8. Es decir, require_same_video=True generaría CERO
            secuencias y require_same_video=False produce secuencias de imágenes
            urbanas SIN NINGUNA relación temporal entre sí.

            Consecuencia: con este dataset el GRU no puede aprender dinámica
            temporal, sólo a ignorar los T-1 primeros frames. Cualquier
            conclusión del tipo "el módulo recurrente no aporta mejora" mide la
            ausencia de señal temporal en los datos, NO las capacidades del
            modelo. Para un experimento temporal válido hace falta BDD100K MOT.
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.T = T
        self.stride = stride
        self.transforms = transforms

        # Build sequences
        if flat:
            stems = sorted(
                p.stem for p in self.labels_dir.glob("*.txt")
                if (self.images_dir / (p.stem + ".jpg")).exists()
            )

            self.sequences = []
            if require_same_video:
                # Agrupar por prefijo de vídeo y crear secuencias dentro de cada grupo
                from collections import defaultdict
                videos: dict[str, list[str]] = defaultdict(list)
                for s in stems:
                    vid = s.rsplit("-", 1)[0] if "-" in s else s
                    videos[vid].append(s)
                for frames in videos.values():
                    frames.sort()
                    step = T * stride
                    for i in range(0, len(frames) - step + 1, stride):
                        seq = frames[i: i + step: stride]
                        if len(seq) == T:
                            self.sequences.append(seq)
            else:
                # Ventana deslizante global sobre todas las imágenes ordenadas.
                # Las secuencias NO tienen continuidad temporal: ver la
                # advertencia metodológica del docstring.
                import warnings
                warnings.warn(
                    "BDD100KSequence: require_same_video=False — las secuencias "
                    "agrupan imágenes de vídeos distintos y no contienen señal "
                    "temporal real. Válido solo como ablación negativa; para un "
                    "experimento temporal usa BDD100K MOT.",
                    stacklevel=2,
                )
                step = T * stride
                for i in range(0, len(stems) - step + 1, stride):
                    seq = stems[i: i + step: stride]
                    if len(seq) == T:
                        self.sequences.append(seq)
        else:
            raise NotImplementedError("Non-flat layout not yet supported.")

        if len(self.sequences) == 0:
            raise RuntimeError(
                f"BDD100KSequence: 0 secuencias generadas en '{images_dir}'. "
                f"Con un frame por vídeo usa require_same_video=False (por defecto)."
            )

        rng = random.Random(seed)
        rng.shuffle(self.sequences)

    def __len__(self):
        return len(self.sequences)

    def _load_frame(self, stem: str):
        img = Image.open(self.images_dir / f"{stem}.jpg").convert("RGB")
        return img

    def _load_labels(self, stem: str, W: int, H: int):
        label_path = self.labels_dir / f"{stem}.txt"
        boxes, labels = [], []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
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
        return boxes, labels

    def __getitem__(self, idx):
        seq_stems = self.sequences[idx]
        frames = [self._load_frame(s) for s in seq_stems]
        W, H = frames[-1].size

        boxes, labels = self._load_labels(seq_stems[-1], W, H)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }

        if self.transforms:
            # Todos los frames de la secuencia deben recibir EXACTAMENTE la misma
            # transformación espacial. Sin esto, RandomHorizontalFlip sorteaba
            # random.random() de forma independiente en cada frame y unos salían
            # espejados y otros no dentro de la misma secuencia, destruyendo la
            # coherencia espacial que el GRU necesita para modelar el movimiento.
            rng_state = random.getstate()
            frames_t = []
            for i, fr in enumerate(frames):
                random.setstate(rng_state)   # misma decisión para cada frame
                if i == len(frames) - 1:
                    fr, target = self.transforms(fr, target)
                else:
                    fr, _ = self.transforms(fr, {
                        "boxes":    target["boxes"].clone(),
                        "labels":   target["labels"].clone(),
                        "image_id": target["image_id"],
                    })
                frames_t.append(fr)
            frames = frames_t

        # Stack: (T, C, H, W)
        frames_tensor = torch.stack(frames)
        return frames_tensor, target
