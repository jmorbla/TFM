"""
Cobertura de anchors de cnn_baseline: qué parte de los objetos anotados
encuentra un anchor con IoU >= 0.5, para cada configuración por la que pasó el
diseño y para variantes auxiliares que aíslan cada cambio.

Solo lectura: no entrena ni modifica nada.

Se evalúa sobre TODOS los objetos de la partición de validación (36 671).

Uso (desde TFM/):
    python experiments/cnn_baseline_anclas_coste/cobertura_anclas.py
"""

import math
import sys
from pathlib import Path

import torch
from torchvision.ops import box_iou

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.cnn_baseline import loss as L  # noqa: E402

LBL = ROOT / "data_yolo" / "labels" / "val"
W0, H0 = 1280, 720
DEV = "cuda" if torch.cuda.is_available() else "cpu"
RATIOS = [0.5, 1.0, 2.0]
UNA, DOS = [1.0], [1.0, 1.6]   # escalas por nivel: 3 o 6 anchors por celda


def feature_shapes(short: int, strides):
    """Tamaño de la entrada (lado corto = short) y de cada mapa de la FPN."""
    s = short / min(H0, W0)
    Hi, Wi = int(H0 * s), int(W0 * s)
    return s, [(math.ceil(Hi / st), math.ceil(Wi / st)) for st in strides]


def anchors(short: int, levels):
    """levels: lista de (stride, tamaño base, escalas)."""
    s, shapes = feature_shapes(short, [st for st, _, _ in levels])
    parts = []
    for (stride, base, scales), (fH, fW) in zip(levels, shapes):
        cx = (torch.arange(fW, device=DEV) + 0.5) * stride
        cy = (torch.arange(fH, device=DEV) + 0.5) * stride
        gy, gx = torch.meshgrid(cy, cx, indexing="ij")
        gx, gy = gx.flatten(), gy.flatten()
        per = []
        for sc in scales:
            for r in RATIOS:
                w, h = base * sc * math.sqrt(r), base * sc / math.sqrt(r)
                per.append(torch.stack([gx - w / 2, gy - h / 2, gx + w / 2, gy + h / 2], 1))
        # Mismo orden que loss._anchors_for_level: celda, luego escala×ratio.
        parts.append(torch.stack(per, 1).reshape(-1, 4))
    return torch.cat(parts), s


def load_gt():
    out = []
    for f in sorted(LBL.glob("*.txt")):
        b = []
        for line in f.read_text().splitlines():
            q = line.split()
            if len(q) != 5:
                continue
            cx, cy, w, h = map(float, q[1:])
            b.append([(cx - w / 2) * W0, (cy - h / 2) * H0, (cx + w / 2) * W0, (cy + h / 2) * H0])
        if b:
            out.append(torch.tensor(b))
    return out


def coverage(gt, short, levels):
    A, s = anchors(short, levels)
    hits = tot = 0
    best = []
    for g in gt:
        m = box_iou(g.to(DEV) * s, A).max(1).values
        hits += int((m >= 0.5).sum())
        tot += len(g)
        best.append(m.cpu())
    return 100 * hits / tot, torch.cat(best).median().item(), len(A)


FINAL = [(st, b, DOS) for st, b in zip(L._STRIDES, L._ANCHOR_BASE_SIZES)]
CONFIGS = [
    ("1 original   45/91/181 px, 3 anc., P3-P5 @416", 416, [(8, 45, UNA), (16, 91, UNA), (32, 181, UNA)]),
    ("2 anclas     12/24/48 px, 6 anc., P3-P5 @416", 416, [(8, 12, DOS), (16, 24, DOS), (32, 48, DOS)]),
    ("  aux: solo reescalar a 16/32/64, P3-P5 @416", 416, [(8, 16, DOS), (16, 32, DOS), (32, 64, DOS)]),
    ("  aux: solo añadir P2 (8) a 12/24/48 @416", 416, [(4, 8, DOS), (8, 12, DOS), (16, 24, DOS), (32, 48, DOS)]),
    ("3 +P2        8/16/32/64 px, 6 anc., P2-P5 @416", 416, FINAL),
    ("  aux: final sin P2, P3-P5 @640", 640, FINAL[1:]),
    ("4 final      8/16/32/64 px, 6 anc., P2-P5 @640", 640, FINAL),
]


def main():
    # Comprobación: la configuración final reproduce exactamente los anchors
    # que genera el propio código del modelo.
    _, shapes = feature_shapes(640, L._STRIDES)
    ref = L.generate_anchors(shapes, DEV)
    mine, _ = anchors(640, FINAL)
    assert ref.shape == mine.shape and torch.allclose(ref, mine), \
        "Los anchors reproducidos no coinciden con models/cnn_baseline/loss.py"

    gt = load_gt()
    print(f"Validación: {len(gt)} imágenes, {sum(len(g) for g in gt)} objetos\n")
    for name, short, levels in CONFIGS:
        c, med, n = coverage(gt, short, levels)
        print(f"{name:48s} cobertura {c:5.1f} %   IoU mediano {med:.3f}   anchors {n}")


if __name__ == "__main__":
    main()
