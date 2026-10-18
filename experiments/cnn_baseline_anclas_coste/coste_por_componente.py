"""
Desglose del coste computacional de cnn_baseline: GFLOPs del backbone, la FPN
y la cabeza sobre cada nivel, a la resolución real del modelo y a la de
YOLO11m.

Misma convención que evaluate.py::model_complexity: thop, GFLOPs = 2 × MACs.
Solo lectura: construye el modelo en memoria sin cargar pesos entrenados (el
coste no depende de ellos) y no escribe nada.

Uso (desde TFM/):
    python experiments/cnn_baseline_anclas_coste/coste_por_componente.py
"""

import copy
import sys
from pathlib import Path

import thop
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.cnn_baseline.model import build_model  # noqa: E402


def gflops(module, *inputs) -> float:
    macs, _ = thop.profile(copy.deepcopy(module), inputs=inputs, verbose=False)
    return macs * 2 / 1e9


RESOLUCIONES = [
    (640, 1137, "entrada real de la CNN (lado corto 640)"),
    (384, 640, "tensor de YOLO11m (640x384)"),
]


def main():
    m = build_model(nc=10, pretrained=False).eval()
    for H, W, tag in RESOLUCIONES:
        x = torch.zeros(1, 3, H, W)
        total = gflops(m, x)
        with torch.no_grad():
            f0 = m.layer0(x)
            f1 = m.layer1(f0)
            f2 = m.layer2(f1)
            f3 = m.layer3(f2)
            fpn = m.fpn({"0": f0, "1": f1, "2": f2, "3": f3})
        backbone = gflops(m.layer0, x) + gflops(m.layer1, f0) + gflops(m.layer2, f1) + gflops(m.layer3, f2)
        head = {k: gflops(m.cls_head, v) + gflops(m.reg_head, v) for k, v in sorted(fpn.items())}

        print(f"\n{tag}: {W}x{H}   total {total:.1f} GFLOPs")
        print(f"  backbone              {backbone:6.1f}")
        print(f"  FPN                   {total - backbone - sum(head.values()):6.1f}")
        for i, (k, v) in enumerate(head.items()):
            hw = tuple(fpn[k].shape[-2:])
            print(f"  cabeza sobre P{i + 2} {str(hw):>11s} {v:6.1f}  ({100 * v / total:.0f} %)")
        print(f"  total sin la cabeza sobre P2: {total - head['0']:.1f}")


if __name__ == "__main__":
    main()
