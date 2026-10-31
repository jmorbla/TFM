"""Variante de 02_curvas_pr: corta cada curva donde acaba el recall alcanzable (sin la caida a 0)."""
import json
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt

RUNS = Path("../runs")
CONF = [("YOLO11m (A)", "yolo_a_final", "#2a78d6"), ("YOLO11m (B)", "yolo_b_final", "#eb6834"),
        ("Faster R-CNN", "frcnn_final", "#1baf7a"), ("SSD300", "ssd_final", "#eda100"),
        ("DETR", "detr_final", "#e87ba4"), ("CNN baseline", "cnnb_final", "#008300")]
CLASES = ["car", "traffic light", "person", "rider"]
x = np.linspace(0, 1, 101)


def cargar(pref, s):
    c = [p for p in RUNS.rglob(f"{pref}_s{s}") if p.is_dir()]
    for f in (c[0] / "eval_results.json", c[0] / "weights" / "eval_results.json"):
        if f.exists():
            return json.load(open(f))


fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for ax, clase in zip(axes.flat, CLASES):
    for etq, pref, col in CONF:
        cur = []
        for s in range(3):
            d = cargar(pref, s)
            if d and d["per_class"].get(clase, {}).get("pr_curve_iou50"):
                a = np.array(d["per_class"][clase]["pr_curve_iou50"], float)
                nz = np.where(a > 0)[0]
                a[nz[-1] + 1:] = np.nan          # recall inalcanzable: no se dibuja
                cur.append(a)
        if not cur:
            continue
        m = np.nanmean(np.array(cur), axis=0)
        ult = np.where(~np.isnan(m))[0][-1]
        ax.plot(x[:ult + 1], m[:ult + 1], color=col, lw=1.8, label=etq)
        ax.plot(x[ult], m[ult], "o", color=col, ms=4)   # final = recall maximo
    ax.set_title(clase); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
h, l = axes.flat[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=6, frameon=False)
fig.suptitle("Curvas Precision-Recall a IoU=0.5 (cada curva termina en su recall máximo)", y=1.08)
fig.tight_layout()
fig.savefig("figuras_pr_iou/02b_curvas_pr_sin_caida.png", dpi=200, bbox_inches="tight")
fig.savefig("figuras_pr_iou/02b_curvas_pr_sin_caida.pdf", bbox_inches="tight")
