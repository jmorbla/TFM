# -*- coding: utf-8 -*-
"""Genera graficas_cualitativas.ipynb: ejemplos reales de detecciones de los 6
modelos (semilla 0) sobre imagenes de validacion seleccionadas.

Reutiliza las funciones de evaluate.py (load_model, get_val_transform,
postprocess_predictions, _to_native) importandolo como modulo -- NO se duplica
logica de inferencia ni se modifica evaluate.py.

Decisiones metodologicas (declaradas tambien en el propio notebook):
  - Seleccion de imagenes DETERMINISTA por numero de instancias de GT (minimo
    no nulo, percentiles 20/40/60/80, maximo), no aleatoria ni a mano -- evita
    la sospecha de cherry-picking de buenos resultados.
  - Umbral de confianza de VISUALIZACION = conf_max_f1 de cada modelo (el
    punto de operacion que ya se usa para P/R/F1 en la evaluacion final), no
    el conf=0.001 del protocolo cuantitativo -- con 0.001 cada imagen mostraria
    decenas de cajas de ruido, irreconocible visualmente. Se declara la
    diferencia explicitamente para que no se confunda con el protocolo.
  - Semilla 0 en los 6 modelos, por consistencia con la Figura 5 de
    generar_graficas_entrenamiento.py (LR schedule), que ya adopto ese criterio
    de "representativo, no las 3 semillas".

Uso (desde TFM/, con los 18 runs entrenados y resumen.json generado):
    python graficas_cualitativas/generar_graficas_cualitativas.py
    python -m jupyter nbconvert --to notebook --execute --inplace graficas_cualitativas/graficas_cualitativas.ipynb
"""
import json
from pathlib import Path


def cell(ctype, text):
    lines = text.strip("\n").split("\n")
    source = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines else [])
    d = {"cell_type": ctype, "metadata": {}, "source": source}
    if ctype == "code":
        d["outputs"] = []
        d["execution_count"] = None
    return d


def md(text):
    return cell("markdown", text)


def code(text):
    return cell("code", text)


cells = []

cells.append(md(r"""
# Ejemplos cualitativos de detección

Cajas predichas (semilla 0 de cada configuración) frente al ground truth, sobre
imágenes reales de validación. Complementa las métricas agregadas de
`graficas_resultados.ipynb` con la pregunta que un tribunal siempre hace:
*"enséñame cómo se ve"*.

## Decisiones metodológicas

- **Selección de imágenes determinista**, por número de instancias de GT
  (mínimo no nulo, percentiles 20/40/60/80, máximo) — no aleatoria ni elegida a
  mano, para que no haya sospecha de haber escogido los mejores casos.
- **Umbral de confianza de visualización = `conf_max_f1` de cada modelo**
  (el mismo punto de operación que ya se usa para P/R/F1 en la evaluación
  final), **no** el `conf=0.001` del protocolo cuantitativo — con 0.001 cada
  imagen mostraría decenas de cajas de ruido y sería ilegible. Esto es solo
  para la figura; no afecta a ninguna métrica.
- **Semilla 0** en los 6 modelos, igual que en la Figura 5 (LR schedule) de
  `graficas_entrenamiento.ipynb`.
- Reutiliza `evaluate.py` (`load_model`, `get_val_transform`,
  `postprocess_predictions`, `_to_native`) importado como módulo — la
  inferencia es exactamente la misma que produce las métricas finales, no una
  reimplementación aparte.
"""))

cells.append(code(r"""
import json
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, "..")  # evaluate.py vive en la raiz de TFM/, no en esta carpeta
import evaluate  # reutiliza load_model, get_val_transform, postprocess_predictions, _to_native

FIG_DIR = Path("figuras_cualitativas")
FIG_DIR.mkdir(exist_ok=True)
DEVICE = torch.device(evaluate.DEVICE_TYPE)
N_IMAGENES = 6

with open("../resumen.json", encoding="utf-8") as f:
    RESUMEN = json.load(f)

bdd = yaml.safe_load(Path("../configs/bdd100k.yaml").read_text(encoding="utf-8"))
IMAGES_VAL = Path("..") / bdd["dataset"]["images_dir"] / "val"
LABELS_VAL = Path("..") / bdd["dataset"]["labels_dir"] / "val"
NC = bdd["dataset"]["nc"]
CLASS_NAMES = evaluate.CLASS_NAMES

# (etiqueta, prefijo del run_name, config yaml, ruta de pesos relativa al run, es_yolo)
CONFIGS = [
    ("YOLO11m (A)",  "yolo_a_final", "../configs/yolo11m.yaml",       "weights/best.pt", True),
    ("YOLO11m (B)",  "yolo_b_final", "../configs/yolo11m_noaug.yaml", "weights/best.pt", True),
    ("Faster R-CNN", "frcnn_final",  "../configs/faster_rcnn.yaml",   "best.pt",         False),
    ("SSD300",       "ssd_final",    "../configs/ssd.yaml",           "best.pt",         False),
    ("DETR",         "detr_final",   "../configs/detr.yaml",          "best.pt",         False),
    ("CNN baseline", "cnnb_final",   "../configs/cnn_baseline.yaml",  "best.pt",         False),
]

RUNS = Path("../runs")


def buscar_pesos(prefijo, sufijo_pesos):
    cands = sorted(RUNS.rglob(f"{prefijo}_s0"))
    for c in cands:
        p = c / sufijo_pesos
        if p.exists():
            return p
    return None


# 10 colores por clase; cada caja lleva además su etiqueta de texto.
CLASS_COLORS = {
    "car":            "#2a78d6",
    "traffic sign":   "#eb6834",
    "traffic light":  "#1baf7a",
    "person":         "#eda100",
    "truck":          "#e87ba4",
    "bus":            "#008300",
    "bike":           "#4a3aa7",
    "rider":          "#e34948",
    "motor":          "#8B5E3C",
    "train":          "#4a2545",
}

print(f"Val: {IMAGES_VAL}  ({sum(1 for _ in LABELS_VAL.glob('*.txt'))} etiquetas)")
"""))

cells.append(md(r"""
## Selección determinista de imágenes

Por número de instancias de GT: mínimo no nulo, percentiles 20/40/60/80 y
máximo — cubre desde escenas casi vacías hasta las más congestionadas.
"""))

cells.append(code(r"""
def cargar_gt(stem):
    W, H = Image.open(IMAGES_VAL / f"{stem}.jpg").size
    boxes, labels = [], []
    for line in (LABELS_VAL / f"{stem}.txt").read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
        boxes.append([(cx - w/2)*W, (cy - h/2)*H, (cx + w/2)*W, (cy + h/2)*H])
        labels.append(c)
    return (np.array(boxes, dtype=np.float32).reshape(-1, 4),
            np.array(labels, dtype=np.int64))


stems = sorted(p.stem for p in LABELS_VAL.glob("*.txt")
               if (IMAGES_VAL / f"{p.stem}.jpg").exists())
conteos = np.array([len(cargar_gt(s)[1]) for s in stems])
orden = np.argsort(conteos)
stems_ordenados = [stems[i] for i in orden]
conteos_ordenados = conteos[orden]

no_vacios = np.where(conteos_ordenados > 0)[0]
idx_pick = [no_vacios[0]]
for pct in (20, 40, 60, 80):
    idx_pick.append(no_vacios[int(len(no_vacios) * pct / 100)])
idx_pick.append(no_vacios[-1])
idx_pick = sorted(set(idx_pick))

SELECCION = [stems_ordenados[i] for i in idx_pick]
for s in SELECCION:
    b, l = cargar_gt(s)
    clases = ", ".join(f"{CLASS_NAMES[c]}x{(l==c).sum()}" for c in sorted(set(l.tolist())))
    print(f"{s}   {len(l)} instancias   [{clases}]")
"""))

cells.append(md(r"""
## Inferencia — un modelo cada vez, se libera GPU antes de cargar el siguiente

(GPU de 8 GB, margen justo con DETR/Faster R-CNN — no se cargan los 6 modelos
a la vez.)
"""))

cells.append(code(r"""
def preds_torchvision(etq, prefijo, config_yaml, sufijo_pesos):
    peso = buscar_pesos(prefijo, sufijo_pesos)
    cfg = yaml.safe_load(Path(config_yaml).read_text(encoding="utf-8"))
    img_size = cfg["training"]["img_size"]
    conf_disp = RESUMEN[etq]["conf_max_f1"]["media"]

    model = evaluate.load_model(
        {"Faster R-CNN": "faster_rcnn", "SSD300": "ssd", "DETR": "detr",
         "CNN baseline": "cnn_baseline"}[etq],
        cfg, str(peso), NC, DEVICE, conf=conf_disp,
    )
    val_tf = evaluate.get_val_transform(
        {"Faster R-CNN": "faster_rcnn", "SSD300": "ssd", "DETR": "detr",
         "CNN baseline": "cnn_baseline"}[etq], img_size,
    )
    nombre_interno = {"Faster R-CNN": "faster_rcnn", "SSD300": "ssd",
                       "DETR": "detr", "CNN baseline": "cnn_baseline"}[etq]

    out = {}
    with torch.inference_mode():
        for stem in SELECCION:
            img = Image.open(IMAGES_VAL / f"{stem}.jpg").convert("RGB")
            target0 = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)}
            img_t, target_t = val_tf(img, target0)

            if nombre_interno in ("detr", "cnn_baseline"):
                imgs_t = img_t.unsqueeze(0).to(DEVICE)
            else:
                imgs_t = [img_t.to(DEVICE)]

            preds_raw = model(imgs_t)
            pred = evaluate.postprocess_predictions(preds_raw, nombre_interno, imgs_t, DEVICE, conf=conf_disp)[0]
            pred = evaluate._to_native(pred, target_t.get("scale"))
            out[stem] = {"boxes": pred["boxes"].numpy(), "labels": pred["labels"].numpy(),
                         "scores": pred["scores"].numpy()}
    del model
    torch.cuda.empty_cache()
    print(f"{etq}: OK ({sum(len(v['labels']) for v in out.values())} cajas en {len(SELECCION)} imagenes, conf>={conf_disp:.3f})")
    return out


def preds_yolo(etq, prefijo, config_yaml, sufijo_pesos):
    from ultralytics import YOLO
    peso = buscar_pesos(prefijo, sufijo_pesos)
    cfg = yaml.safe_load(Path(config_yaml).read_text(encoding="utf-8"))
    img_size = cfg["training"]["img_size"]
    conf_disp = RESUMEN[etq]["conf_max_f1"]["media"]

    model = YOLO(str(peso))
    out = {}
    for stem in SELECCION:
        im = Image.open(IMAGES_VAL / f"{stem}.jpg").convert("RGB")
        r = model.predict(im, imgsz=img_size, conf=conf_disp, iou=0.5,
                          max_det=evaluate.MAX_DET, verbose=False,
                          device=0 if evaluate.DEVICE_TYPE == "cuda" else "cpu")[0]
        b = r.boxes
        out[stem] = {"boxes": b.xyxy.cpu().numpy(), "labels": b.cls.cpu().numpy().astype(int),
                     "scores": b.conf.cpu().numpy()}
    del model
    torch.cuda.empty_cache()
    print(f"{etq}: OK ({sum(len(v['labels']) for v in out.values())} cajas en {len(SELECCION)} imagenes, conf>={conf_disp:.3f})")
    return out


PREDICCIONES = {}
for etq, prefijo, config_yaml, sufijo_pesos, es_yolo in CONFIGS:
    fn = preds_yolo if es_yolo else preds_torchvision
    PREDICCIONES[etq] = fn(etq, prefijo, config_yaml, sufijo_pesos)
"""))

cells.append(md(r"""
## Dibujo — un panel por archivo (GT y cada modelo por separado)

Antes se componían GT + 6 modelos en una sola figura 2×4: en la memoria, cada
panel quedaba a ~25% del ancho de página — ilegible. Ahora cada panel se
guarda como imagen independiente (PNG + PDF), a resolución nativa y sin
encogerse dentro de una cuadrícula, para componer la disposición final con
subfiguras directamente en LaTeX, controlando el tamaño impreso de cada una.

Nomenclatura: `qual_{n}_{stem}__{panel}.pdf`, con `panel` ∈
`gt`, `yolo11m_a`, `yolo11m_b`, `faster_rcnn`, `ssd300`, `detr`,
`cnn_baseline`.
"""))

cells.append(code(r"""
SLUGS = {
    "YOLO11m (A)":  "yolo11m_a",
    "YOLO11m (B)":  "yolo11m_b",
    "Faster R-CNN": "faster_rcnn",
    "SSD300":       "ssd300",
    "DETR":         "detr",
    "CNN baseline": "cnn_baseline",
}


def color_de(label_idx):
    return CLASS_COLORS.get(CLASS_NAMES[label_idx], "#898781")


def dibujar(ax, img, boxes, labels, scores=None, titulo=""):
    ax.imshow(img)
    for i, (box, lbl) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box
        c = color_de(int(lbl))
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        linewidth=1.8, edgecolor=c, facecolor="none"))
        texto = CLASS_NAMES[int(lbl)]
        if scores is not None:
            texto += f" {scores[i]:.2f}"
        ax.text(x1, max(y1 - 3, 0), texto, fontsize=6.5, color="white",
                bbox=dict(facecolor=c, edgecolor="none", pad=1.0, alpha=0.9))
    ax.set_title(titulo, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def guardar_panel(fig, nombre):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{nombre}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{nombre}.pdf", bbox_inches="tight")
    plt.close(fig)


for n, stem in enumerate(SELECCION, start=1):
    img = np.array(Image.open(IMAGES_VAL / f"{stem}.jpg").convert("RGB"))
    gt_boxes, gt_labels = cargar_gt(stem)
    W, H = Image.open(IMAGES_VAL / f"{stem}.jpg").size
    figsize = (6.4, 6.4 * H / W)

    fig, ax = plt.subplots(figsize=figsize)
    dibujar(ax, img, gt_boxes, gt_labels, titulo=f"Ground truth ({len(gt_labels)} inst.)")
    guardar_panel(fig, f"qual_{n:02d}_{stem}__gt")

    for etq, *_ in CONFIGS:
        p = PREDICCIONES[etq][stem]
        fig, ax = plt.subplots(figsize=figsize)
        dibujar(ax, img, p["boxes"], p["labels"], p["scores"],
                titulo=f"{etq} ({len(p['labels'])} det.)")
        guardar_panel(fig, f"qual_{n:02d}_{stem}__{SLUGS[etq]}")

    print(f"{stem}.jpg: 7 paneles guardados")
"""))

cells.append(md(r"""
## Plantilla LaTeX para componer la cuadrícula (ejemplo, 2 columnas)

Cada panel es un PDF independiente en `figuras_cualitativas/`. Ejemplo para
componer GT + 6 modelos en subfiguras de 2 columnas (ajustar `width` según el
número de columnas que se quiera por figura):

```latex
\begin{figure}[htbp]
  \centering
  \begin{subfigure}{0.48\linewidth}
    \includegraphics[width=\linewidth]{figuras_cualitativas/qual_01_<stem>__gt.pdf}
    \caption{Ground truth}
  \end{subfigure}
  \begin{subfigure}{0.48\linewidth}
    \includegraphics[width=\linewidth]{figuras_cualitativas/qual_01_<stem>__yolo11m_a.pdf}
    \caption{YOLO11m (A)}
  \end{subfigure}
  % ... repetir para yolo11m_b, faster_rcnn, ssd300, detr, cnn_baseline
  \caption{Detecciones sobre \texttt{<stem>.jpg}}
  \label{fig:qual-<stem>}
\end{figure}
```
"""))

cells.append(md(r"""
## Resumen de ficheros generados
"""))

cells.append(code(r"""
for p in sorted(FIG_DIR.glob("*.pdf")):
    print(p.name)
"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (tfm)", "language": "python", "name": "tfm"},
        "language_info": {"name": "python", "version": "3.11.15"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path(__file__).parent / "graficas_cualitativas.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Notebook escrito: {out_path} ({len(cells)} celdas)")
