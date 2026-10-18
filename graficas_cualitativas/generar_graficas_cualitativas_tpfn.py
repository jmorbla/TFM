# -*- coding: utf-8 -*-
"""Genera graficas_cualitativas_tpfn.ipynb: variante de graficas_cualitativas.ipynb
que colorea cada caja predicha por su estado frente al GT (TP/FP) y marca los
GT no detectados (FN), en vez de colorear por clase.

Comparte selección de imágenes e inferencia con generar_graficas_cualitativas.py
(mismas 6 imágenes, semilla 0, mismo conf_max_f1 de visualización) -- SOLO
cambia el criterio de color del dibujo y la carpeta de salida, para no
sobreescribir las figuras ya generadas por el otro script.

Útil para responder "¿cómo sé qué modelo es mejor mirando el GT?": con color
por clase hay que comparar caja a caja a ojo; con color por TP/FP/FN se ve
directamente cuántos aciertos, falsas alarmas y objetos perdidos tiene cada
modelo en esa imagen (recuento en el propio título del panel).

Criterio de emparejamiento TP/FP/FN: greedy por score descendente, misma
clase, IoU >= 0.5 -- el mismo que usa `DetectionEvaluator._match()` en
evaluate.py (no se reutiliza esa clase directamente porque está acoplada a
la acumulación global de métricas sobre las 2000 imágenes de val; aquí solo
hace falta el emparejamiento de una imagen suelta).

Uso (desde TFM/, con los 18 runs entrenados y resumen.json generado):
    python graficas_cualitativas/generar_graficas_cualitativas_tpfn.py
    python -m jupyter nbconvert --to notebook --execute --inplace graficas_cualitativas/graficas_cualitativas_tpfn.ipynb
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
# Ejemplos cualitativos de detección — color por TP / FP / FN

Variante de `graficas_cualitativas.ipynb`: en vez de colorear cada caja por su
clase, aquí se colorea por su estado frente al ground truth, para poder
comparar modelos de un vistazo sin tener que emparejar cajas a ojo:

- **Verde** = acierto (TP): predicción que se solapa (IoU ≥ 0.5) con un GT de
  su misma clase.
- **Rojo** = falsa alarma (FP): predicción sin GT correspondiente, o con la
  clase equivocada.
- **Azul discontinuo** = objeto perdido (FN): caja de GT sin ninguna
  predicción encima. Azul en vez de un tono cálido a propósito: con rojo
  (FP) y verde (TP) ya ocupados, un tercer color cálido (ámbar/naranja) se
  confundía con el rojo a tamaño pequeño; el azul es inconfundible con
  ambos y además no sufre la confusión rojo-verde del daltonismo más común.

Cada panel lleva además el recuento TP/FP/FN en el título, así la comparación
entre modelos no depende de contar cajas de colores a ojo.

**No sustituye a las tablas de `graficas_resultados.ipynb`** (mAP, AP por
clase/tamaño, significancia estadística) — es evidencia cualitativa sobre
*estas* 6 imágenes concretas, no una métrica agregada sobre las 2000 de val.

## Decisiones metodológicas (idénticas a `graficas_cualitativas.ipynb`)

- **Mismas 6 imágenes** (selección determinista por nº de instancias de GT:
  mínimo no nulo, percentiles 20/40/60/80, máximo).
- **Mismo umbral de confianza de visualización** (`conf_max_f1` de cada
  modelo, media sobre 3 semillas) y **misma semilla 0**.
- **Emparejamiento TP/FP/FN**: greedy por score descendente, misma clase,
  IoU ≥ 0.5 — el mismo criterio que usa `DetectionEvaluator._match()` en
  `evaluate.py` para las métricas agregadas.
- Salida en `figuras_cualitativas_tpfn/` (carpeta separada de
  `figuras_cualitativas/`, no se sobrescribe nada).
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
from torchvision.ops import box_iou

sys.path.insert(0, "..")  # evaluate.py vive en la raiz de TFM/, no en esta carpeta
import evaluate  # reutiliza load_model, get_val_transform, postprocess_predictions, _to_native

FIG_DIR = Path("figuras_cualitativas_tpfn")
FIG_DIR.mkdir(exist_ok=True)
DEVICE = torch.device(evaluate.DEVICE_TYPE)
N_IMAGENES = 6
IOU_MATCH = 0.5  # mismo umbral que map50 y que P/R/F1 en evaluate.py

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


# Color de ESTADO (TP/FP/FN), no de clase -- mismos tonos que las figuras de
# graficas_resultados.py (STATUS_GOOD/STATUS_CRITICAL) para que el "verde
# bueno / rojo malo" sea consistente en toda la memoria.
STATUS_TP = "#0ca30c"
STATUS_FP = "#d03b3b"
STATUS_FN = "#1f5fd6"
INK_SECONDARY = "#52514e"

print(f"Val: {IMAGES_VAL}  ({sum(1 for _ in LABELS_VAL.glob('*.txt'))} etiquetas)")
"""))

cells.append(md(r"""
## Selección determinista de imágenes (idéntica a graficas_cualitativas.ipynb)
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
## Emparejamiento TP / FP / FN

Greedy por score descendente, misma clase, IoU ≥ 0.5 — mismo criterio que
`DetectionEvaluator._match()` en `evaluate.py` (reimplementado aquí en lugar de
importado porque esa clase está acoplada a la acumulación global de métricas
sobre las 2000 imágenes de val; aquí solo hace falta emparejar una imagen
suelta).
"""))

cells.append(code(r"""
def clasificar_predicciones(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels,
                             iou_thresh=IOU_MATCH):
    # Devuelve (es_tp, gt_matched): es_tp por prediccion, gt_matched por GT.
    n_p, n_g = len(pred_scores), len(gt_labels)
    es_tp = np.zeros(n_p, dtype=bool)
    gt_matched = np.zeros(n_g, dtype=bool)
    if n_p and n_g:
        order = np.argsort(-pred_scores)
        iou = box_iou(torch.as_tensor(pred_boxes[order], dtype=torch.float32),
                       torch.as_tensor(gt_boxes, dtype=torch.float32)).numpy()
        for oi, i in enumerate(order):
            row = iou[oi].copy()
            row[gt_matched] = -1
            row[gt_labels != pred_labels[i]] = -1
            j = int(row.argmax())
            if row[j] >= iou_thresh:
                es_tp[i] = True
                gt_matched[j] = True
    return es_tp, gt_matched
"""))

cells.append(md(r"""
## Dibujo — un panel por archivo, color por TP / FP / FN
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


def dibujar_gt(ax, img, boxes, labels, titulo=""):
    ax.imshow(img)
    for box, lbl in zip(boxes, labels):
        x1, y1, x2, y2 = box
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        linewidth=1.8, edgecolor=INK_SECONDARY, facecolor="none"))
        ax.text(x1, max(y1 - 3, 0), CLASS_NAMES[int(lbl)], fontsize=6.5, color="white",
                bbox=dict(facecolor=INK_SECONDARY, edgecolor="none", pad=1.0, alpha=0.9))
    ax.set_title(titulo, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def dibujar_modelo(ax, img, boxes, labels, scores, gt_boxes, gt_labels, titulo=""):
    ax.imshow(img)
    es_tp, gt_matched = clasificar_predicciones(boxes, labels, scores, gt_boxes, gt_labels)

    for i, (box, lbl, sc) in enumerate(zip(boxes, labels, scores)):
        x1, y1, x2, y2 = box
        c = STATUS_TP if es_tp[i] else STATUS_FP
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        linewidth=1.8, edgecolor=c, facecolor="none"))
        ax.text(x1, max(y1 - 3, 0), f"{CLASS_NAMES[int(lbl)]} {sc:.2f}", fontsize=6.5,
                color="white", bbox=dict(facecolor=c, edgecolor="none", pad=1.0, alpha=0.9))

    for j, (box, lbl) in enumerate(zip(gt_boxes, gt_labels)):
        if gt_matched[j]:
            continue
        x1, y1, x2, y2 = box
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.8,
                                        edgecolor=STATUS_FN, facecolor="none", linestyle="--"))
        ax.text(x1, max(y1 - 3, 0), f"{CLASS_NAMES[int(lbl)]} (FN)", fontsize=6.5,
                color="white", bbox=dict(facecolor=STATUS_FN, edgecolor="none", pad=1.0, alpha=0.9))

    n_tp, n_fp, n_fn = int(es_tp.sum()), int((~es_tp).sum()), int((~gt_matched).sum())
    leyenda = [
        patches.Patch(edgecolor=STATUS_TP, facecolor="none", label=f"TP = {n_tp}"),
        patches.Patch(edgecolor=STATUS_FP, facecolor="none", label=f"FP = {n_fp}"),
        patches.Patch(edgecolor=STATUS_FN, facecolor="none", linestyle="--", label=f"FN = {n_fn}"),
    ]
    ax.legend(handles=leyenda, loc="lower right", fontsize=7, frameon=True,
              facecolor="white", framealpha=0.85, edgecolor="none")
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
    dibujar_gt(ax, img, gt_boxes, gt_labels, titulo=f"Ground truth ({len(gt_labels)} inst.)")
    guardar_panel(fig, f"qual_{n:02d}_{stem}__gt")

    for etq, *_ in CONFIGS:
        p = PREDICCIONES[etq][stem]
        fig, ax = plt.subplots(figsize=figsize)
        dibujar_modelo(ax, img, p["boxes"], p["labels"], p["scores"], gt_boxes, gt_labels,
                       titulo=etq)
        guardar_panel(fig, f"qual_{n:02d}_{stem}__{SLUGS[etq]}")

    print(f"{stem}.jpg: 7 paneles guardados")
"""))

cells.append(md(r"""
## Resumen de ficheros generados

Todas las figuras quedan en `figuras_cualitativas_tpfn/` — carpeta separada de
`figuras_cualitativas/`, así que las imágenes coloreadas por clase de la otra
versión no se pierden.
"""))

cells.append(code(r"""
for p in sorted(FIG_DIR.glob("*.pdf")):
    print(p.name)
"""))

cells.append(md(r"""
## Cómo leer estas figuras

- **El recuento TP/FP/FN del título/leyenda es el dato objetivo** — no hace
  falta contar cajas a ojo para saber si un modelo acierta más que otro en
  esa imagen concreta.
- Un modelo con **muchos FN (azul discontinuo)** está perdiendo objetos —
  coherente con recall bajo en las tablas agregadas.
- Un modelo con **muchos FP (rojo)** está alucinando detecciones o
  confundiendo clases — coherente con precisión baja.
- Esto sigue siendo evidencia sobre 6 imágenes, no una métrica: para afirmar
  que un modelo es mejor que otro en general hay que apoyarse en el mAP, el
  AP por clase/tamaño y la significancia estadística de
  `graficas_resultados.ipynb` / `significancia_estadistica/` — aquí solo se
  ve *cómo* se manifiesta esa diferencia en casos concretos.
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

out_path = Path(__file__).parent / "graficas_cualitativas_tpfn.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Notebook escrito: {out_path} ({len(cells)} celdas)")
