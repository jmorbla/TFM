"""
Evaluate any trained model on the BDD100K validation split.
Reports mAP@50, mAP@50-95 (global y por clase) y FPS.

Uso (desde tfm-comparativa/):

    # YOLO11m
    python evaluate.py --model yolo11 \
        --weights runs/detect/runs/yolo11/yolo11m_20260805_210422/weights/best.pt

    # Faster R-CNN
    python evaluate.py --model faster_rcnn \
        --weights runs/faster_rcnn/faster_rcnn_20260814_111431/best.pt \
        --config configs/faster_rcnn.yaml

    # SSD
    python evaluate.py --model ssd \
        --weights runs/ssd/<run>/best.pt --config configs/ssd.yaml

    # DETR
    python evaluate.py --model detr \
        --weights runs/detr/<run>/best.pt --config configs/detr.yaml

    # CNN Baseline
    python evaluate.py --model cnn_baseline \
        --weights runs/cnn_baseline/<run>/best.pt --config configs/cnn_baseline.yaml

    # CNN+GRU
    python evaluate.py --model cnn_gru \
        --weights runs/cnn_gru/<run>/best.pt --config configs/cnn_gru.yaml

Nota FPS: se mide con batch_size=1 para reflejar la latencia real de
inferencia imagen a imagen. Pasar --batch_size N mide throughput de batch.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from data.dataset import BDD100KDetection, BDD100KSequence
from data.transforms import build_transforms

# ── Constantes ────────────────────────────────────────────────────────────────

# ── Consola: forzar UTF-8 ────────────────────────────────────────────────────
# Este script imprime caracteres de caja y flechas. Cuando lo lanza una persona
# desde una terminal moderna funciona, pero cuando lo lanza otro script por
# subprocess la salida es una tubería y Python cae al encoding del sistema
# (cp1252 en Windows), que no los admite: UnicodeEncodeError DESPUES de haber
# hecho todo el trabajo. Se fuerza UTF-8 y, si el terminal no lo admite, se
# degrada a reemplazar el carácter en vez de abortar.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CLASS_NAMES = [
    "person", "rider", "car", "truck", "bus",
    "train", "motor", "bike", "traffic light", "traffic sign",
]

# ── Sobre el desbalanceo de clases ───────────────────────────────────────────
# BDD100K está fuertemente desbalanceado: 'car' es el 55 % de las instancias y
# 'train' aparece en solo 14 de las 10 000 imágenes de validación (15 instancias
# en total). Es un límite del dataset, no del muestreo: ninguna semilla ni
# estrategia de muestreo lo corrige (ver notebooks/01_desbalanceo_clases.ipynb).
#
# NO se excluye ninguna clase de la métrica. El mAP se calcula sobre las 10,
# que es el estándar de BDD100K y lo que hace comparable el trabajo con la
# literatura. En su lugar, se reporta el número de instancias de GT junto al AP
# de cada clase, para que la fiabilidad de cada cifra sea evidente sin necesidad
# de descartar nada.
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
YOLO_DATA_YAML = "bdd100k_yolo.yaml"   # YAML en formato Ultralytics

# ── Protocolo de evaluación común a TODOS los modelos ────────────────────────
# El mAP integra la curva precisión-recall completa. Si cada arquitectura filtra
# sus predicciones a un umbral distinto, la curva queda truncada en un punto
# distinto y los mAP dejan de ser comparables entre sí. Antes convivían cuatro
# umbrales (DETR 0.5, CNN 0.3, Faster R-CNN 0.05, YOLO 0.001); ahora uno solo.
DEFAULT_CONF = 0.001      # umbral de confianza único (convención COCO/Ultralytics)
# Detecciones máximas por imagen. 100 es la convención COCO y además el tope que
# torchmetrics aplica por defecto: si aquí se pusiera 300, torchmetrics seguiría
# recortando a 100 los modelos torch mientras YOLO sí usaría 300, reintroduciendo
# justo la asimetría que este protocolo elimina. Se pasa explícito a ambos.
MAX_DET = 100
MAX_DET_THRESHOLDS = [1, 10, MAX_DET]   # para MeanAveragePrecision
PR_SCORE_THRESH = 0.5     # umbral de operación para Precision/Recall puntuales
PR_IOU_THRESH = 0.5
WARMUP_BATCHES = 10       # descartados del cronómetro (cuDNN autotune, init CUDA)
_TOPK_PRE_NMS = 1000      # candidatos que entran al NMS en los modelos CNN propios


# ── Carga de modelos ──────────────────────────────────────────────────────────

def load_model(model_name: str, cfg: dict, weights_path: str, nc: int, device,
               conf: float = DEFAULT_CONF):
    """Instancia el modelo y carga los pesos del checkpoint.

    `conf` se inyecta en los detectores de torchvision, que filtran internamente
    antes de devolver las detecciones: sin esto, Faster R-CNN cortaría a 0.05 y
    SSD a 0.01 mientras el resto de modelos usa otro umbral distinto.
    """
    if model_name == "yolo11":
        from ultralytics import YOLO
        return YOLO(weights_path)   # Ultralytics gestiona el device internamente

    if model_name == "faster_rcnn":
        from models.faster_rcnn.model import build_model
        model = build_model(nc=nc, pretrained=False)
        model.roi_heads.score_thresh = conf
        model.roi_heads.detections_per_img = MAX_DET
    elif model_name == "ssd":
        from models.ssd.model import build_model
        model = build_model(nc=nc, pretrained=False)
        model.score_thresh = conf
        model.detections_per_img = MAX_DET
    elif model_name == "detr":
        from models.detr.model import DETRWrapper
        model = DETRWrapper(nc=nc, pretrained=False)
    elif model_name == "cnn_baseline":
        from models.cnn_baseline.model import build_model
        model = build_model(nc=nc, pretrained=False)
    elif model_name == "cnn_gru":
        from models.cnn_gru.model import build_model
        seq = cfg.get("sequence", {})
        gru = cfg.get("gru", {})
        model = build_model(
            nc=nc,
            T=seq.get("T", 4),
            gru_hidden=gru.get("hidden_size", 256),
            pretrained=False,
        )
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")

    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    return model.to(device).eval()


# ── Transforms por modelo (igual que en train.py) ─────────────────────────────

def get_val_transform(model_name: str, img_size: int):
    """
    Devuelve el transform de validación correcto para cada arquitectura.
    Faster R-CNN y SSD normalizan internamente → normalize=False.
    """
    if model_name == "faster_rcnn":
        return build_transforms(None, train=False, normalize=False)
    elif model_name == "ssd":
        return build_transforms(300, train=False, square=True, normalize=False)
    elif model_name == "detr":
        return build_transforms(img_size, train=False, max_size=1333)
    else:
        return build_transforms(img_size, train=False)


# ── Evaluación ────────────────────────────────────────────────────────────────

def collate_fn(batch):
    return tuple(zip(*batch))


def _smooth(y, f: float = 0.1):
    """Filtro de caja, réplica de ultralytics/utils/metrics.py::smooth."""
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


class DetectionEvaluator:
    """
    Acumulador único de métricas para los 5 modelos.

    Todo se acumula en coordenadas NATIVAS (1280×720), no en las de la entrada
    redimensionada de cada modelo. Es imprescindible por dos motivos:
      1. El desglose de AP por tamaño de objeto usa umbrales absolutos de área
         (COCO: 32² y 96² px). Sin normalizar, "objeto pequeño" significaría
         cosas distintas en SSD (300 px) y en DETR (800 px).
      2. El resize de SSD es anisótropo (300×300 sobre una imagen 16:9), lo que
         distorsiona el IoU respecto a los demás modelos.

    Precision, Recall y F1 se calculan en el punto de MÁXIMO F1 de la curva,
    que es el punto de operación que Ultralytics reporta para YOLO
    (ultralytics/utils/metrics.py:839). Antes se usaba un umbral fijo de 0.5
    para los modelos torch mientras YOLO usaba el máximo-F1: las columnas de
    P y R de la tabla no eran comparables entre sí.
    """

    N_CONF_POINTS = 1000   # rejilla de confianza, igual que Ultralytics

    def __init__(self, nc: int = 10, iou_thresh: float = PR_IOU_THRESH):
        self.nc = nc
        self.iou_thresh = iou_thresh
        self.metric = MeanAveragePrecision(
            class_metrics=True, iou_type="bbox",
            max_detection_thresholds=MAX_DET_THRESHOLDS,
            # extended_summary expone los arrays crudos precision/recall/classes
            # que pycocotools calcula internamente para llegar al mAP agregado.
            # No cambia ninguna cifra ya existente (map, map_50, map_per_class...,
            # verificado): solo añade lo necesario para el AP por umbral de IoU
            # y la curva PR completa por clase (ver _extended()).
            extended_summary=True,
        )
        self._conf, self._tp, self._cls = [], [], []
        self._gt_per_class = np.zeros(nc, dtype=np.int64)
        self.seen_classes: set = set()

    # ── acumulación ──────────────────────────────────────────────────────────
    def update(self, preds: list, targets: list):
        """preds y targets ya en coordenadas nativas, en CPU."""
        self.metric.update(preds, targets)
        for p, t in zip(preds, targets):
            self.seen_classes.update(p["labels"].tolist())
            self.seen_classes.update(t["labels"].tolist())
            for c in t["labels"].tolist():
                self._gt_per_class[c] += 1
            self._match(p, t)

    def _match(self, p, t):
        """Emparejamiento greedy por score descendente, misma clase, IoU >= umbral."""
        from torchvision.ops import box_iou as tv_box_iou
        scores, labels_p, boxes_p = p["scores"], p["labels"], p["boxes"]
        if len(scores) == 0:
            return
        order = scores.argsort(descending=True)
        scores, labels_p, boxes_p = scores[order], labels_p[order], boxes_p[order]
        boxes_g, labels_g = t["boxes"], t["labels"]

        tp = np.zeros(len(scores), dtype=np.float32)
        if len(boxes_g):
            iou = tv_box_iou(boxes_p, boxes_g)
            matched = torch.zeros(len(boxes_g), dtype=torch.bool)
            for i in range(len(scores)):
                iou_i = iou[i].clone()
                iou_i[matched] = -1                        # GT ya emparejado
                iou_i[labels_g != labels_p[i]] = -1        # exigir misma clase
                best_iou, best_j = iou_i.max(dim=0)
                if best_iou.item() >= self.iou_thresh:
                    tp[i] = 1.0
                    matched[best_j] = True

        self._conf.append(scores.numpy())
        self._tp.append(tp)
        self._cls.append(labels_p.numpy())

    # ── cálculo ──────────────────────────────────────────────────────────────
    def compute(self) -> dict:
        res = self.metric.compute()

        # mapa clase → AP@50-95 (torchmetrics indexa map_per_class por el orden
        # de las clases únicas vistas, no por la etiqueta numérica)
        sorted_seen = sorted(self.seen_classes)
        per_class = res.get("map_per_class")
        class_ap = {}
        if per_class is not None:
            for j, c in enumerate(sorted_seen):
                if j < len(per_class):
                    class_ap[int(c)] = per_class[j].item()

        p, r, f1, conf = self._pr_at_max_f1()
        ap_por_iou, pr_curve = self._extended(res)

        return {
            "map50":      res["map_50"].item(),
            "map50_95":   res["map"].item(),
            "map_small":  res["map_small"].item(),
            "map_medium": res["map_medium"].item(),
            "map_large":  res["map_large"].item(),
            "precision": p, "recall": r, "f1": f1, "conf_max_f1": conf,
            "class_ap": class_ap,
            # Instancias de GT realmente evaluadas por clase. Se reporta junto
            # al AP para que se vea de un vistazo qué cifras son fiables y
            # cuáles descansan en un puñado de objetos.
            "gt_per_class": {int(i): int(n) for i, n in enumerate(self._gt_per_class)},
            # AP por umbral de IoU (media sobre clases con GT) y curva PR
            # completa por clase a IoU=0.5 -- ver _extended().
            "ap_por_iou": ap_por_iou,
            "iou_thresholds": [round(float(t), 3) for t in self.metric.iou_thresholds],
            "pr_curve": pr_curve,
        }

    def _extended(self, res) -> tuple:
        """AP por umbral de IoU y curva PR completa por clase, a partir de
        extended_summary=True (arrays crudos de pycocotools).

        `res["precision"]` tiene forma (T, R, K, A, M): T=10 umbrales de IoU
        (0.50:0.05:0.95), R=101 puntos de recall (rejilla estándar COCO
        0..1), K=clases vistas, A=4 rangos de área (se usa 0="all"),
        M=3 topes de detecciones por imagen (se usa el último = 100, igual
        que en el resto del protocolo). -1 marca "clase sin GT en esa celda"
        (convención COCO) y se enmascara antes de promediar -- promediar el
        -1 tal cual, como con el centinela de evaluate.py, destrozaría la
        media.

        La curva PR usa el mismo IoU=0.5 que ya reporta `map50`, pero NO es
        la misma curva que `_pr_at_max_f1()`: esta es el cálculo estándar de
        pycocotools (interpolación a 101 puntos de recall fijos); la de
        `_pr_at_max_f1` es una réplica del criterio de Ultralytics para el
        punto de máximo F1. Coexisten con propósitos distintos.
        """
        prec = res.get("precision")
        classes = res.get("classes")
        if prec is None or classes is None:
            return None, None
        prec = prec.numpy()

        ap_por_iou = []
        for t in range(prec.shape[0]):
            col_aps = []
            for k in range(prec.shape[2]):
                col = prec[t, :, k, 0, -1]
                valid = col[col > -1]
                if len(valid):
                    col_aps.append(float(valid.mean()))
            ap_por_iou.append(float(np.mean(col_aps)) if col_aps else None)

        pr_curve = {}
        for k, cls_label in enumerate(classes.tolist()):
            col = prec[0, :, k, 0, -1]
            pr_curve[int(cls_label)] = [None if v <= -1 else round(float(v), 4) for v in col]

        return ap_por_iou, pr_curve

    def _pr_at_max_f1(self) -> tuple:
        """Curvas P-R por clase interpoladas sobre una rejilla de confianza.

        NOTA sobre el F1 devuelto: es la MEDIA de los F1 por clase, no el F1
        calculado a partir de la P y la R medias. Por eso puede quedar por
        debajo de ambas y no es un error — es exactamente lo que hace
        Ultralytics (`Metric.mf1` devuelve `self.f1.mean()`), y mantenerlo
        idéntico es lo que permite comparar estas cifras con las suyas.
        """
        if not self._conf:
            return 0.0, 0.0, 0.0, 0.0
        conf = np.concatenate(self._conf)
        tp   = np.concatenate(self._tp)
        cls  = np.concatenate(self._cls)

        order = (-conf).argsort()
        conf, tp, cls = conf[order], tp[order], cls[order]

        x = np.linspace(0, 1, self.N_CONF_POINTS)
        p_curve, r_curve = [], []
        for c in np.unique(cls):
            n_gt = self._gt_per_class[int(c)]
            if n_gt == 0:
                continue                      # clase sin GT: no aporta a P/R
            m = cls == c
            tpc = tp[m].cumsum()
            fpc = (1 - tp[m]).cumsum()
            recall    = tpc / (n_gt + 1e-16)
            precision = tpc / (tpc + fpc + 1e-16)
            # -x / -conf porque conf viene en orden descendente
            r_curve.append(np.interp(-x, -conf[m], recall,    left=0))
            p_curve.append(np.interp(-x, -conf[m], precision, left=1))

        if not p_curve:
            return 0.0, 0.0, 0.0, 0.0
        p_curve = np.stack(p_curve)
        r_curve = np.stack(r_curve)
        f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + 1e-16)
        i = _smooth(f1_curve.mean(0), 0.1).argmax()      # punto de máximo F1
        return (float(p_curve[:, i].mean()), float(r_curve[:, i].mean()),
                float(f1_curve[:, i].mean()), float(x[i]))


def postprocess_predictions(preds_raw, model_name: str, imgs_t, device,
                            conf: float = DEFAULT_CONF):
    """Normaliza la salida cruda de cada arquitectura al formato de torchmetrics.

    Se ejecuta DENTRO de la región cronometrada porque el postproceso
    (decodificación de anchors, NMS) es parte del coste real de inferencia y
    debe contarse igual en todos los modelos.

    train.py la reutiliza para su validación por época: así el criterio con el
    que se elige best.pt y el que produce las métricas finales no pueden
    divergir.
    """
    if model_name in ("faster_rcnn", "ssd"):
        # torchvision devuelve labels 1-indexed; 0 = background
        out = []
        for p in preds_raw:
            keep = p["labels"] > 0          # descarta fondo
            out.append({
                "boxes":  p["boxes"][keep].cpu(),
                "scores": p["scores"][keep].cpu(),
                "labels": (p["labels"][keep] - 1).cpu(),  # → 0-indexed
            })
        return out
    if model_name == "detr":
        return _detr_postprocess(preds_raw, imgs_t, conf=conf)
    if model_name in ("cnn_baseline", "cnn_gru"):
        cls_preds, reg_preds = preds_raw
        # imgs_t: (B,3,H,W) para baseline; (B,T,3,H,W) para gru → shape[-2:] = (H,W)
        return _cnn_postprocess(
            cls_preds, reg_preds, imgs_t.shape[-2:], device, score_thresh=conf
        )
    raise ValueError(f"Modelo no soportado en evaluate_torchvision: {model_name}")


@torch.inference_mode()
def evaluate_torchvision(model, loader, device, model_name: str, nc: int = 10,
                         conf: float = DEFAULT_CONF):
    """
    Pasa el loader por el modelo, acumula predicciones y calcula métricas
    con torchmetrics MeanAveragePrecision.

    Faster R-CNN / SSD devuelven labels 1-indexed (0 = background).
    Restamos 1 y filtramos fondo para alinearlos con los 0-indexed del GT.

    Además de mAP, calcula Precision y Recall globales a umbral fijo
    (IoU ≥ 0.5, score ≥ 0.5) mediante PRAccumulator para que sean
    comparables con los valores que reporta Ultralytics para YOLO11m.

    nc : número de clases (sin contar background).  Se usa para construir
         el mapa correcto clase→AP tras metric.compute().

    Protocolo de medición de FPS
    ----------------------------
    El cronómetro cubre forward + postproceso, con torch.cuda.synchronize() a
    ambos lados y descartando WARMUP_BATCHES iteraciones iniciales. Antes se
    medía solo el forward, lo que favorecía a cnn_baseline/cnn_gru: su NMS y su
    decodificación de anchors quedaban fuera del reloj mientras que el NMS
    interno de Faster R-CNN y SSD sí se contabilizaba.

    NOTA sobre torchmetrics:
        map_per_class NO está indexado por label numérico (0..nc-1) sino
        por posición en el conjunto ordenado de clases únicas vistas
        (predicciones + GT).  Rastreamos seen_classes durante el bucle
        para reconstruir el mapeo correcto.
    """
    ev = DetectionEvaluator(nc=nc)
    total_time = 0.0
    n_images = 0
    on_cuda = DEVICE_TYPE == "cuda"

    for i, (imgs, targets) in enumerate(tqdm(loader, desc="Evaluando")):
        if model_name in ("cnn_gru", "detr", "cnn_baseline"):
            # Estos modelos esperan un tensor (B, 3, H, W) en lugar de lista
            imgs_t = torch.stack([img.to(device) for img in imgs])
        else:
            imgs_t = [img.to(device) for img in imgs]

        # Las iteraciones de warmup siguen contando para el mAP; solo se
        # excluyen del cronómetro.
        timed = i >= WARMUP_BATCHES
        if on_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with autocast(device_type=DEVICE_TYPE):
            preds_raw = model(imgs_t)
        preds_tm = postprocess_predictions(preds_raw, model_name, imgs_t, device, conf)

        if on_cuda:
            torch.cuda.synchronize()
        if timed:
            total_time += time.perf_counter() - t0
            n_images   += len(imgs)

        targets_tm = [
            {"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()}
            for t in targets
        ]

        # Volver a coordenadas NATIVAS antes de acumular: los umbrales de área
        # del AP por tamaño son absolutos, así que hay que deshacer el resize
        # propio de cada arquitectura para que signifiquen lo mismo en todas.
        scales = [t.get("scale") for t in targets]
        preds_tm   = [_to_native(p, s) for p, s in zip(preds_tm, scales)]
        targets_tm = [_to_native(t, s) for t, s in zip(targets_tm, scales)]

        ev.update(preds_tm, targets_tm)

    if n_images == 0:
        raise RuntimeError(
            f"Val set demasiado pequeño: no quedan batches tras descartar "
            f"{WARMUP_BATCHES} de warmup. Reduce WARMUP_BATCHES o usa más imágenes."
        )
    out = ev.compute()
    out["fps"] = n_images / total_time
    return out


def _to_native(d: dict, scale) -> dict:
    """Devuelve las cajas a píxeles de la imagen original (1280×720).

    `scale` lo escribe data/transforms.py::Resize. Si el modelo no redimensiona
    (Faster R-CNN trabaja a resolución nativa), no hay nada que deshacer.
    """
    if scale is None:
        return d
    sx, sy = float(scale[0]), float(scale[1])
    if sx == 1.0 and sy == 1.0:
        return d
    b = d["boxes"].clone()
    if len(b):
        b[:, [0, 2]] /= sx
        b[:, [1, 3]] /= sy
    return {**d, "boxes": b}


_DETR_PROCESSOR = None


def _get_detr_processor():
    """DetrImageProcessor cacheado.

    Antes se reconstruía con from_pretrained() en CADA batch: además de ser
    lento, contaminaba la medición de tiempos de DETR con E/S de disco.
    """
    global _DETR_PROCESSOR
    if _DETR_PROCESSOR is None:
        from transformers import DetrImageProcessor
        _DETR_PROCESSOR = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    return _DETR_PROCESSOR


def _detr_postprocess(outputs, imgs, conf: float = DEFAULT_CONF):
    """Convierte logits de HuggingFace DETR → list of {boxes, scores, labels}.

    Con un head custom (num_labels=nc, ignore_mismatched_sizes=True) el head
    tiene shape (nc+1,): clases [0…nc-1] + no-object al final.
    post_process_object_detection hace prob[..., :-1].max(-1), así que los
    labels devueltos ya son 0-indexed [0, nc-1] — mismo rango que el GT.
    NO se aplica ningún desplazamiento.
    """
    processor = _get_detr_processor()
    target_sizes = [img.shape[-2:] for img in imgs]
    hf_preds = processor.post_process_object_detection(
        outputs, threshold=conf, target_sizes=target_sizes
    )
    return [
        {
            "boxes":  p["boxes"].cpu(),
            "scores": p["scores"].cpu(),
            "labels": p["labels"].cpu(),   # ya 0-indexed; sin desplazamiento
        }
        for p in hf_preds
    ]


def _cnn_postprocess(
    cls_preds: list,
    reg_preds: list,
    img_hw,
    device,
    score_thresh: float = DEFAULT_CONF,
    nms_iou: float = 0.5,
    nc: int = 10,
) -> list:
    """
    Decodifica las salidas brutas de CNNBaseline / CNNGRUDetector.

    cls_preds : list de tensors (B, A*nc, fH, fW) — uno por nivel FPN
    reg_preds : list de tensors (B, A*4,  fH, fW) — uno por nivel FPN
    img_hw    : (H, W) de la imagen de entrada (para recortar cajas)

    Pipeline:
        1. Genera anchors idénticos a los usados en training.
        2. Aplana y decodifica deltas de regresión → coordenadas absolutas.
        3. Sigmoid en logits de clasificación → score por clase.
        4. Threshold + NMS por imagen.
    """
    from torchvision.ops import batched_nms as tv_batched_nms
    from models.cnn_baseline.loss import NUM_ANCHORS, generate_anchors

    A  = NUM_ANCHORS   # escalas × aspect ratios (ver loss.py)
    B  = cls_preds[0].shape[0]
    H, W = int(img_hw[-2]), int(img_hw[-1])

    # ── Anchors (mismo esquema que en training) ───────────────────────────────
    feature_shapes = [(int(p.shape[2]), int(p.shape[3])) for p in cls_preds]
    anchors = generate_anchors(feature_shapes, device)   # (N_total, 4) xyxy

    # ── Aplanar predicciones → (B, N_total, nc) y (B, N_total, 4) ────────────
    cls_flat, reg_flat = [], []
    for cp, rp in zip(cls_preds, reg_preds):
        B_, _, fH, fW = cp.shape
        cls_flat.append(
            cp.reshape(B_, A, nc, fH, fW)
              .permute(0, 3, 4, 1, 2)           # (B, fH, fW, A, nc)
              .reshape(B_, fH * fW * A, nc)
        )
        reg_flat.append(
            rp.reshape(B_, A, 4, fH, fW)
              .permute(0, 3, 4, 1, 2)           # (B, fH, fW, A, 4)
              .reshape(B_, fH * fW * A, 4)
        )
    cls_all = torch.cat(cls_flat, dim=1).float()   # (B, N, nc)
    reg_all = torch.cat(reg_flat, dim=1).float()   # (B, N, 4)

    # ── Decode cajas (inverso de encode_boxes en loss.py) ────────────────────
    aw  = (anchors[:, 2] - anchors[:, 0]).clamp(min=1e-6)   # (N,)
    ah  = (anchors[:, 3] - anchors[:, 1]).clamp(min=1e-6)
    acx = anchors[:, 0] + 0.5 * aw
    acy = anchors[:, 1] + 0.5 * ah

    gcx = reg_all[..., 0] * aw + acx   # (B, N)
    gcy = reg_all[..., 1] * ah + acy
    gw  = reg_all[..., 2].exp() * aw
    gh  = reg_all[..., 3].exp() * ah

    x1 = (gcx - 0.5 * gw).clamp(0, W)
    y1 = (gcy - 0.5 * gh).clamp(0, H)
    x2 = (gcx + 0.5 * gw).clamp(0, W)
    y2 = (gcy + 0.5 * gh).clamp(0, H)
    boxes_all = torch.stack([x1, y1, x2, y2], dim=-1)   # (B, N, 4)

    # ── Por imagen: threshold + NMS por clase ─────────────────────────────────
    results = []
    for b in range(B):
        scores_b, labels_b = torch.sigmoid(cls_all[b]).max(dim=1)   # (N,)
        boxes_b = boxes_all[b]   # (N, 4)

        keep = scores_b >= score_thresh
        scores_b = scores_b[keep];  labels_b = labels_b[keep];  boxes_b = boxes_b[keep]

        # Con score_thresh=0.001 sobreviven decenas de miles de anchors; recortar
        # a los mejores antes del NMS mantiene el coste acotado sin alterar el mAP.
        if scores_b.numel() > _TOPK_PRE_NMS:
            top = scores_b.topk(_TOPK_PRE_NMS).indices
            scores_b = scores_b[top]; labels_b = labels_b[top]; boxes_b = boxes_b[top]

        if scores_b.numel() > 0:
            # batched_nms aplica el NMS de forma independiente por clase.
            # Con nms() a secas, un coche podía suprimir un semáforo solapado.
            nms_keep = tv_batched_nms(boxes_b, scores_b, labels_b, nms_iou)[:MAX_DET]
            scores_b = scores_b[nms_keep]
            labels_b = labels_b[nms_keep]
            boxes_b  = boxes_b[nms_keep]

        results.append({
            "boxes":  boxes_b.cpu(),
            "scores": scores_b.cpu(),
            "labels": labels_b.cpu(),
        })
    return results


# ── Evaluación de YOLO por el MISMO camino que el resto ──────────────────────

def evaluate_yolo(weights: str, images_dir: str, labels_dir: str,
                  nc: int, conf: float, imgsz: int) -> dict:
    """Evalúa YOLO con el acumulador común en lugar de con model.val().

    Motivo: `model.val()` de Ultralytics reporta P y R en el punto de máximo F1
    y con su propio perfilador de tiempos, mientras que el resto de modelos
    pasaba por nuestro bucle. Aunque el criterio de P/R ya se ha unificado,
    seguían siendo dos implementaciones distintas de la misma métrica. Aquí YOLO
    predice con su propio preprocesado (letterbox, que es lo correcto para él)
    pero las métricas las calcula exactamente el mismo código que las de los demás.

    Las cajas que devuelve Ultralytics ya vienen en coordenadas nativas, que es
    justo el espacio en el que acumula DetectionEvaluator.
    """
    from PIL import Image
    from ultralytics import YOLO

    model = YOLO(weights)
    ev = DetectionEvaluator(nc=nc)
    img_dir, lbl_dir = Path(images_dir), Path(labels_dir)
    stems = sorted(p.stem for p in lbl_dir.glob("*.txt")
                   if (img_dir / f"{p.stem}.jpg").exists())

    total_ms, n_timed = 0.0, 0
    for i, stem in enumerate(tqdm(stems, desc="Evaluando (yolo11)")):
        im = Image.open(img_dir / f"{stem}.jpg").convert("RGB")
        W, H = im.size
        r = model.predict(im, imgsz=imgsz, conf=conf, iou=0.5,
                          max_det=MAX_DET, verbose=False, device=0
                          if DEVICE_TYPE == "cuda" else "cpu")[0]

        # Mismo criterio que los modelos torch: inferencia + postproceso.
        # 'preprocess' se excluye en todos (allí equivale a las transforms del
        # DataLoader, que tampoco se cronometran).
        if i >= WARMUP_BATCHES:
            total_ms += r.speed.get("inference", 0.0) + r.speed.get("postprocess", 0.0)
            n_timed += 1

        b = r.boxes
        preds = [{
            "boxes":  b.xyxy.cpu().float(),
            "scores": b.conf.cpu().float(),
            "labels": b.cls.cpu().long(),
        }]
        gt_b, gt_l = [], []
        for line in (lbl_dir / f"{stem}.txt").read_text().splitlines():
            q = line.split()
            if len(q) != 5:
                continue
            c, cx, cy, w, h = int(q[0]), *map(float, q[1:])
            gt_b.append([(cx - w / 2) * W, (cy - h / 2) * H,
                         (cx + w / 2) * W, (cy + h / 2) * H])
            gt_l.append(c)
        targets = [{
            "boxes":  torch.tensor(gt_b, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(gt_l, dtype=torch.int64),
        }]
        ev.update(preds, targets)

    out = ev.compute()
    out["fps"] = 1000.0 / (total_ms / n_timed) if n_timed and total_ms else float("nan")
    return out



# ── Complejidad del modelo: GFLOPs, parámetros y resolución real ─────────────
# El FPS por sí solo NO separa arquitectura de ingeniería. YOLO viene de
# Ultralytics (convoluciones fusionadas, NMS optimizado), Faster R-CNN y SSD de
# las implementaciones de referencia de torchvision, y DETR de HuggingFace: la
# calidad de implementación entra en el cronómetro. Y cada modelo corre a su
# resolución nativa, con ~11x de diferencia en píxeles entre SSD300 (300x300) y
# Faster R-CNN (1333x750).
#
# Los GFLOPs son la medida de coste independiente de la implementación. Se
# reportan SIEMPRE junto al FPS y a la resolución: sin las tres cifras, la
# comparación de velocidad no es interpretable.
#
# Convención: GFLOPs = MACs x 2, la misma que usa Ultralytics. Validado contra
# el valor publicado de YOLO11m (68.0 GFLOPs a 640 px).

def eval_input_shape(model_name: str, img_size: int) -> tuple:
    """(H, W) que ve realmente el modelo, partiendo de los 1280x720 nativos.

    Replica lo que hace get_val_transform() -- y, en Faster R-CNN, lo que hace
    GeneralizedRCNNTransform internamente, que es donde se decide de verdad.
    """
    W0, H0 = 1280, 720
    if model_name == "ssd":
        return 300, 300                      # square=True: deforma a 300x300
    if model_name in ("faster_rcnn", "detr"):
        short = 800 if model_name == "faster_rcnn" else img_size
        s = short / min(H0, W0)
        if s * max(H0, W0) > 1333:           # tope del lado largo
            s = 1333 / max(H0, W0)
        return int(H0 * s), int(W0 * s)
    if model_name == "yolo11":
        # OJO: en Ultralytics `imgsz` fija el lado LARGO, no el corto. Sobre
        # 1280x720 la escala es 640/1280 = 0.5 y el resultado se rellena hasta
        # el multiplo de stride 32: 640x384. No es lo mismo que el img_size de
        # cnn_baseline, que fija el lado CORTO (escala 0.889).
        sc = img_size / max(H0, W0)
        h, w = int(round(H0 * sc)), int(round(W0 * sc))
        return ((h + 31) // 32) * 32, ((w + 31) // 32) * 32
    s = img_size / min(H0, W0)               # cnn_baseline: lado corto = img_size
    return int(H0 * s), int(W0 * s)


def model_complexity(model, model_name: str, img_size: int, device=None) -> dict:
    """GFLOPs, millones de parámetros y resolución de entrada real.

    Nunca aborta la evaluación: si el perfilado falla, devuelve None en la
    métrica que no se pudo calcular y el resto del informe sigue igual.
    """
    import copy

    H, W = eval_input_shape(model_name, img_size)
    out = {"input_h": H, "input_w": W, "params_m": None, "gflops": None}
    try:
        out["params_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
    except Exception:
        pass
    try:
        import thop
        net = copy.deepcopy(model).float().eval().cpu()
        x = torch.zeros(1, 3, H, W)
        # Los detectores de torchvision reciben una LISTA de tensores 3D.
        inputs = ([x[0]],) if model_name in ("faster_rcnn", "ssd") else (x,)
        macs, _ = thop.profile(net, inputs=inputs, verbose=False)
        out["gflops"] = round(macs * 2 / 1e9, 1)
    except Exception as e:
        print(f"  [aviso] GFLOPs no calculables ({type(e).__name__}: {e})")
    return out

# ── Impresión de resultados ───────────────────────────────────────────────────

def print_results(res: dict, class_names: list):
    pct = lambda v: "  n/a " if v != v else f"{v * 100:5.2f}%"   # NaN-safe
    print("\n" + "=" * 66)
    print(f"  mAP@50:        {pct(res['map50'])}   (10 clases, estándar BDD100K)")
    print(f"  mAP@50-95:     {pct(res['map50_95'])}   (10 clases, estándar BDD100K)")
    print("-" * 66)
    print(f"  Precisión:     {pct(res['precision'])}   ┐")
    print(f"  Recall:        {pct(res['recall'])}   ├ en el punto de máximo F1")
    print(f"  F1:            {pct(res['f1'])}   │  (conf = {res['conf_max_f1']:.3f})")
    print(f"  IoU del match: {PR_IOU_THRESH}      ┘")
    print("-" * 66)
    print(f"  FPS:           {res['fps']:.1f}   (batch=1, forward+postproceso)")
    _res = (f"{res['input_w']}x{res['input_h']}"
            if res.get("input_w") else "n/d")
    _gf  = f"{res['gflops']:.1f}" if res.get("gflops") else "n/d"
    _pm  = f"{res['params_m']:.2f} M" if res.get("params_m") else "n/d"
    print(f"  Resolución:    {_res}   (entrada real, desde 1280x720)")
    print(f"  GFLOPs:        {_gf}   Parámetros: {_pm}")
    print("  El FPS depende de la implementación y de la resolución;")
    print("  los GFLOPs son la comparación de coste entre arquitecturas.")
    print("-" * 66)
    print("  AP@50-95 por tamaño de objeto (área en píxeles NATIVOS):")
    print(f"    small   (<32²  px) : {pct(res['map_small'])}")
    print(f"    medium  (32²-96²)  : {pct(res['map_medium'])}")
    print(f"    large   (>96²  px) : {pct(res['map_large'])}")
    print("-" * 66)
    class_ap = res.get("class_ap", {})
    gt = res.get("gt_per_class", {})
    if class_ap:
        print("  AP@50-95 por clase   (instancias de GT evaluadas):")
        for i, name in enumerate(class_names):
            val = class_ap.get(i)
            n = gt.get(i, 0)
            # El recuento hace visible qué cifras se apoyan en pocos objetos,
            # sin necesidad de excluir ninguna clase de la métrica.
            aviso = "  ← pocas instancias" if 0 < n < 200 else ""
            if val is None or val < 0:
                print(f"    {name:<16}   n/a   ({n:>6} inst){aviso}")
            else:
                print(f"    {name:<16} {val * 100:5.2f}%  ({n:>6} inst){aviso}")
    print("=" * 66)


def save_results(res: dict, class_names: list, model_name: str,
                 weights_path: str, img_size, conf: float):
    """Guarda las métricas en un JSON junto al checkpoint evaluado.

    Las clases sin instancias en GT ni en predicciones se marcan como -100
    (sentinel de "clase no evaluable"; distinto de 0 = detectado 0 %).
    """
    r2 = lambda v: None if v != v else round(v * 100, 2)   # NaN -> null en JSON
    out = {
        "model":      model_name,
        "weights":    weights_path,
        # Resolución de entrada: variable de primer nivel de la comparativa.
        # Determina el tamaño aparente de los objetos y hay que reportarla
        # junto al mAP, no como detalle de configuración.
        "img_size":   img_size,
        "conf":       conf,
        "max_det":    MAX_DET,
        # mAP sobre las 10 clases de BDD100K. No se excluye ninguna.
        "map50":      r2(res["map50"]),
        "map50_95":   r2(res["map50_95"]),
        # P/R/F1 en el punto de máximo F1, mismo criterio que Ultralytics
        "precision":   r2(res["precision"]),
        "recall":      r2(res["recall"]),
        "f1":          r2(res["f1"]),
        "conf_max_f1": round(res["conf_max_f1"], 4),
        "pr_iou":      PR_IOU_THRESH,
        # AP por tamaño, con el área medida en píxeles nativos (1280x720)
        "map_small":   r2(res["map_small"]),
        "map_medium":  r2(res["map_medium"]),
        "map_large":   r2(res["map_large"]),
        "fps":         round(res["fps"], 1),
        # Coste computacional. El FPS mezcla arquitectura, implementación y
        # resolución; los GFLOPs aíslan la arquitectura. Van juntos siempre.
        "gflops":      res.get("gflops"),
        "params_m":    res.get("params_m"),
        "input_h":     res.get("input_h"),
        "input_w":     res.get("input_w"),
        # AP por umbral de IoU (0.50:0.05:0.95), media sobre clases con GT.
        # None en una posición = ningún umbral tuvo clase evaluable (no debería
        # pasar con 10 clases y 2000 imágenes, pero se preserva por si acaso).
        "iou_thresholds": res.get("iou_thresholds"),
        "ap_por_iou":  [None if v is None else round(v * 100, 2)
                        for v in (res.get("ap_por_iou") or [])],
        "per_class":   {},
    }
    # AP por clase junto al número de instancias evaluadas: permite juzgar la
    # fiabilidad de cada cifra sin excluir clases de la métrica principal.
    class_ap = res.get("class_ap", {})
    gt = res.get("gt_per_class", {})
    pr_curve = res.get("pr_curve") or {}
    for i, name in enumerate(class_names):
        val = class_ap.get(i)
        out["per_class"][name] = {
            "ap50_95":     -100.0 if (val is None or val < 0) else round(val * 100, 2),
            "gt_instances": gt.get(i, 0),
            # Curva P-R completa a IoU=0.5 (rejilla estándar COCO de 101 puntos
            # de recall, 0..1); None si la clase no tenía GT evaluable.
            "pr_curve_iou50": pr_curve.get(i),
        }

    json_path = Path(weights_path).parent / "eval_results.json"
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultados guardados en: {json_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evalúa un modelo entrenado sobre la partición val de BDD100K."
    )
    parser.add_argument(
        "--model", required=True,
        choices=["yolo11", "faster_rcnn", "ssd", "detr", "cnn_baseline", "cnn_gru"],
    )
    parser.add_argument("--weights", required=True,  help="Ruta al checkpoint best.pt")
    parser.add_argument("--config",  default=None,   help="YAML de config del modelo")
    parser.add_argument("--data",    default="configs/bdd100k.yaml")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch de inferencia (default=1 para FPS real)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help="Umbral de confianza común a todos los modelos. "
                             "Mantener el valor por defecto para que los mAP "
                             "sean comparables entre arquitecturas.")
    args = parser.parse_args()

    # YOLO delega en el validador de Ultralytics
    bdd = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    if bdd["dataset"]["images_dir"] is None:
        raise ValueError("Configura dataset.images_dir en configs/bdd100k.yaml")
    nc = bdd["dataset"]["nc"]
    images_val = str(Path(bdd["dataset"]["images_dir"]) / "val")
    labels_val = str(Path(bdd["dataset"]["labels_dir"]) / "val")

    # YOLO predice con su propio preprocesado pero las métricas las calcula el
    # mismo acumulador que las del resto de modelos.
    if args.model == "yolo11":
        cfg = (yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
               if args.config else {})
        img_size = cfg.get("training", {}).get("img_size", 640)
        print(f"Modelo  : yolo11    Pesos: {args.weights}")
        # conf y max_det SI son comunes a los 5 modelos; img_size NO: cada
        # arquitectura corre en su resolución nativa y hay que declararlo.
        print(f"img_size: {img_size}   Conf: {args.conf} (común)  max_det: {MAX_DET} (común)")
        res = evaluate_yolo(args.weights, images_val, labels_val,
                            nc=nc, conf=args.conf, imgsz=img_size)
        try:
            from ultralytics import YOLO as _Y
            res.update(model_complexity(_Y(args.weights).model, "yolo11", img_size))
        except Exception as e:
            print(f"  [aviso] complejidad no calculable ({type(e).__name__}: {e})")
        print_results(res, CLASS_NAMES)
        save_results(res, CLASS_NAMES, "yolo11", args.weights, img_size, args.conf)
        return

    if args.config is None:
        parser.error("--config es obligatorio para modelos que no sean yolo11")

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    img_size = cfg["training"]["img_size"]
    device   = torch.device(DEVICE_TYPE)

    print(f"Modelo  : {args.model}")
    print(f"Pesos   : {args.weights}")
    print(f"Device  : {device}")
    # conf y max_det son comunes a los 5 modelos; img_size NO.
    print(f"img_size: {img_size}   Conf: {args.conf} (común)  max_det: {MAX_DET} (común)")

    model  = load_model(args.model, cfg, args.weights, nc, device, conf=args.conf)
    val_tf = get_val_transform(args.model, img_size)

    if args.model == "cnn_gru":
        seq = cfg.get("sequence", {})
        val_ds = BDD100KSequence(
            images_val, labels_val,
            T=seq.get("T", 4), transforms=val_tf,
        )
    else:
        val_ds = BDD100KDetection(images_val, labels_val, transforms=val_tf)

    loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_fn, pin_memory=(DEVICE_TYPE == "cuda"),
    )
    print(f"Val set : {len(val_ds)} imágenes  |  Batch: {args.batch_size}")

    res = evaluate_torchvision(
        model, loader, device, args.model, nc=nc, conf=args.conf
    )
    res.update(model_complexity(model, args.model, img_size, device))
    print_results(res, CLASS_NAMES)
    save_results(res, CLASS_NAMES, args.model, args.weights, img_size, args.conf)


if __name__ == "__main__":
    main()
