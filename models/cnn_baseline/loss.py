"""
Anchor-based detection loss for CNNBaseline (and CNNGRUDetector).

Pipeline:
  1. Generate anchors matching each FPN level's grid (3 anchors/cell × 3 levels).
  2. Match each anchor to ground-truth boxes by IoU threshold.
  3. Focal Loss  → classification  (handles the huge pos/neg imbalance).
  4. Smooth-L1   → box regression  (only on positive anchors).

Reference sizes are for img_size=416; the code is resolution-agnostic because
feature-map shapes are read directly from the model's output tensors.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou


# ── Anchor configuration (must match model.py) ───────────────────────────────
#   _ANCHOR_BASE_SIZES[i] = geometric mean of (min,max) for FPN level i
# Configuración derivada empíricamente del propio BDD100K.
#
# El dataset es, ante todo, un problema de objetos pequeños: traffic light
# (16 px de mediana) y traffic sign (21 px) suman el 33 % de las instancias.
#
# Evolución de la cobertura de GT con IoU >= 0.5 (medida sobre 6 000 objetos):
#   45/91/181 px, 3 anchors, P3-P5  ->  24.5 %   (configuración original)
#   12/24/48 px, 6 anchors, P3-P5   ->  60.0 %
#   8/16/32/64 px, 6 anchors, P2-P5 ->  87.5 %   (ésta, a 416 px)
#                                       90.6 %   (a 640 px, la resolución actual)
#
# El nivel P2 (stride 4) es el que marca la diferencia y sale gratis del
# backbone: mobilenet_v3_small.features[:2] ya produce stride 4, sólo había
# que conectarlo a la FPN en lugar de descartarlo.
_ANCHOR_BASE_SIZES = [8.0, 16.0, 32.0, 64.0]   # niveles FPN P2-P5
_ASPECT_RATIOS = [0.5, 1.0, 2.0]               # w/h — GT: p10=0.46, p50=1.11, p90=2.05
_ANCHOR_SCALES = [1.0, 1.6]                    # 2 escalas por nivel
_STRIDES = [4, 8, 16, 32]

# Anchors por celda = escalas × ratios.
# model.py DEBE usar este valor como num_anchors, y evaluate.py para desaplanar.
NUM_ANCHORS = len(_ANCHOR_SCALES) * len(_ASPECT_RATIOS)
# Número de niveles de la FPN. model.py construye una entrada por cada uno.
NUM_FPN_LEVELS = len(_STRIDES)

# Prior de la focal loss (Lin et al. 2017, §4.1): probabilidad inicial que se
# asigna a la clase positiva al inicializar el bias del head de clasificación.
FOCAL_PRIOR = 0.01

POS_IOU_THRESH = 0.50
NEG_IOU_THRESH = 0.40
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0


# ── Anchor generation ─────────────────────────────────────────────────────────

def _anchors_for_level(
    base_size: float, stride: int, fH: int, fW: int, device
) -> torch.Tensor:
    """
    Returns (fH * fW * A, 4) anchors in xyxy image-pixel coords, A = NUM_ANCHORS.
    Order: for every spatial cell (row-major), the A variants generated as
    `for escala: for ratio:` — el mismo orden que asume el reshape de
    DetectionLoss.forward() y el de _cnn_postprocess() en evaluate.py.
    """
    cx = (torch.arange(fW, device=device) + 0.5) * stride   # (fW,)
    cy = (torch.arange(fH, device=device) + 0.5) * stride   # (fH,)
    grid_y, grid_x = torch.meshgrid(cy, cx, indexing="ij")   # (fH, fW)
    cx_flat = grid_x.flatten()   # (fH*fW,)
    cy_flat = grid_y.flatten()   # (fH*fW,)

    per_anchor = []
    for scale in _ANCHOR_SCALES:
        for r in _ASPECT_RATIOS:
            w = base_size * scale * math.sqrt(r)
            h = base_size * scale / math.sqrt(r)
            boxes = torch.stack(
                [cx_flat - w / 2, cy_flat - h / 2, cx_flat + w / 2, cy_flat + h / 2],
                dim=1,
            )  # (fH*fW, 4)
            per_anchor.append(boxes)

    # (fH*fW, A, 4)  →  (fH*fW*A, 4)
    return torch.stack(per_anchor, dim=1).reshape(-1, 4)


def generate_anchors(
    feature_shapes: list[tuple[int, int]], device
) -> torch.Tensor:
    """
    All anchors for the 3 FPN levels concatenated: (N_total, 4) xyxy.
    feature_shapes: [(fH0, fW0), (fH1, fW1), (fH2, fW2)]
    """
    parts = []
    for i, (fH, fW) in enumerate(feature_shapes):
        parts.append(
            _anchors_for_level(_ANCHOR_BASE_SIZES[i], _STRIDES[i], fH, fW, device)
        )
    return torch.cat(parts, dim=0)


# ── Box encoding / decoding ───────────────────────────────────────────────────

def encode_boxes(gt_xyxy: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """
    Encode GT boxes as (tx, ty, tw, th) deltas relative to matched anchors.
    Standard RCNN-style parameterisation.
    """
    aw = (anchors[:, 2] - anchors[:, 0]).clamp(min=1e-6)
    ah = (anchors[:, 3] - anchors[:, 1]).clamp(min=1e-6)
    acx = anchors[:, 0] + 0.5 * aw
    acy = anchors[:, 1] + 0.5 * ah

    gw = (gt_xyxy[:, 2] - gt_xyxy[:, 0]).clamp(min=1e-6)
    gh = (gt_xyxy[:, 3] - gt_xyxy[:, 1]).clamp(min=1e-6)
    gcx = gt_xyxy[:, 0] + 0.5 * gw
    gcy = gt_xyxy[:, 1] + 0.5 * gh

    tx = (gcx - acx) / aw
    ty = (gcy - acy) / ah
    tw = torch.log(gw / aw)
    th = torch.log(gh / ah)
    return torch.stack([tx, ty, tw, th], dim=1)


# ── Focal loss ────────────────────────────────────────────────────────────────

def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    alpha: float = FOCAL_ALPHA,
    gamma: float = FOCAL_GAMMA,
) -> torch.Tensor:
    """
    Binary focal loss summed over all entries.
    logits/targets: (N, nc) — multi-label, one binary sigmoid per class.
    valid_mask    : (N,) bool. Si se pasa, los anchors con False quedan
                    EXCLUIDOS de la pérdida (banda de ignorados). Es necesario
                    porque focal loss con gamma=2 amplifica precisamente los
                    ejemplos ambiguos, que son los que hay que ignorar.
    """
    p = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * (1.0 - p_t) ** gamma * bce      # (N, nc)
    if valid_mask is not None:
        loss = loss[valid_mask]
    return loss.sum()


# ── Main loss class ───────────────────────────────────────────────────────────

class DetectionLoss(nn.Module):
    """
    Combined Focal + Smooth-L1 detection loss.

    Accepts the raw outputs of CNNBaseline / CNNGRUDetector:
        cls_preds: list of (B, A*nc, fH, fW)  — one tensor per FPN level
        reg_preds: list of (B, A*4,  fH, fW)  — one tensor per FPN level
        targets  : list of dicts {'boxes': (M,4) xyxy, 'labels': (M,)}

    Returns a scalar loss tensor.
    """

    def __init__(self, nc: int = 10, lambda_reg: float = 1.0):
        super().__init__()
        self.nc = nc
        self.lambda_reg = lambda_reg
        self._anchor_cache: dict = {}

    def _get_anchors(self, feature_shapes, device):
        key = (tuple(feature_shapes), str(device))
        if key not in self._anchor_cache:
            self._anchor_cache[key] = generate_anchors(feature_shapes, device)
        return self._anchor_cache[key]

    def forward(
        self,
        cls_preds: list[torch.Tensor],
        reg_preds: list[torch.Tensor],
        targets: list[dict],
    ) -> torch.Tensor:
        B = cls_preds[0].shape[0]
        device = cls_preds[0].device
        A = NUM_ANCHORS  # anchors per cell = escalas × ratios

        # ── Feature shapes & anchors ─────────────────────────────────────────
        feature_shapes = [(p.shape[2], p.shape[3]) for p in cls_preds]
        anchors = self._get_anchors(feature_shapes, device)  # (N_total, 4)
        N = anchors.shape[0]

        # ── Reshape predictions to (B, N_total, nc) and (B, N_total, 4) ──────
        # Conv output channels are ordered [a0_c0, ..., a0_c(nc-1), a1_c0, ...]
        # i.e. A groups of nc — reshape to (B, A, nc, fH, fW) → (B, fH, fW, A, nc)
        cls_flat, reg_flat = [], []
        for cp, rp in zip(cls_preds, reg_preds):
            B_, _, fH, fW = cp.shape
            cls_flat.append(
                cp.reshape(B_, A, self.nc, fH, fW)
                  .permute(0, 3, 4, 1, 2)    # (B, fH, fW, A, nc)
                  .reshape(B_, fH * fW * A, self.nc)
            )
            reg_flat.append(
                rp.reshape(B_, A, 4, fH, fW)
                  .permute(0, 3, 4, 1, 2)    # (B, fH, fW, A, 4)
                  .reshape(B_, fH * fW * A, 4)
            )
        cls_all = torch.cat(cls_flat, dim=1)  # (B, N, nc)
        reg_all = torch.cat(reg_flat, dim=1)  # (B, N, 4)

        total_cls = cls_all.new_zeros(1)
        total_reg = cls_all.new_zeros(1)
        num_pos   = 0

        for b in range(B):
            gt_boxes  = targets[b]["boxes"].to(device)    # (M, 4) xyxy
            gt_labels = targets[b]["labels"].to(device)   # (M,)
            M = gt_boxes.shape[0]

            if M == 0:
                # All anchors are background — only cls loss, no reg loss
                cls_tgt = torch.zeros(N, self.nc, device=device)
                total_cls = total_cls + focal_loss(cls_all[b], cls_tgt)
                continue

            # ── IoU matching ─────────────────────────────────────────────────
            iou = box_iou(anchors, gt_boxes)        # (N, M)
            max_iou, best_gt = iou.max(dim=1)       # (N,)

            pos_mask    = max_iou >= POS_IOU_THRESH
            assigned_gt = best_gt.clone()           # GT que supervisa a cada anchor

            # Cada GT reclama su mejor anchor, garantizando que ningún objeto se
            # quede sin supervisión (crítico: solo el 60 % de los GT de BDD100K
            # alcanza IoU >= 0.5 con algún anchor).
            # assigned_gt DEBE actualizarse a la vez que pos_mask; en caso
            # contrario el anchor forzado aprendería la caja y la clase de OTRO
            # objeto — el fallo que dejaba al 28.9 % de los GT sin supervisar.
            best_anchor = iou.argmax(dim=0)                              # (M,)
            pos_mask[best_anchor] = True
            assigned_gt[best_anchor] = torch.arange(M, device=device)

            # Banda de ignorados: ni positivos claros ni negativos claros.
            # Se excluyen de la pérdida de clasificación en lugar de contarlos
            # como fondo (uso real de NEG_IOU_THRESH).
            ignore_mask = (
                (~pos_mask)
                & (max_iou >= NEG_IOU_THRESH)
                & (max_iou < POS_IOU_THRESH)
            )
            valid_mask = ~ignore_mask

            # ── Classification ────────────────────────────────────────────────
            cls_tgt = torch.zeros(N, self.nc, device=device)
            pos_idx = pos_mask.nonzero(as_tuple=True)[0]
            if len(pos_idx):
                cls_tgt[pos_idx, gt_labels[assigned_gt[pos_idx]]] = 1.0
            total_cls = total_cls + focal_loss(cls_all[b], cls_tgt, valid_mask=valid_mask)

            # ── Regression (positives only) ───────────────────────────────────
            if len(pos_idx):
                reg_tgt = encode_boxes(gt_boxes[assigned_gt[pos_idx]], anchors[pos_idx])
                reg_loss = F.smooth_l1_loss(reg_all[b][pos_idx], reg_tgt, reduction="sum")
                total_reg = total_reg + reg_loss
                num_pos  += len(pos_idx)

        denom = max(num_pos, 1)
        return total_cls / denom + self.lambda_reg * total_reg / denom
