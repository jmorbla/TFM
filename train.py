"""
Unified training entry point for all models except YOLO11
(which delegates to the ultralytics API via models/yolo11/train.py).

Usage:
    python train.py --model faster_rcnn --config configs/faster_rcnn.yaml
    python train.py --model ssd         --config configs/ssd.yaml
    python train.py --model detr        --config configs/detr.yaml
    python train.py --model cnn_baseline --config configs/cnn_baseline.yaml
    python train.py --model cnn_gru     --config configs/cnn_gru.yaml

All models train on the same 8 000-image BDD100K subset defined in
configs/bdd100k.yaml, sharing the same train/val split for a fair comparison.
"""

import argparse
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from data.dataset import BDD100KDetection, BDD100KSequence
from data.transforms import build_transforms
from evaluate import MAX_DET_THRESHOLDS, postprocess_predictions

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_WORKERS = 2          # keep low to avoid CLOCK_WATCHDOG_TIMEOUT on Windows
DEVICE_TYPE  = "cuda" if torch.cuda.is_available() else "cpu"

# Umbral de confianza de la validación intra-entrenamiento. Más alto que el de
# evaluate.py (0.001) porque aquí solo se necesita ordenar épocas entre sí, no
# reportar la métrica final: subirlo acelera mucho la validación sin cambiar
# prácticamente cuál es la mejor época.
VAL_CONF = 0.05


# ── Reproducibilidad ──────────────────────────────────────────────────────────

def set_seed(seed: int):
    """Fija todas las fuentes de aleatoriedad del entrenamiento.

    Necesario para la fase final del plan experimental: reportar media ±
    desviación típica sobre 3 semillas exige que las semillas sean realmente
    distintas y que cada run sea reproducible. Afecta a la inicialización de las
    cabezas, al barajado del DataLoader y al RandomHorizontalFlip de las
    transforms (que usa el módulo `random`).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init(worker_id: int):
    """Semilla distinta pero determinista por worker del DataLoader."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


# ── Model factory ─────────────────────────────────────────────────────────────

def get_model(name: str, cfg: dict, nc: int):
    if name == "faster_rcnn":
        from models.faster_rcnn.model import build_model
        return build_model(nc=nc, pretrained=cfg.get("pretrained", True))
    elif name == "ssd":
        from models.ssd.model import build_model
        return build_model(nc=nc, pretrained=cfg.get("pretrained", True))
    elif name == "detr":
        from models.detr.model import DETRWrapper
        return DETRWrapper(nc=nc, pretrained=cfg.get("pretrained", True))
    elif name == "cnn_baseline":
        from models.cnn_baseline.model import build_model
        return build_model(nc=nc, pretrained=cfg.get("pretrained_backbone", True))
    elif name == "cnn_gru":
        from models.cnn_gru.model import build_model
        seq = cfg.get("sequence", {})
        gru = cfg.get("gru", {})
        return build_model(
            nc=nc,
            T=seq.get("T", 4),
            gru_hidden=gru.get("hidden_size", 256),
            pretrained=cfg.get("pretrained_backbone", True),
        )
    else:
        raise ValueError(f"Unknown model: {name}")


# ── Optimizer ─────────────────────────────────────────────────────────────────

def get_optimizer(model, cfg: dict, model_name: str = ""):
    tr      = cfg["training"]
    opt_name = tr.get("optimizer", "SGD").lower()
    lr      = tr.get("lr", tr.get("lr0", 1e-3))
    wd      = tr.get("weight_decay", 1e-4)

    # DETR uses a lower LR for the ResNet backbone than for the transformer head
    if model_name == "detr" and "lr_backbone" in tr:
        lr_bb = tr["lr_backbone"]
        backbone_params = [
            p for n, p in model.named_parameters()
            if "backbone" in n and p.requires_grad
        ]
        other_params = [
            p for n, p in model.named_parameters()
            if "backbone" not in n and p.requires_grad
        ]
        print(f"  DETR lr groups — backbone: {lr_bb}  |  rest: {lr}")
        return torch.optim.AdamW(
            [{"params": backbone_params, "lr": lr_bb},
             {"params": other_params,   "lr": lr}],
            weight_decay=wd,
        )

    params = [p for p in model.parameters() if p.requires_grad]
    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif opt_name == "adam":
        return torch.optim.Adam(params, lr=lr)
    else:  # SGD
        return torch.optim.SGD(
            params, lr=lr, momentum=tr.get("momentum", 0.9), weight_decay=wd
        )


# ── Scheduler ─────────────────────────────────────────────────────────────────

def get_scheduler(optimizer, cfg: dict, epochs: int):
    tr    = cfg["training"]
    sched = tr.get("lr_scheduler", "cosine")
    if sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif sched == "step":
        step  = tr.get("lr_step_size", tr.get("lr_drop", 8))
        gamma = tr.get("lr_gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)
    return None


# ── Collate ───────────────────────────────────────────────────────────────────

def collate_fn(batch):
    return tuple(zip(*batch))


# ── One epoch of training ─────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, optimizer, scaler, device, amp: bool,
    model_name: str, writer, epoch, loss_fn=None, grad_clip: float = 0.0,
):
    model.train()
    total_loss = 0.0

    for imgs, targets in tqdm(loader, desc=f"Epoch {epoch + 1} train", leave=False):
        # Move to device
        if model_name == "cnn_gru":
            imgs = torch.stack(list(imgs)).to(device)   # (B, T, 3, H, W)
        elif model_name == "cnn_baseline":
            imgs = torch.stack(list(imgs)).to(device)   # (B, 3, H, W)
        else:
            imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # torchvision detectors use 0 = background; dataset labels are 0-indexed
        # → shift labels +1 so pedestrian=1, rider=2, …, traffic sign=10
        if model_name in ("faster_rcnn", "ssd"):
            targets = [
                {**t, "labels": t["labels"] + 1} for t in targets
            ]

        optimizer.zero_grad()

        with autocast(device_type=DEVICE_TYPE, enabled=amp):
            if model_name == "detr":
                # DETRWrapper.forward returns {"loss": scalar, ...}
                out  = model(torch.stack(imgs), targets)
                loss = out["loss"]

            elif model_name in ("faster_rcnn", "ssd"):
                # torchvision detectors return a dict of losses in training mode
                loss_dict = model(imgs, targets)
                loss = sum(loss_dict.values())

            else:
                # cnn_baseline / cnn_gru — use custom DetectionLoss
                cls_preds, reg_preds = model(imgs)
                loss = loss_fn(cls_preds, reg_preds, targets)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    avg = total_loss / max(len(loader), 1)
    writer.add_scalar("Loss/train", avg, epoch)
    return avg


# ── One epoch of validation ───────────────────────────────────────────────────

@torch.inference_mode()
def validate_one_epoch(
    model, loader, device, model_name: str, writer, epoch,
    amp: bool, conf: float = VAL_CONF,
):
    """Calcula mAP@50-95 y mAP@50 sobre el split de validación.

    Existe porque seleccionar best.pt por pérdida de ENTRENAMIENTO no es una
    selección en absoluto: la pérdida de train baja de forma monótona, así que
    la 'mejor' época acababa siendo siempre la última y no había ninguna
    protección frente al sobreajuste.

    Reutiliza postprocess_predictions() de evaluate.py para que el criterio de
    selección y la métrica final se calculen exactamente igual.

    Nota: en Faster R-CNN y SSD el filtrado por confianza lo hace torchvision
    internamente, así que aquí `conf` no les aplica y usan su umbral por defecto.
    Es irrelevante para el objetivo de esta función —ordenar épocas de UN mismo
    modelo entre sí—, pero por eso los valores que imprime no son comparables
    entre arquitecturas. Las cifras comparables son siempre las de evaluate.py.
    """
    model.eval()
    metric = MeanAveragePrecision(
        iou_type="bbox", max_detection_thresholds=MAX_DET_THRESHOLDS,
    )

    for imgs, targets in tqdm(loader, desc=f"Epoch {epoch + 1} val", leave=False):
        if model_name in ("cnn_gru", "cnn_baseline", "detr"):
            imgs_t = torch.stack([img.to(device) for img in imgs])
        else:
            imgs_t = [img.to(device) for img in imgs]

        with autocast(device_type=DEVICE_TYPE, enabled=amp):
            preds_raw = model(imgs_t)

        preds_tm = postprocess_predictions(preds_raw, model_name, imgs_t, device, conf)
        targets_tm = [
            {"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()} for t in targets
        ]
        metric.update(preds_tm, targets_tm)

    res = metric.compute()
    map50_95 = float(res["map"])
    map50    = float(res["map_50"])
    writer.add_scalar("mAP/val_50-95", map50_95, epoch)
    writer.add_scalar("mAP/val_50",    map50,    epoch)
    return map50_95, map50


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True,
        choices=["faster_rcnn", "ssd", "detr", "cnn_baseline", "cnn_gru"],
    )
    parser.add_argument("--config",   required=True)
    parser.add_argument("--data",     default="configs/bdd100k.yaml")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--runs_dir", default="runs",
                        help="Raíz donde guardar el run. Los experimentos "
                             "aparte usan la suya para no mezclar salidas.")
    parser.add_argument("--val_every", type=int, default=1,
                        help="Validar cada N épocas (1 = todas)")
    parser.add_argument("--val_conf", type=float, default=VAL_CONF,
                        help="Umbral de confianza de la validación por época")
    parser.add_argument("--patience", type=int, default=15,
                        help="Épocas sin mejorar el mAP de val antes de parar "
                             "(0 = sin early stopping). IDÉNTICO en los 5 modelos.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Semilla de reproducibilidad. La fase final del "
                             "plan usa 0, 1 y 2 para obtener la barra de error.")
    parser.add_argument("--lr", type=float, default=None,
                        help="Sobrescribe training.lr del YAML (fase de tuning)")
    parser.add_argument("--lr_backbone", type=float, default=None,
                        help="Sobrescribe training.lr_backbone (solo DETR)")
    args = parser.parse_args()

    set_seed(args.seed)

    # encoding explícito: read_text() usaría cp1252 en Windows y reventaría con
    # los comentarios acentuados y los separadores Unicode de los YAML.
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    bdd = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))

    # Overrides de CLI: permiten barrer el learning rate sin duplicar YAMLs.
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.lr_backbone is not None:
        cfg["training"]["lr_backbone"] = args.lr_backbone

    if bdd["dataset"]["images_dir"] is None:
        raise ValueError("Set dataset.images_dir in configs/bdd100k.yaml")

    tr       = cfg["training"]
    nc       = bdd["dataset"]["nc"]
    img_size = tr["img_size"]
    amp      = tr.get("amp", True)
    epochs   = tr["epochs"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = args.run_name or f"{args.model}_{timestamp}"
    run_dir   = Path(args.runs_dir) / args.model / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save configs for reproducibility
    shutil.copy(args.config, run_dir / "model_config.yaml")
    shutil.copy(args.data,   run_dir / "data_config.yaml")

    device = torch.device(DEVICE_TYPE)
    print(f"Model : {args.model}")
    print(f"Device: {device}")
    print(f"Run   : {run_dir}")
    print(f"Seed  : {args.seed}   lr: {tr.get('lr', tr.get('lr0'))}   "
          f"epochs(techo): {epochs}   patience: {args.patience}")

    # ── Transforms (model-specific) ──────────────────────────────────────────
    # Torchvision detectors (Faster R-CNN, SSD) apply their own normalisation
    # and resizing internally — we must NOT double-normalise.
    if args.model == "faster_rcnn":
        # Pass raw (non-normalised) tensors at original size; model handles the rest
        train_tf = build_transforms(None, train=True,  normalize=False)
        val_tf   = build_transforms(None, train=False, normalize=False)
    elif args.model == "ssd":
        # SSD-300 internal transform resizes to 300 × 300 and normalises
        train_tf = build_transforms(300, train=True,  square=True, normalize=False)
        val_tf   = build_transforms(300, train=False, square=True, normalize=False)
    elif args.model == "detr":
        # HuggingFace DETR: we pre-process (resize + normalise); no internal transform
        train_tf = build_transforms(img_size, train=True,  max_size=1333)
        val_tf   = build_transforms(img_size, train=False, max_size=1333)
    else:
        # cnn_baseline, cnn_gru: standard pipeline (resize + normalise)
        train_tf = build_transforms(img_size, train=True)
        val_tf   = build_transforms(img_size, train=False)

    # ── Dataset ──────────────────────────────────────────────────────────────
    images_train = str(Path(bdd["dataset"]["images_dir"]) / "train")
    images_val   = str(Path(bdd["dataset"]["images_dir"]) / "val")
    labels_train = str(Path(bdd["dataset"]["labels_dir"]) / "train")
    labels_val   = str(Path(bdd["dataset"]["labels_dir"]) / "val")

    if args.model == "cnn_gru":
        seq = cfg.get("sequence", {})
        T   = seq.get("T", 4)
        train_ds = BDD100KSequence(
            images_train, labels_train, T=T,
            stride=seq.get("stride", 2), transforms=train_tf,
        )
        val_ds = BDD100KSequence(
            images_val, labels_val, T=T,
            stride=seq.get("stride", 2), transforms=val_tf,
        )
    else:
        train_ds = BDD100KDetection(images_train, labels_train, transforms=train_tf)
        val_ds   = BDD100KDetection(images_val,   labels_val,   transforms=val_tf)

    pin = bdd["training"].get("pin_memory", True) and DEVICE_TYPE == "cuda"
    g = torch.Generator()
    g.manual_seed(args.seed)   # barajado del train_loader reproducible
    train_loader = DataLoader(
        train_ds, batch_size=tr["batch_size"], shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=pin,
        generator=g, worker_init_fn=_worker_init,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tr["batch_size"], shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=pin,
        worker_init_fn=_worker_init,
    )
    print(f"Train : {len(train_ds)} images  |  Val: {len(val_ds)} images")

    # ── Model, optimizer, scheduler ──────────────────────────────────────────
    model     = get_model(args.model, cfg, nc).to(device)
    optimizer = get_optimizer(model, cfg, model_name=args.model)
    scheduler = get_scheduler(optimizer, cfg, epochs)
    scaler    = GradScaler(device=DEVICE_TYPE, enabled=amp)
    writer    = SummaryWriter(log_dir=str(run_dir / "tb"))

    # Loss function (only needed for custom CNN models)
    loss_fn = None
    if args.model == "cnn_baseline":
        from models.cnn_baseline.loss import DetectionLoss
        loss_fn = DetectionLoss(nc=nc).to(device)
    elif args.model == "cnn_gru":
        from models.cnn_gru.loss import DetectionLoss
        loss_fn = DetectionLoss(nc=nc).to(device)

    # Log hyperparameters to TensorBoard
    hparams = {
        "model":        args.model,
        "epochs":       epochs,
        "batch_size":   tr["batch_size"],
        "img_size":     img_size,
        "optimizer":    tr.get("optimizer", "SGD"),
        "lr":           tr.get("lr", tr.get("lr0", "-")),
        "weight_decay": tr.get("weight_decay", "-"),
        "amp":          amp,
        "patience":     args.patience,
        "val_conf":     args.val_conf,
        "seed":         args.seed,
    }
    writer.add_text("hparams", str(hparams), 0)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_map   = -1.0
    best_epoch = -1
    epochs_sin_mejora = 0

    for epoch in range(epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler,
            device, amp, args.model, writer, epoch,
            loss_fn=loss_fn, grad_clip=tr.get("grad_clip", 0.0),
        )
        if scheduler:
            scheduler.step()

        # ── Validación real ──────────────────────────────────────────────────
        do_val = ((epoch + 1) % args.val_every == 0) or (epoch == epochs - 1)
        if do_val:
            map50_95, map50 = validate_one_epoch(
                model, val_loader, device, args.model, writer, epoch,
                amp, conf=args.val_conf,
            )
        else:
            map50_95 = map50 = float("nan")

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]
        val_str = (
            f"mAP50-95={map50_95 * 100:.2f}%  mAP50={map50 * 100:.2f}%"
            if do_val else "val omitida"
        )
        print(
            f"[{epoch + 1:03d}/{epochs}] "
            f"loss={train_loss:.4f}  {val_str}  lr={lr_now:.2e}  t={elapsed:.1f}s"
        )

        ckpt = {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss":            train_loss,
            "val_map50_95":    map50_95,
            "val_map50":       map50,
        }
        torch.save(ckpt, run_dir / "last.pt")

        # best.pt se elige por mAP de VALIDACIÓN, no por pérdida de entrenamiento.
        if do_val and map50_95 > best_map:
            best_map   = map50_95
            best_epoch = epoch + 1
            epochs_sin_mejora = 0
            torch.save(ckpt, run_dir / "best.pt")
            print(f"         ↳ nuevo mejor modelo (época {best_epoch})")
        elif do_val:
            epochs_sin_mejora += args.val_every
            if args.patience and epochs_sin_mejora >= args.patience:
                print(
                    f"\nEarly stopping: {epochs_sin_mejora} épocas sin mejorar "
                    f"el mAP de validación (mejor: época {best_epoch})."
                )
                break

    writer.add_hparams(
        {k: str(v) for k, v in hparams.items()},
        {"best_val_map50_95": best_map},
    )
    writer.close()
    print(f"\nEntrenamiento completado.")
    print(f"Mejor mAP@50-95 de validación: {best_map * 100:.2f}%  (época {best_epoch})")
    print(f"Run guardado en: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
