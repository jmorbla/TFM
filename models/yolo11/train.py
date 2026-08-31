"""
YOLO11m fine-tuning on BDD100K via Ultralytics API.

Usage (desde tfm-comparativa/):
    python models/yolo11/train.py
    python models/yolo11/train.py --run_name yolo11m_run2
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import YOLO

CONFIG_PATH   = "configs/yolo11m.yaml"
DATA_YAML     = "bdd100k_yolo.yaml"
RUNS_DIR      = "runs/yolo11"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default=CONFIG_PATH)
    parser.add_argument("--data",     default=DATA_YAML)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--runs_dir", default=RUNS_DIR,
                        help="Raíz donde guardar el run (los experimentos "
                             "aparte usan la suya para no mezclar salidas)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Semilla de reproducibilidad (fase final: 0, 1, 2)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Sobrescribe training.lr0 del YAML (fase de tuning)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tr  = cfg["training"]
    aug = cfg.get("augmentation", {})

    if args.lr is not None:
        tr["lr0"] = args.lr

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = args.run_name or f"yolo11m_{timestamp}"

    model = YOLO(cfg["weights"])
    model.train(
        data=args.data,
        epochs=tr["epochs"],
        imgsz=tr["img_size"],
        batch=tr["batch_size"],
        optimizer=tr["optimizer"],
        lr0=tr["lr0"],
        lrf=tr["lrf"],
        momentum=tr["momentum"],
        weight_decay=tr["weight_decay"],
        warmup_epochs=tr["warmup_epochs"],
        cos_lr=tr["cos_lr"],
        amp=tr["amp"],
        patience=tr["patience"],
        save_period=tr["save_period"],
        seed=args.seed,     # reproducibilidad: fase final con semillas 0, 1 y 2
        workers=2,          # reducido para evitar CLOCK_WATCHDOG_TIMEOUT
        cache=False,        # sin cache en RAM (evita presión de memoria)
        project=args.runs_dir,
        name=run_name,
        **aug,
    )

    # Copia el config usado al directorio del run para reproducibilidad
    run_dir = Path(args.runs_dir) / run_name
    if run_dir.exists():
        shutil.copy(args.config, run_dir / "model_config.yaml")
        shutil.copy(args.data,   run_dir / "data_config.yaml")
        print(f"\nConfigs guardados en: {run_dir}")


if __name__ == "__main__":
    main()
