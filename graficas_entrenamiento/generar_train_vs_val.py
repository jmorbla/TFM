# -*- coding: utf-8 -*-
"""Train loss frente a val loss (o val mAP) por modelo y semilla, para vigilar sobreajuste.

YOLO11m (A/B): results.csv trae loss de train y de val (box+cls+dfl) por epoca -> se
dibujan juntas.
CNN, Faster R-CNN, SSD300, DETR: train.py solo persiste Loss/train y mAP de val; NO hay
loss de validacion. Se dibuja Loss/train (eje izq.) y mAP@50-95 val (eje dcho.).

Uso (desde TFM/graficas_entrenamiento/): python generar_train_vs_val.py
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS = Path("../runs")
FIG = Path("figuras_entrenamiento")
FIG.mkdir(exist_ok=True)
TRAIN_C, VAL_C, MAP_C = "#2a78d6", "#d03b3b", "#0ca30c"

YOLO = [("YOLO11m (A)", "yolo_a_final", "11_yolo_a_train_vs_val"),
        ("YOLO11m (B)", "yolo_b_final", "12_yolo_b_train_vs_val")]
TB = [("CNN baseline", "cnnb_final", "13_cnn_train_vs_valmap"),
      ("Faster R-CNN", "frcnn_final", "14_frcnn_train_vs_valmap"),
      ("SSD300", "ssd_final", "15_ssd_train_vs_valmap"),
      ("DETR", "detr_final", "16_detr_train_vs_valmap")]

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "font.size": 10})


def carpeta(nombre):
    c = sorted(p for p in RUNS.rglob(nombre) if p.is_dir())
    return c[0] if c else None


def tb(nombre):
    c = carpeta(nombre)
    ea = EventAccumulator(str(c / "tb"), size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ("Loss/train", "mAP/val_50-95"):
        d = pd.DataFrame([(e.step, e.value, e.wall_time) for e in ea.Scalars(tag)],
                         columns=["step", "v", "t"])
        d = d.sort_values("t").drop_duplicates("step", keep="last").sort_values("step")
        out[tag] = d.set_index("step")["v"]
    df = pd.DataFrame(out).reset_index()
    df["epoch"] = df["step"] + 1
    return df


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(FIG / f"{nombre}.png", dpi=150)
    fig.savefig(FIG / f"{nombre}.pdf")
    plt.close(fig)
    print("OK", nombre)


for etq, pref, fname in YOLO:
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for s, ax in enumerate(axs):
        df = pd.read_csv(carpeta(f"{pref}_s{s}") / "results.csv")
        df.columns = [c.strip() for c in df.columns]
        tr = df[["train/box_loss", "train/cls_loss", "train/dfl_loss"]].sum(axis=1)
        va = df[["val/box_loss", "val/cls_loss", "val/dfl_loss"]].sum(axis=1)
        ax.plot(df["epoch"], tr, color=TRAIN_C, label="train")
        ax.plot(df["epoch"], va, color=VAL_C, label="val")
        mejor = df["metrics/mAP50-95(B)"].idxmax()
        ax.axvline(df["epoch"][mejor], color="grey", ls="--", lw=1, label=f"mejor mAP (ép. {df['epoch'][mejor]})")
        ax.set_title(f"{etq} · semilla {s}")
        ax.set_xlabel("época")
        ax.set_ylabel("loss (box + cls + dfl)")
        ax2 = ax.twinx()
        ax2.grid(False)
        ax2.plot(df["epoch"], df["metrics/mAP50-95(B)"] * 100, color=MAP_C, label="mAP@50-95 val")
        ax2.set_ylabel("mAP@50-95 val (%)", color=MAP_C)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=8)
    fig.suptitle(f"{etq}: loss train, loss val y mAP val")
    guardar(fig, fname)

for etq, pref, fname in TB:
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for s, ax in enumerate(axs):
        df = tb(f"{pref}_s{s}")
        ax.plot(df["epoch"], df["Loss/train"], color=TRAIN_C, label="loss train")
        ax.set_xlabel("época")
        ax.set_ylabel("loss train", color=TRAIN_C)
        ax2 = ax.twinx()
        ax2.grid(False)
        v = df.dropna(subset=["mAP/val_50-95"])
        ax2.plot(v["epoch"], v["mAP/val_50-95"] * 100, color=MAP_C, label="mAP val")
        ax2.set_ylabel("mAP@50-95 val (%)", color=MAP_C)
        mejor = v.loc[v["mAP/val_50-95"].idxmax(), "epoch"]
        ax.axvline(mejor, color="grey", ls="--", lw=1)
        ax.set_title(f"{etq} · semilla {s} (mejor ép. {int(mejor)})")
    fig.suptitle(f"{etq}: loss train y mAP val (no hay loss de val registrada)")
    guardar(fig, fname)
