"""
Convierte las etiquetas JSON por imagen de BDD100K → formato YOLO (un .txt por imagen).

Estructura de etiquetas de BDD100K:
    bdd100k_labels/100k/train/<image_stem>.json
    bdd100k_labels/100k/val/<image_stem>.json
    bdd100k_labels/100k/test/<image_stem>.json   # held-out, nunca usado en train/val

Cada JSON contiene un fotograma con una lista de objetos. Solo se convierten
los objetos con clave 'box2d' y una categoría de CLASS_MAP; las anotaciones de
carril/área (solo poly2d) se omiten sin aviso.

Salida (formato YOLO):
    bdd100k_labels_yolo/train/<image_stem>.txt
    bdd100k_labels_yolo/val/<image_stem>.txt
    bdd100k_labels_yolo/test/<image_stem>.txt

Uso (desde la raíz del repositorio):
    python data/prepare_bdd100k.py
    python data/prepare_bdd100k.py --subset_train 8000 --subset_val 2000 --subset_test 2000
    python data/prepare_bdd100k.py --subset_train 0 --subset_val 0 --subset_test 0  # todas las imágenes
"""

import argparse
import json
import random
from pathlib import Path

CLASS_MAP = {
    "person":        0,   # BDD100K usa "person", no "pedestrian"
    "rider":         1,
    "car":           2,
    "truck":         3,
    "bus":           4,
    "train":         5,
    "motor":         6,   # BDD100K usa "motor", no "motorcycle"
    "bike":          7,   # BDD100K usa "bike", no "bicycle"
    "traffic light": 8,
    "traffic sign":  9,
}

IMG_W, IMG_H = 1280, 720  # resolución nativa de BDD100K


def box_to_yolo(x1: float, y1: float, x2: float, y2: float) -> tuple:
    x1, x2 = max(0.0, x1), min(IMG_W, x2)
    y1, y2 = max(0.0, y1), min(IMG_H, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    cx = (x1 + x2) / 2 / IMG_W
    cy = (y1 + y2) / 2 / IMG_H
    w  = (x2 - x1) / IMG_W
    h  = (y2 - y1) / IMG_H
    return cx, cy, w, h


def convert_json(json_path: Path) -> list[str]:
    """Lee un JSON de BDD100K y devuelve sus líneas de etiqueta YOLO (puede estar vacío)."""
    data = json.loads(json_path.read_text())
    frames = data.get("frames", [])
    if not frames:
        return []

    objects = frames[0].get("objects", [])
    lines = []
    for obj in objects:
        cat = obj.get("category", "")
        if cat not in CLASS_MAP:
            continue
        box2d = obj.get("box2d")
        if box2d is None:
            continue
        result = box_to_yolo(box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"])
        if result is None:
            continue
        cx, cy, w, h = result
        lines.append(f"{CLASS_MAP[cat]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def convert_split(ann_dir: Path, out_dir: Path, n_subset: int, seed: int = 42,
                  clean: bool = True):
    json_files = sorted(ann_dir.glob("*.json"))
    total = len(json_files)

    if n_subset and n_subset < total:
        # shuffle + prefijo (no rng.sample): subsets de distinto tamaño y misma
        # semilla quedan anidados (8k ⊂ 16k ⊂ 32k).
        rng = random.Random(seed)
        rng.shuffle(json_files)
        json_files = json_files[:n_subset]
        print(f"  Sampling {n_subset} / {total} images (seed={seed}, anidado)")
    else:
        print(f"  Using all {total} images")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Sin esta limpieza, cambiar --subset_* o --seed dejaba los .txt del muestreo
    # anterior en el directorio: build_yolo_dataset.py los recogía con su glob y
    # se acababa entrenando sobre la UNIÓN de dos muestreos distintos.
    if clean:
        stale = list(out_dir.glob("*.txt"))
        for f in stale:
            f.unlink()
        if stale:
            print(f"  Limpiados {len(stale)} .txt de una ejecución anterior")

    written = skipped = 0

    for jp in json_files:
        lines = convert_json(jp)
        if not lines:
            skipped += 1
            continue
        (out_dir / jp.with_suffix(".txt").name).write_text("\n".join(lines))
        written += 1

    print(f"  Written: {written}  |  Skipped (no det. labels): {skipped}")


def main():
    parser = argparse.ArgumentParser(description="BDD100K JSON → YOLO labels")
    parser.add_argument("--ann_dir",      default="bdd100k_labels/100k",   help="Root of per-image JSON labels")
    parser.add_argument("--out_dir",      default="bdd100k_labels_yolo",   help="Output root for YOLO .txt files")
    parser.add_argument("--subset_train", type=int, default=8000,          help="Train images to sample (0 = all)")
    parser.add_argument("--subset_val",   type=int, default=2000,          help="Val images to sample (0 = all)")
    parser.add_argument("--subset_test",  type=int, default=2000,          help="Test images to sample (0 = all)")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--no_clean",     action="store_true",
                        help="No borrar los .txt previos (arriesga mezclar muestreos)")
    args = parser.parse_args()

    ann  = Path(args.ann_dir)
    out  = Path(args.out_dir)
    clean = not args.no_clean

    print("Converting train split...")
    convert_split(ann / "train", out / "train", args.subset_train, args.seed, clean)

    print("Converting val split...")
    convert_split(ann / "val",   out / "val",   args.subset_val,   args.seed, clean)

    print("Converting test split...")
    convert_split(ann / "test",  out / "test",  args.subset_test,  args.seed, clean)

    print("\nDone. YOLO labels written to:", out.resolve())


if __name__ == "__main__":
    main()
