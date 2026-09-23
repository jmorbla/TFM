"""
Construye los datasets anidados para la curva de aprendizaje.

Genera un subset de entrenamiento por cada tamaño pedido, TODOS ANIDADOS entre
sí (8k ⊂ 16k ⊂ 32k), y con el MISMO conjunto de validación en los tres. Esa es
la condición que hace interpretable la curva: si al aumentar el tamaño cambiara
también la identidad de las imágenes o el conjunto de evaluación, la diferencia
de mAP mezclaría varios efectos y no mediría el que interesa.

Salida por cada tamaño N:
    bdd100k_labels_yolo_<N>/{train,val}/*.txt
    data_yolo_<N>/{images,labels}/{train,val}/     (hardlinks, sin coste en disco)
    bdd100k_yolo_<N>.yaml                          (para Ultralytics)

Uso (desde TFM/):
    python data/build_learning_curve.py                    # 8000 16000 32000
    python data/build_learning_curve.py --sizes 8000 16000 # otros tamaños
    python data/build_learning_curve.py --verify_only      # solo comprobar
"""

import argparse
import subprocess
import sys
from pathlib import Path

VAL_SUBSET = 2000     # idéntico en todos los tamaños
SEED = 42
CLASS_NAMES = ["person", "rider", "car", "truck", "bus", "train",
               "motor", "bike", "traffic light", "traffic sign"]

# Todo el experimento vive aquí, aislado del dataset principal del TFM para no
# mezclar los datos de la comparativa con los de esta prueba puntual.
BASE = Path("experiments/curva_aprendizaje/datasets")


def tag(size: int) -> str:
    return f"{size // 1000}k"


def labels_dir(size: int) -> Path:
    return BASE / f"labels_{size // 1000}k"


def data_dir(size: int) -> Path:
    return BASE / f"data_yolo_{size // 1000}k"


def yaml_for(size: int) -> Path:
    """YAML de Ultralytics apuntando al data_yolo de este tamaño.

    Ruta ABSOLUTA a propósito: Ultralytics resuelve las rutas relativas contra
    su DATASETS_DIR global, no contra la ubicación del YAML.
    """
    p = BASE / f"yolo_{size // 1000}k.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    p.write_text(
        f"# Generado por data/build_learning_curve.py — subset de {size} imágenes\n"
        f"path: {data_dir(size).resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )
    return p


def project_yaml_for(size: int) -> Path:
    """YAML en formato del proyecto (estilo configs/bdd100k.yaml).

    Lo necesitan train.py y evaluate.py — incluida la evaluación de YOLO, que
    recorre images_dir/labels_dir directamente para calcular las métricas con el
    mismo acumulador que el resto de modelos.
    """
    p = BASE / f"bdd100k_{tag(size)}.yaml"
    names = "\n".join(f"    {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    p.write_text(
        f"# Generado por data/build_learning_curve.py — subset de {size} imágenes\n"
        "dataset:\n"
        "  name: bdd100k\n"
        "  images_dir: bdd100k_images_100k/100k\n"
        f"  labels_dir: {labels_dir(size).as_posix()}\n"
        "  ann_dir: bdd100k_labels/100k\n"
        f"  names:\n{names}\n"
        f"  nc: {len(CLASS_NAMES)}\n"
        "training:\n"
        "  img_size: 640\n"
        "  batch_size: 8\n"
        "  num_workers: 2\n"
        "  pin_memory: true\n",
        encoding="utf-8",
    )
    return p


def build(size: int):
    labels, out = labels_dir(size), data_dir(size)
    print(f"\n{'=' * 70}\n  TAMAÑO {size}  ->  {labels}/  y  {out}/\n{'=' * 70}")

    subprocess.run([sys.executable, "data/prepare_bdd100k.py",
                    "--out_dir", str(labels),
                    "--subset_train", str(size),
                    "--subset_val", str(VAL_SUBSET),
                    "--seed", str(SEED)], check=True)
    subprocess.run([sys.executable, "data/build_yolo_dataset.py",
                    "--labels_src", str(labels), "--out", str(out)], check=True)
    print(f"  YAML Ultralytics: {yaml_for(size)}")
    print(f"  YAML proyecto   : {project_yaml_for(size)}")
    return labels


def verify(sizes: list):
    """Comprueba anidamiento del train e identidad del val."""
    print(f"\n{'=' * 70}\n  VERIFICACIÓN\n{'=' * 70}")
    tr, va = {}, {}
    for s in sizes:
        d = labels_dir(s)
        tr[s] = {p.stem for p in (d / "train").glob("*.txt")}
        va[s] = {p.stem for p in (d / "val").glob("*.txt")}
        print(f"  {s:>6}: train={len(tr[s]):>6}  val={len(va[s]):>5}")

    ok = True
    print("\n  Anidamiento del train (cada subset debe contener al anterior):")
    for a, b in zip(sizes, sizes[1:]):
        contenido = tr[a] <= tr[b]
        ok &= contenido
        print(f"    {a} ⊂ {b} : {'OK' if contenido else 'FALLO'}"
              f"   ({len(tr[a] & tr[b])}/{len(tr[a])} comunes)")

    print("\n  Conjunto de validación idéntico en todos los tamaños:")
    igual = all(va[s] == va[sizes[0]] for s in sizes)
    ok &= igual
    print(f"    {'OK' if igual else 'FALLO'}")

    print("\n  Sin fuga train/val:")
    for s in sizes:
        fuga = tr[s] & va[s]
        ok &= not fuga
        print(f"    {s:>6}: {len(fuga)} imágenes compartidas "
              f"{'OK' if not fuga else 'FALLO'}")

    print(f"\n  {'TODO CORRECTO' if ok else 'HAY PROBLEMAS'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[8000, 16000, 32000])
    ap.add_argument("--verify_only", action="store_true")
    ap.add_argument("--yaml_only", action="store_true",
                    help="Regenera solo los YAML (tras mover los datos de sitio)")
    args = ap.parse_args()
    sizes = sorted(args.sizes)

    BASE.mkdir(parents=True, exist_ok=True)
    if args.yaml_only:
        for s in sizes:
            print(f"  {yaml_for(s)}  y  {project_yaml_for(s)}")
    elif not args.verify_only:
        for s in sizes:
            build(s)
    sys.exit(0 if verify(sizes) else 1)


if __name__ == "__main__":
    main()
