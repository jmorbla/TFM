"""
Construye data_yolo/ con la estructura que Ultralytics espera:
    data_yolo/images/train/  ← hardlinks a las imágenes muestreadas
    data_yolo/images/val/
    data_yolo/labels/train/  ← copia los .txt de bdd100k_labels_yolo/
    data_yolo/labels/val/

Los hardlinks son instantáneos y no ocupan espacio adicional en disco.
Las imágenes incluidas son exactamente las que tienen label en bdd100k_labels_yolo/.

Uso (desde la raíz del repositorio):
    python data/build_yolo_dataset.py
"""

import argparse
import os
import shutil
from pathlib import Path

IMAGES_SRC  = Path("bdd100k_images_100k/100k")
LABELS_SRC  = Path("bdd100k_labels_yolo")
OUT         = Path("data_yolo")


def build_split(split: str, clean: bool = True,
                labels_src: Path = None, out: Path = None):
    labels_src = labels_src or LABELS_SRC
    out = out or OUT
    img_src = IMAGES_SRC / split
    lbl_src = labels_src / split
    img_dst = out / "images" / split
    lbl_dst = out / "labels" / split

    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    # Purga de ejecuciones anteriores: si el muestreo cambió, los hardlinks y
    # .txt viejos sobrevivirían y Ultralytics entrenaría sobre imágenes que ya
    # no forman parte del subset. También se elimina el .cache de Ultralytics,
    # que de lo contrario conserva el índice del dataset antiguo.
    if clean:
        n = 0
        for d, pat in ((img_dst, "*.jpg"), (lbl_dst, "*.txt")):
            for f in d.glob(pat):
                f.unlink(); n += 1
        for cache in out.glob(f"labels/{split}.cache"):
            cache.unlink()
        if n:
            print(f"  {split}: limpiados {n} ficheros de una ejecución anterior")

    label_files = list(lbl_src.glob("*.txt"))
    print(f"  {split}: {len(label_files)} imágenes con label")

    missing = 0
    for lbl in label_files:
        stem = lbl.stem
        src_img = img_src / f"{stem}.jpg"
        dst_img = img_dst / f"{stem}.jpg"
        dst_lbl = lbl_dst / lbl.name

        if not src_img.exists():
            missing += 1
            continue

        # Hardlink a la imagen (sin copiar datos)
        if not dst_img.exists():
            os.link(src_img, dst_img)

        # Copia el .txt siempre (sobreescribe para reflejar cambios en las etiquetas)
        shutil.copy2(lbl, dst_lbl)

    if missing:
        print(f"  Advertencia: {missing} imágenes no encontradas en {img_src}")


def main():
    parser = argparse.ArgumentParser(description="Construye data_yolo/ para Ultralytics")
    parser.add_argument("--no_clean", action="store_true",
                        help="No purgar la construcción anterior (arriesga mezclar subsets)")
    parser.add_argument("--labels_src", default=str(LABELS_SRC),
                        help="Raíz de los .txt en formato YOLO")
    parser.add_argument("--out", default=str(OUT),
                        help="Directorio de salida (uno por tamaño en la curva "
                             "de aprendizaje: data_yolo_8k, data_yolo_16k, ...)")
    args = parser.parse_args()
    clean = not args.no_clean
    labels_src, out = Path(args.labels_src), Path(args.out)

    print(f"Construyendo {out}/ desde {labels_src}/ ...")
    build_split("train", clean, labels_src, out)
    build_split("val", clean, labels_src, out)
    print(f"\nHecho. Estructura en: {out.resolve()}")
    print(f"El YAML de Ultralytics debe apuntar a: {out.resolve().as_posix()}")


if __name__ == "__main__":
    main()
