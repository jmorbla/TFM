"""
Estadísticas del subconjunto de BDD100K empleado en la comparativa.

Produce las cifras que `evaluate.py` NO puede dar, porque no dependen de ningún
modelo sino de las anotaciones: número de instancias y tamaño mediano de objeto
por clase. Rellena la tabla `tab:met-clases` de docs/metodologia.tex.

Definición de tamaño: raíz cuadrada del área de la caja en píxeles NATIVOS
(1280x720), es decir, el lado de un cuadrado de área equivalente. Es la misma
convención con la que se obtuvieron las cifras ya citadas en la memoria
(traffic light 16 px, traffic sign 21 px), verificada al implementar esto.

Uso (desde TFM/):
    python stats_dataset.py
    python stats_dataset.py --split train
    python stats_dataset.py --labels bdd100k_labels_yolo/val
"""

import argparse
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CLASS_NAMES = [
    "person", "rider", "car", "truck", "bus",
    "train", "motor", "bike", "traffic light", "traffic sign",
]
IMG_W, IMG_H = 1280, 720          # resolución nativa de BDD100K


def recolectar(labels_dir: Path) -> tuple:
    """Devuelve (tamaños por clase, nº de imágenes, nº de ficheros vacíos)."""
    tam = defaultdict(list)
    n_img = n_vacios = 0
    for f in sorted(labels_dir.glob("*.txt")):
        n_img += 1
        lineas = [l for l in f.read_text(encoding="utf-8").split("\n") if l.strip()]
        if not lineas:
            n_vacios += 1
        for l in lineas:
            p = l.split()
            if len(p) != 5:
                continue
            c = int(p[0])
            w_px = float(p[3]) * IMG_W
            h_px = float(p[4]) * IMG_H
            tam[c].append((w_px * h_px) ** 0.5)
    return tam, n_img, n_vacios


def fmt_es(x: float, dec: int = 1) -> str:
    """Formato español: coma decimal."""
    return f"{x:.{dec}f}".replace(".", ",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=None,
                    help="Directorio de etiquetas YOLO. Por defecto: "
                         "data_yolo/labels/<split>")
    ap.add_argument("--split", default="val", choices=["train", "val"],
                    help="Partición a analizar (default: val, que es sobre la "
                         "que se reportan las métricas)")
    args = ap.parse_args()

    d = Path(args.labels) if args.labels else Path(f"data_yolo/labels/{args.split}")
    if not d.exists():
        sys.exit(f"No existe {d}. ¿Has ejecutado data/prepare_bdd100k.py?")

    tam, n_img, n_vacios = recolectar(d)
    todos = [s for v in tam.values() for s in v]
    if not todos:
        sys.exit(f"{d} no contiene ninguna anotación.")

    print(f"\n{d}  —  {n_img} imágenes, {len(todos)} instancias")
    if n_vacios:
        print(f"  ({n_vacios} imágenes sin ninguna anotación)")
    print("=" * 78)
    print(f"{'Id':>3} {'Clase':<15}{'Inst.':>9}{'% del tot':>11}"
          f"{'Mediana':>10}{'p25':>8}{'p75':>8}")
    print("-" * 78)

    for i, nombre in enumerate(CLASS_NAMES):
        v = sorted(tam.get(i, []))
        if not v:
            print(f"{i:>3} {nombre:<15}{0:>9}{'—':>11}{'—':>10}{'—':>8}{'—':>8}")
            continue
        pct = 100 * len(v) / len(todos)
        med = st.median(v)
        # Con muy pocas instancias los cuantiles no significan nada; se marcan.
        if len(v) >= 4:
            q = st.quantiles(v, n=4)
            p25, p75 = f"{q[0]:.1f}", f"{q[2]:.1f}"
        else:
            p25 = p75 = "n/a"
        print(f"{i:>3} {nombre:<15}{len(v):>9}{pct:>10.2f}%"
              f"{med:>10.1f}{p25:>8}{p75:>8}")

    print("-" * 78)
    print(f"{'':>3} {'GLOBAL':<15}{len(todos):>9}{100.0:>10.2f}%"
          f"{st.median(todos):>10.1f}")

    # ── Cifras que la memoria cita en prosa ──────────────────────────────────
    peq = sum(1 for s in todos if s < 32)
    print(f"\nObjetos por debajo de 32 px EN RESOLUCIÓN NATIVA "
          f"(umbral 'small' de MS COCO): {100 * peq / len(todos):.1f} %")
    señal = len(tam.get(8, [])) + len(tam.get(9, []))
    print(f"traffic light + traffic sign: {100 * señal / len(todos):.1f} % "
          f"de las instancias")
    conteos = {i: len(v) for i, v in tam.items() if v}
    if conteos:
        mx, mn = max(conteos.values()), min(conteos.values())
        c_mx = CLASS_NAMES[max(conteos, key=conteos.get)]
        c_mn = CLASS_NAMES[min(conteos, key=conteos.get)]
        print(f"Desbalanceo: {mx}:{mn} ({c_mx} frente a {c_mn}) = "
              f"{mx / mn:.0f}:1")

    # ── Tamaño aparente en la entrada REAL de cada arquitectura ─────────────
    # El umbral 'small' de 32 px no se aplica sobre la imagen nativa sino sobre
    # lo que cada modelo ve. Las escalas salen de evaluate.py::eval_input_shape
    # y difieren en un factor de 3.3 entre SSD300 y Faster R-CNN, así que la
    # dificultad del problema NO es la misma para los seis modelos.
    ESCALAS = [
        ("SSD300",        0.313, "300x300"),
        ("YOLO11m (A/B)", 0.500, "640x384"),
        ("CNN baseline",  0.889, "1137x640"),
        ("Faster R-CNN",  1.041, "1333x749"),
        ("DETR",          1.041, "1333x749"),
    ]
    print("\n" + "=" * 78)
    print("Tamaño aparente de los objetos en la entrada real de cada modelo:")
    print("=" * 78)
    print(f"{'Modelo':<16}{'Entrada':>11}{'Escala':>9}{'Mediana':>10}"
          f"{'<32 px':>9}{'<16 px':>9}")
    print("-" * 78)
    tl = tam.get(8, [])   # traffic light, la clase más pequeña y el 13.8 %
    for nombre, esc, entrada in ESCALAS:
        esc_med = st.median(todos) * esc
        p32 = 100 * sum(1 for s in todos if s * esc < 32) / len(todos)
        p16 = 100 * sum(1 for s in todos if s * esc < 16) / len(todos)
        print(f"{nombre:<16}{entrada:>11}{esc:>9.3f}{esc_med:>9.1f}px"
              f"{p32:>8.1f}%{p16:>8.1f}%")
    print("-" * 78)
    if tl:
        print("Semáforo mediano (16,1 px nativos) en cada entrada: " + "  ".join(
            f"{n.split()[0]} {st.median(tl) * e:.1f}px" for n, e, _ in ESCALAS))

    # ── Filas LaTeX para tab:met-clases ──────────────────────────────────────
    print("\n" + "=" * 78)
    print("Filas para tab:met-clases de docs/metodologia.tex (copiar y pegar):")
    print("=" * 78)
    for i, nombre in enumerate(CLASS_NAMES):
        v = tam.get(i, [])
        inst = f"{len(v):,}".replace(",", "\\,")
        med = f"{fmt_es(st.median(v))} px" if v else "---"
        print(f"{i} & \\texttt{{{nombre}}} & {inst} & {med} \\\\")


if __name__ == "__main__":
    main()
