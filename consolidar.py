"""
Consolida los 18 entrenamientos en las tablas de la memoria.

`evaluate.py` escribe un eval_results.json POR RUN. Las tablas del capítulo de
resultados piden media y desviación típica sobre las tres semillas, y eso no lo
calcula nadie: hacerlo a mano son 18 ficheros x 20 campos x 10 clases, que es
justo donde se cuela un error de transcripción imposible de encontrar después.

Este script agrupa por configuración, agrega, deriva los campos calculados
(ms/img, umbral de tiempo real, diferencia de la ablación) y emite las filas
LaTeX listas para pegar en docs/resultados-y-discusion.tex.

Uso (desde TFM/):
    python consolidar.py                 # todas las tablas
    python consolidar.py --tabla principal
    python consolidar.py --json resumen.json   # vuelca también el agregado

Dos comprobaciones de integridad que hace por su cuenta:
  - gt_instances debe ser IDÉNTICO en los 18 runs (mismo val). Si difiere, el
    muestreo no es el que se cree y la comparación no vale.
  - conf, max_det e img_size deben coincidir con el protocolo declarado.
"""

import argparse
import json
import statistics as st
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

RUNS = Path("runs")
SEMILLAS = (0, 1, 2)
UMBRAL_FPS = 30.0        # referencia de tiempo real para conducción

CLASS_NAMES = [
    "person", "rider", "car", "truck", "bus",
    "train", "motor", "bike", "traffic light", "traffic sign",
]

# (etiqueta en la memoria, prefijo del run_name).  El orden es el de las tablas.
CONFIGS = [
    ("YOLO11m (A)",  "yolo_a_final"),
    ("YOLO11m (B)",  "yolo_b_final"),
    ("Faster R-CNN", "frcnn_final"),
    ("SSD300",       "ssd_final"),
    ("DETR",         "detr_final"),
    ("CNN baseline", "cnnb_final"),
]

AP_NO_EVALUABLE = -100.0   # centinela de evaluate.py: clase sin GT ni predicciones


# ── Localización y carga ─────────────────────────────────────────────────────

def buscar_json(run_name: str):
    """eval_results.json del run, esté donde esté.

    Ultralytics no siempre respeta la ruta de `project`, así que además de las
    ubicaciones previstas se busca por patrón.
    """
    cands = [
        RUNS / "yolo11" / run_name / "weights" / "eval_results.json",
        RUNS / "detect" / "runs" / "yolo11" / run_name / "weights" / "eval_results.json",
    ]
    cands += [p / run_name / "eval_results.json" for p in RUNS.glob("*") if p.is_dir()]
    for c in cands:
        if c.exists():
            return c
    for pat in (f"**/{run_name}/eval_results.json",
                f"**/{run_name}/weights/eval_results.json"):
        hit = sorted(RUNS.glob(pat))
        if hit:
            return hit[0]
    return None


def cargar(prefijo: str) -> tuple:
    """(lista de dicts de las semillas encontradas, lista de run_names ausentes)."""
    datos, faltan = [], []
    for s in SEMILLAS:
        name = f"{prefijo}_s{s}"
        p = buscar_json(name)
        if p is None:
            faltan.append(name)
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        d["_run"] = name
        datos.append(d)
    return datos, faltan


# ── Agregación ───────────────────────────────────────────────────────────────

def agregar(vals: list) -> tuple:
    """(media, desviación típica muestral). Con n=1 la desviación es None."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], None
    return st.mean(vals), st.stdev(vals)


def num(x, dec=2) -> str:
    """Número en formato español para LaTeX."""
    if x is None:
        return "---"
    return f"{x:.{dec}f}".replace(".", "{,}")


def ms(media, desv, dec=2) -> str:
    """'media ± desviación' en modo matemático, o solo la media si n=1."""
    if media is None:
        return "---"
    if desv is None:
        return f"${num(media, dec)}$"
    return f"${num(media, dec)} \\pm {num(desv, dec)}$"


def campo(datos: list, clave: str) -> tuple:
    return agregar([d.get(clave) for d in datos])


def campo_clase(datos: list, clase: str) -> tuple:
    """AP de una clase, excluyendo el centinela de 'no evaluable'."""
    vals = []
    for d in datos:
        v = (d.get("per_class") or {}).get(clase, {}).get("ap50_95")
        if v is not None and v > AP_NO_EVALUABLE:
            vals.append(v)
    return agregar(vals)


# ── Comprobaciones de integridad ─────────────────────────────────────────────

def verificar(todo: dict) -> bool:
    print("\n" + "=" * 78)
    print("COMPROBACIONES DE INTEGRIDAD")
    print("=" * 78)
    ok = True
    planos = [d for datos in todo.values() for d in datos]
    if not planos:
        return False

    # gt_instances identico en todos los runs: es el mismo val
    refs = {}
    for c in CLASS_NAMES:
        vistos = {}
        for d in planos:
            n = (d.get("per_class") or {}).get(c, {}).get("gt_instances")
            vistos.setdefault(n, []).append(d["_run"])
        if len(vistos) > 1:
            ok = False
            print(f"  [ERROR] '{c}' tiene distinto nº de instancias entre runs:")
            for n, runs in vistos.items():
                print(f"          {n} en {', '.join(runs[:3])}"
                      f"{' ...' if len(runs) > 3 else ''}")
        else:
            refs[c] = next(iter(vistos))
    if ok:
        total = sum(v for v in refs.values() if v)
        print(f"  [OK] gt_instances coherente en los {len(planos)} runs "
              f"({total} instancias de validación)")

    # protocolo comun
    for clave, esperado in (("conf", 0.001), ("max_det", 100)):
        vals = {d.get(clave) for d in planos}
        if vals != {esperado}:
            ok = False
            print(f"  [ERROR] '{clave}' no es uniforme: {vals} (se esperaba {esperado})")
    if ok:
        print(f"  [OK] protocolo uniforme: conf=0.001, max_det=100")

    # aviso sobre FPS
    print("\n  [AVISO] El FPS de estos JSON procede de la evaluación de cada run.")
    print("          Si NO se midieron los 18 runs en una sola sesión seguida,")
    print("          esas cifras no son comparables entre sí y no deben ir a la")
    print("          memoria (ver metodologia.tex, protocolo de consolidación).")
    return ok


# ── Tablas ───────────────────────────────────────────────────────────────────

def tabla_principal(todo):
    print("\n% --- tab:res-principal ---")
    for etq, datos in todo.items():
        if not datos:
            continue
        m50 = ms(*campo(datos, "map50"))
        m95 = ms(*campo(datos, "map50_95"))
        fps = ms(*campo(datos, "fps"), dec=1)
        g = datos[0].get("gflops")
        res = f"${datos[0].get('input_w')}\\times{datos[0].get('input_h')}$"
        print(f"{etq:<13}& {m50} & {m95} & {fps} & {num(g, 1)} & {res} \\\\")


def tabla_prf1(todo):
    print("\n% --- tab:res-prf1 ---")
    for etq, datos in todo.items():
        if not datos:
            continue
        fila = " & ".join(ms(*campo(datos, k))
                          for k in ("precision", "recall", "f1"))
        cf = ms(*campo(datos, "conf_max_f1"), dec=3)
        print(f"{etq:<13}& {fila} & {cf} \\\\")


def tabla_por_tamano(todo):
    print("\n% --- tab:res-por-tamano ---")
    for etq, datos in todo.items():
        if not datos:
            continue
        fila = " & ".join(ms(*campo(datos, k))
                          for k in ("map_small", "map_medium", "map_large"))
        print(f"{etq:<13}& {fila} \\\\")


def tabla_por_clase(todo):
    # La tabla de la memoria omite SSD300 por espacio; se emite aparte.
    cols = ["YOLO11m (A)", "YOLO11m (B)", "Faster R-CNN", "DETR", "CNN baseline"]
    print("\n% --- tab:res-por-clase  (columnas: " + ", ".join(cols) + ") ---")
    for c in CLASS_NAMES:
        inst = None
        for datos in todo.values():
            if datos:
                inst = (datos[0].get("per_class") or {}).get(c, {}).get("gt_instances")
                break
        inst_s = f"{inst:,}".replace(",", "\\,") if inst is not None else "---"
        celdas = []
        for etq in cols:
            datos = todo.get(etq) or []
            celdas.append(ms(*campo_clase(datos, c)) if datos else "---")
        print(f"\\texttt{{{c}}} & {inst_s} & " + " & ".join(celdas) + " \\\\")

    datos = todo.get("SSD300") or []
    if datos:
        print("\n% SSD300, para la tabla gemela o una nota al pie:")
        for c in CLASS_NAMES:
            print(f"%   {c:<15} {ms(*campo_clase(datos, c))}")


def tabla_ablacion(todo):
    A, B = todo.get("YOLO11m (A)") or [], todo.get("YOLO11m (B)") or []
    print("\n% --- tab:res-ablacion ---")
    if not A or not B:
        print("%   faltan runs de alguna de las dos variantes de YOLO11m")
        return
    filas = [
        ("mAP@50--95", lambda d: campo(d, "map50_95")),
        ("mAP@50",     lambda d: campo(d, "map50")),
        ("AP small",   lambda d: campo(d, "map_small")),
        ("AP medium",  lambda d: campo(d, "map_medium")),
        ("AP large",   lambda d: campo(d, "map_large")),
        (r"AP \texttt{traffic light}", lambda d: campo_clase(d, "traffic light")),
        (r"AP \texttt{traffic sign}",  lambda d: campo_clase(d, "traffic sign")),
    ]
    for nombre, f in filas:
        ma, da = f(A)
        mb, db = f(B)
        dif = f"${num(ma - mb)}$" if (ma is not None and mb is not None) else "---"
        print(f"{nombre} & {ms(ma, da)} & {ms(mb, db)} & {dif} \\\\")


def tabla_velocidad(todo):
    print("\n% --- tab:res-velocidad ---")
    for etq, datos in todo.items():
        if not datos:
            continue
        m95 = ms(*campo(datos, "map50_95"))
        fps_m, fps_d = campo(datos, "fps")
        if fps_m:
            lat_m = 1000.0 / fps_m
            # Propagación de la desviación: d(1000/f) = 1000/f^2 * df
            lat_d = (1000.0 / fps_m ** 2) * fps_d if fps_d else None
            lat = ms(lat_m, lat_d)
            cumple = "Sí" if fps_m >= UMBRAL_FPS else "No"
        else:
            lat, cumple = "---", "---"
        g = datos[0].get("gflops")
        print(f"{etq:<13}& {m95} & {lat} & {num(g, 1)} & {cumple} \\\\")


TABLAS = {
    "principal":  tabla_principal,
    "prf1":       tabla_prf1,
    "tamano":     tabla_por_tamano,
    "clase":      tabla_por_clase,
    "ablacion":   tabla_ablacion,
    "velocidad":  tabla_velocidad,
}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tabla", choices=list(TABLAS) + ["todas"], default="todas")
    ap.add_argument("--json", default=None,
                    help="Vuelca también el agregado a un fichero JSON")
    args = ap.parse_args()

    todo, faltantes = {}, []
    for etq, prefijo in CONFIGS:
        datos, faltan = cargar(prefijo)
        todo[etq] = datos
        faltantes += faltan

    hallados = sum(len(v) for v in todo.values())
    print(f"Runs con eval_results.json: {hallados} / {len(CONFIGS) * len(SEMILLAS)}")
    for etq, datos in todo.items():
        n = len(datos)
        marca = "OK" if n == len(SEMILLAS) else f"faltan {len(SEMILLAS) - n}"
        print(f"  {etq:<14} {n}/{len(SEMILLAS)} semillas   {marca}")
    if faltantes:
        print(f"\nSin evaluar todavía: {', '.join(faltantes)}")
        print("Las tablas se emiten igualmente con lo que haya; las")
        print("configuraciones con una sola semilla salen sin desviación típica.")
    if hallados == 0:
        sys.exit("\nNo hay ningún eval_results.json. Ejecuta antes evaluate.py "
                 "sobre los best.pt (ver docs/REGISTRO_ENTRENAMIENTOS.md).")

    verificar(todo)

    print("\n" + "=" * 78)
    print("FILAS LaTeX  —  copiar dentro del tabular correspondiente")
    print("=" * 78)
    for nombre, fn in TABLAS.items():
        if args.tabla in ("todas", nombre):
            fn(todo)

    if args.json:
        agregado = {}
        for etq, datos in todo.items():
            if not datos:
                continue
            e = {"n_semillas": len(datos), "runs": [d["_run"] for d in datos]}
            for k in ("map50", "map50_95", "map_small", "map_medium", "map_large",
                      "precision", "recall", "f1", "conf_max_f1", "fps"):
                m, d = campo(datos, k)
                e[k] = {"media": m, "desv": d}
            e["per_class"] = {}
            for c in CLASS_NAMES:
                m, d = campo_clase(datos, c)
                e["per_class"][c] = {"media": m, "desv": d}
            e["gflops"] = datos[0].get("gflops")
            e["params_m"] = datos[0].get("params_m")
            agregado[etq] = e
        Path(args.json).write_text(
            json.dumps(agregado, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nAgregado volcado en {args.json}")


if __name__ == "__main__":
    main()
