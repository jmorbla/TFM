"""
Curva de aprendizaje: ¿cuántas imágenes de entrenamiento hacen falta?

Entrena el mismo modelo con subsets ANIDADOS de 8k / 16k / 32k imágenes,
evaluando siempre sobre el MISMO conjunto de validación (2 000 imágenes), y
registra el mAP alcanzado y el coste. El objetivo es ver dónde satura la curva
para elegir con datos el tamaño del split de la comparativa principal.

Por qué un script y no un notebook: esto son entre 18 y 52 h de GPU. Un kernel
de Jupyter abierto durante días es frágil (se cae la conexión, el portátil
suspende, se pierde el estado). Este script es RESUMIBLE: cada run completado
queda anotado en resultados.json y, si lo relanzas, se salta lo ya hecho.
El análisis y las gráficas van aparte, en analisis.ipynb, que solo lee el JSON.

Uso (desde TFM/):
    python experiments/curva_aprendizaje/run_curve.py --models yolo11m
    python experiments/curva_aprendizaje/run_curve.py --models yolo11m cnn_baseline
    python experiments/curva_aprendizaje/run_curve.py --dry_run     # solo plan
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

EXP = Path("experiments/curva_aprendizaje")
DATASETS = EXP / "datasets"
RUNS = EXP / "runs"
RESULTS = EXP / "resultados.json"

SIZES = [8000, 16000, 32000]

# Épocas por tamaño. NO son constantes: al multiplicar los datos hay que reducir
# las épocas para que el nº de imágenes vistas quede en el mismo orden y la
# comparación no confunda "más datos" con "más cómputo". No se dividen
# exactamente por k porque más datos únicos requieren algo más de recorrido.
EPOCHS = {
    "yolo11m":      {8000: 70, 16000: 50, 32000: 36},
    "cnn_baseline": {8000: 85, 16000: 58, 32000: 41},
}
# NOTA sobre el techo y el early stopping: con patience=15, un techo ajustado
# impide que el early stopping llegue a dispararse en los subsets grandes, y el
# punto terminaría infraentrenado — sesgando la curva EN CONTRA de "más datos".
# Por eso eval_curve.py registra la época real de parada: si epochs_run ==
# epochs_cap, ese punto NO convergió y su mAP es una cota inferior.
PATIENCE = 15          # idéntico en todos los puntos de la curva
SEED = 0               # una sola semilla: interesa la FORMA de la curva


def tag(size: int) -> str:
    return f"{size // 1000}k"


def run_name(model: str, size: int) -> str:
    return f"curva_{model}_{tag(size)}"


def load_results() -> dict:
    if RESULTS.exists():
        return json.loads(RESULTS.read_text(encoding="utf-8"))
    return {}


def save_results(d: dict):
    RESULTS.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")


def build_cmd(model: str, size: int) -> list:
    """Comando de entrenamiento para un punto de la curva."""
    name = run_name(model, size)
    epochs = EPOCHS[model][size]

    if model == "yolo11m":
        # Ruta ABSOLUTA: con una relativa, Ultralytics la resuelve contra su
        # settings['runs_dir'] y escribe en runs/detect/<lo que le pases>/,
        # mezclando las salidas del experimento con las del proyecto principal.
        return [sys.executable, "models/yolo11/train.py",
                "--config", "configs/yolo11m.yaml",
                "--data", str(DATASETS / f"yolo_{tag(size)}.yaml"),
                "--runs_dir", str((RUNS / "yolo11m").resolve()),
                "--run_name", name, "--seed", str(SEED)]

    # cnn_baseline pasa por train.py, que lee las rutas del YAML del proyecto.
    # Lo genera build_learning_curve.py junto al dataset (no aquí), porque
    # evaluate.py también lo necesita — incluso para YOLO.
    data_yaml = DATASETS / f"bdd100k_{tag(size)}.yaml"
    if not data_yaml.exists():
        sys.exit(f"Falta {data_yaml}. Ejecuta antes:\n"
                 f"  python data/build_learning_curve.py --yaml_only")

    return [sys.executable, "train.py", "--model", "cnn_baseline",
            "--config", "configs/cnn_baseline.yaml",
            "--data", str(data_yaml),
            "--runs_dir", str(RUNS),
            "--run_name", name, "--seed", str(SEED),
            "--patience", str(PATIENCE)]


def epochs_override(model: str, size: int) -> Path | None:
    """Duplica el config del modelo con las épocas de este punto de la curva."""
    import yaml as _yaml
    src = Path("configs/yolo11m.yaml" if model == "yolo11m"
               else "configs/cnn_baseline.yaml")
    cfg = _yaml.safe_load(src.read_text(encoding="utf-8"))
    cfg["training"]["epochs"] = EPOCHS[model][size]
    if model == "yolo11m":
        cfg["training"]["patience"] = PATIENCE
    dst = DATASETS / f"cfg_{model}_{tag(size)}.yaml"
    dst.write_text(_yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                   encoding="utf-8")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["yolo11m"],
                    choices=["yolo11m", "cnn_baseline"])
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--dry_run", action="store_true", help="Solo imprime el plan")
    ap.add_argument("--force", action="store_true", help="Reejecuta runs ya hechos")
    args = ap.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    res = load_results()

    plan = [(m, s) for m in args.models for s in sorted(args.sizes)]
    print(f"{'='*72}\n  PLAN: {len(plan)} entrenamientos\n{'='*72}")
    for m, s in plan:
        k = run_name(m, s)
        estado = "YA HECHO" if k in res and not args.force else "pendiente"
        print(f"  {k:<28} {EPOCHS[m][s]:>3} épocas  {estado}")
    if args.dry_run:
        return

    for m, s in plan:
        k = run_name(m, s)
        if k in res and not args.force:
            print(f"\n[skip] {k} ya completado ({res[k].get('map50_95')}% mAP)")
            continue

        cfg = epochs_override(m, s)
        cmd = build_cmd(m, s)
        # sustituye el config base por el que lleva las épocas de este punto
        cmd[cmd.index("--config") + 1] = str(cfg)

        print(f"\n{'='*72}\n  {k}  —  {s} imágenes, {EPOCHS[m][s]} épocas"
              f"\n  {' '.join(cmd)}\n{'='*72}")
        t0 = time.time()
        r = subprocess.run(cmd)
        dur = (time.time() - t0) / 3600

        if r.returncode != 0:
            print(f"  FALLÓ (código {r.returncode}). No se anota; relanza para reintentar.")
            continue

        res[k] = {"model": m, "train_size": s, "epochs_cap": EPOCHS[m][s],
                  "seed": SEED, "horas": round(dur, 2),
                  "fecha": datetime.now().isoformat(timespec="seconds")}
        save_results(res)
        print(f"  Completado en {dur:.2f} h  ->  {RESULTS}")

    print(f"\n{'='*72}")
    print(f"  {len(res)} runs registrados en {RESULTS}")
    print("  Siguiente paso: evaluar con eval_curve.py y analizar en analisis.ipynb")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()

