"""
Evalúa todos los runs de la curva de aprendizaje con el protocolo común
(evaluate.py: conf=0.001, max_det=100, métricas en píxeles nativos) y consolida
los resultados en resultados.json.

Separado de run_curve.py a propósito: el entrenamiento son horas y la
evaluación minutos. Si algo falla en la evaluación no hay que reentrenar.

Uso (desde TFM/):
    python experiments/curva_aprendizaje/eval_curve.py
    python experiments/curva_aprendizaje/eval_curve.py --force
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

EXP = Path("experiments/curva_aprendizaje")
DATASETS = EXP / "datasets"
RUNS = EXP / "runs"
RESULTS = EXP / "resultados.json"


def weights_for(model: str, name: str) -> Path | None:
    """Ubicación del best.pt según el modelo (Ultralytics anida en weights/)."""
    cands = [RUNS / "yolo11m" / name / "weights" / "best.pt",
             RUNS / model / name / "best.pt"]
    return next((c for c in cands if c.exists()), None)


def epocas_reales(model: str, w: Path) -> tuple:
    """(épocas ejecutadas, mejor época) del run.

    Dato imprescindible para leer la curva: si las épocas ejecutadas coinciden
    con el techo, el early stopping NO llegó a saltar y ese punto está
    infraentrenado — su mAP es una cota inferior, no su rendimiento real.
    """
    import torch
    run_dir = w.parent.parent if model == "yolo11m" else w.parent
    ejecutadas = best = None
    try:
        best = int(torch.load(w, map_location="cpu", weights_only=False)["epoch"]) + 1
    except Exception:
        pass
    if model == "yolo11m":
        csv = run_dir / "results.csv"
        if csv.exists():
            ejecutadas = sum(1 for _ in csv.open(encoding="utf-8")) - 1
    else:
        last = run_dir / "last.pt"
        if last.exists():
            try:
                ejecutadas = int(torch.load(last, map_location="cpu",
                                            weights_only=False)["epoch"]) + 1
            except Exception:
                pass
    return ejecutadas, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Reevalúa runs que ya tienen métricas")
    args = ap.parse_args()

    if not RESULTS.exists():
        sys.exit(f"No existe {RESULTS}. Ejecuta antes run_curve.py")
    res = json.loads(RESULTS.read_text(encoding="utf-8"))

    for name, info in res.items():
        if "map50_95" in info and not args.force:
            print(f"[skip] {name} ya evaluado ({info['map50_95']}%)")
            continue

        model, size = info["model"], info["train_size"]
        w = weights_for(model, name)
        if w is None:
            print(f"[falta] {name}: no se encuentra best.pt")
            continue

        tag = f"{size // 1000}k"
        cmd = [sys.executable, "evaluate.py", "--weights", str(w)]
        if model == "yolo11m":
            cmd += ["--model", "yolo11", "--config", "configs/yolo11m.yaml",
                    "--data", str(DATASETS / f"bdd100k_{tag}.yaml")]
        else:
            cmd += ["--model", "cnn_baseline",
                    "--config", "configs/cnn_baseline.yaml",
                    "--data", str(DATASETS / f"bdd100k_{tag}.yaml")]

        print(f"\n=== Evaluando {name} ===\n{' '.join(cmd)}")
        if subprocess.run(cmd).returncode != 0:
            print(f"  FALLÓ la evaluación de {name}")
            continue

        ev = json.loads((w.parent / "eval_results.json").read_text(encoding="utf-8"))
        # Todas las métricas del protocolo + los parámetros con los que se
        # obtuvieron, para que resultados.json sea autocontenido.
        info.update({k: ev.get(k) for k in (
            "map50", "map50_95",                          # mAP (10 clases)
            "map_small", "map_medium", "map_large",       # AP por tamaño
            "precision", "recall", "f1", "conf_max_f1",   # punto de máximo F1
            "pr_iou", "fps", "img_size", "conf", "max_det",
        )})
        info["per_class"] = ev.get("per_class")           # AP + instancias de GT

        ejecutadas, best_ep = epocas_reales(model, w)
        info["epochs_run"] = ejecutadas
        info["best_epoch"] = best_ep
        # Si se agotó el techo, el early stopping no actuó y el punto no convergió
        info["convergio"] = (ejecutadas is not None
                             and ejecutadas < info["epochs_cap"])
        RESULTS.write_text(json.dumps(res, indent=2, ensure_ascii=False),
                           encoding="utf-8")
        print(f"  mAP@50-95 = {info['map50_95']}%")

    print(f"\n{'='*86}")
    print(f"{'run':<26}{'imgs':>7}{'épocas':>10}{'mAP50':>8}{'mAP50-95':>10}"
          f"{'small':>7}{'horas':>7}  estado")
    print("-" * 86)
    for n, i in sorted(res.items(), key=lambda kv: (kv[1]["model"], kv[1]["train_size"])):
        if "map50_95" not in i:
            continue
        ep = f"{i.get('epochs_run', '?')}/{i['epochs_cap']}"
        estado = "convergió" if i.get("convergio") else "TOPE — no convergió"
        print(f"{n:<26}{i['train_size']:>7}{ep:>10}{i['map50']:>8}{i['map50_95']:>10}"
              f"{i.get('map_small', '-'):>7}{i.get('horas', 0):>7.1f}  {estado}")
    print("=" * 86)
    if any("map50_95" in i and not i.get("convergio") for i in res.values()):
        print("AVISO: los puntos marcados 'TOPE' agotaron el techo de épocas sin que")
        print("saltara el early stopping. Su mAP es una COTA INFERIOR: la curva")
        print("subestima ahí el beneficio de más datos. Súbeles el techo y relanza")
        print("con --force si quieres el valor real.")
    print("Analiza la curva en experiments/curva_aprendizaje/analisis.ipynb")


if __name__ == "__main__":
    main()
