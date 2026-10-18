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
import re
import subprocess
import sys

import yaml
from pathlib import Path

EXP = Path("experiments/curva_aprendizaje")
DATASETS = EXP / "datasets"
RUNS = EXP / "runs"
RESULTS = EXP / "resultados.json"

# La tabla resumen usa 'Δ' y la consola de Windows arranca en cp1252, lo que
# hacia reventar el print con UnicodeEncodeError DESPUES de haber evaluado.
# Se fuerza UTF-8 en stdout/stderr; si el terminal no lo admite, se degrada a
# reemplazar el caracter en vez de abortar.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass



def weights_for(model: str, name: str) -> Path | None:
    """Localiza el best.pt del run.

    Ultralytics NO respeta una ruta relativa en `project`: la resuelve contra su
    propio settings['runs_dir'] y acaba escribiendo en
    `runs/detect/<project>/<name>/weights/best.pt`. Por eso se prueban varias
    ubicaciones y, como último recurso, se busca por patrón: así funciona tanto
    con los runs ya entrenados como con los que se lancen en el futuro.
    """
    cands = [
        RUNS / "yolo11m" / name / "weights" / "best.pt",           # ruta pretendida
        Path("runs/detect") / RUNS / "yolo11m" / name / "weights" / "best.pt",
        Path("runs") / RUNS / "yolo11m" / name / "weights" / "best.pt",
        RUNS / model / name / "best.pt",                            # train.py propio
    ]
    hit = next((c for c in cands if c.exists()), None)
    if hit:
        return hit
    # Reserva: buscar el run por nombre allá donde haya caído
    for pat in (f"**/{name}/weights/best.pt", f"**/{name}/best.pt"):
        found = sorted(Path("runs").glob(pat)) + sorted(RUNS.glob(pat))
        if found:
            return found[0]
    return None




def descubrir_huerfanos(res: dict, adoptar: bool) -> int:
    """Localiza runs con best.pt en disco que no estén en resultados.json.

    run_curve.py registra un run SOLO cuando el entrenamiento termina, porque
    necesita la duración para el campo `horas`. Un run interrumpido -- Ctrl-C,
    divergencia a NaN, corte de luz -- deja su best.pt en disco pero queda
    invisible para este script, que itera sobre resultados.json.

    No se adoptan solos a propósito: un run interrumpido es una cota inferior y
    entra en la curva solo si lo decides. Con --adoptar se registran.
    """
    vistos, huerfanos = set(), []
    for ck in sorted(RUNS.glob("**/best.pt")):
        run_dir = ck.parent.parent if ck.parent.name == "weights" else ck.parent
        name = run_dir.name
        if name in res or name in vistos:
            continue
        vistos.add(name)
        huerfanos.append((name, run_dir))

    if not huerfanos:
        return 0

    print("")
    print(f"{len(huerfanos)} run(s) con best.pt en disco SIN entrada en resultados.json:")
    adoptados = 0
    for name, run_dir in huerfanos:
        m = re.match(r"curva_(.+)_(\d+)k$", name)
        if not m:
            print(f"  [ignorado] {name}: el nombre no encaja con curva_<modelo>_<N>k")
            continue
        model, size = m.group(1), int(m.group(2)) * 1000
        cap = None
        cfg = run_dir / "model_config.yaml"
        if cfg.exists():
            try:
                cap = yaml.safe_load(cfg.read_text(encoding="utf-8"))["training"]["epochs"]
            except Exception:
                pass
        if not adoptar:
            print(f"  [huerfano] {name}  ({model}, {size} imgs, techo {cap})"
                  f"  -> registralo con --adoptar")
            continue
        res[name] = {"model": model, "train_size": size, "epochs_cap": cap,
                     "seed": 0, "horas": None, "fecha": None,
                     "interrumpido": True}
        adoptados += 1
        print(f"  [adoptado] {name}  ({model}, {size} imgs, techo {cap})")
    if adoptados:
        print("  AVISO: son runs interrumpidos. Su mAP es una COTA INFERIOR y")
        print("  los campos derivados de last.pt pueden ser basura si divergieron.")
    return adoptados

MESETA_PP = 0.3   # ganancia (pp de mAP) en las últimas 10 épocas para considerar meseta


def epocas_reales(model: str, w: Path) -> tuple:
    """(épocas ejecutadas, mejor época, ganancia en las últimas 10 épocas).

    Agotar el techo NO implica estar infraentrenado: un run puede llegar al
    final ya en meseta sin que el early stopping haya llegado a saltar (el
    `patience` de Ultralytics cuenta sobre su propia métrica de fitness, que
    puede seguir moviéndose por milésimas). Por eso, en vez de mirar solo el
    contador de épocas, se mide la TENDENCIA real de la curva de mAP.
    """
    import csv as _csv
    import torch
    run_dir = w.parent.parent if model == "yolo11m" else w.parent
    ejecutadas = best = tendencia = None
    try:
        best = int(torch.load(w, map_location="cpu", weights_only=False)["epoch"]) + 1
    except Exception:
        pass

    if model == "yolo11m":
        f = run_dir / "results.csv"
        if f.exists():
            rows = [{k.strip(): v for k, v in r.items()}
                    for r in _csv.DictReader(f.open(encoding="utf-8"))]
            serie = [float(r["metrics/mAP50-95(B)"]) * 100 for r in rows]
            ejecutadas = len(serie)
            if len(serie) > 11:
                tendencia = serie[-1] - serie[-11]
    else:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        last = run_dir / "last.pt"
        if last.exists():
            try:
                ejecutadas = int(torch.load(last, map_location="cpu",
                                            weights_only=False)["epoch"]) + 1
            except Exception:
                pass
        tb = sorted(run_dir.glob("tb/events*"))
        if tb:
            try:
                ea = EventAccumulator(str(tb[0])); ea.Reload()
                if "mAP/val_50-95" in ea.Tags()["scalars"]:
                    serie = [s.value * 100 for s in ea.Scalars("mAP/val_50-95")]
                    if len(serie) > 11:
                        tendencia = serie[-1] - serie[-11]
            except Exception:
                pass
    return ejecutadas, best, tendencia


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Reevalúa runs que ya tienen métricas")
    ap.add_argument("--adoptar", action="store_true",
                    help="Registra en la curva los runs interrumpidos que"
                         " tengan best.pt en disco pero no entrada en el JSON")
    args = ap.parse_args()

    if not RESULTS.exists():
        sys.exit(f"No existe {RESULTS}. Ejecuta antes run_curve.py")
    res = json.loads(RESULTS.read_text(encoding="utf-8"))

    descubrir_huerfanos(res, args.adoptar)

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

        ejecutadas, best_ep, tend = epocas_reales(model, w)
        info["epochs_run"] = ejecutadas
        info["best_epoch"] = best_ep
        info["ganancia_ult10ep"] = None if tend is None else round(tend, 2)
        info["imgs_vistas"] = (ejecutadas or 0) * size
        # Convergió si paró antes del techo (saltó el early stopping) o si,
        # habiendo agotado el techo, la curva ya estaba en meseta.
        info["convergio"] = bool(
            (ejecutadas is not None and ejecutadas < info["epochs_cap"])
            or (tend is not None and tend < MESETA_PP)
        )
        RESULTS.write_text(json.dumps(res, indent=2, ensure_ascii=False),
                           encoding="utf-8")
        print(f"  mAP@50-95 = {info['map50_95']}%")

    print(f"\n{'='*100}")
    print(f"{'run':<24}{'imgs':>7}{'épocas':>9}{'img-vistas':>12}{'mAP50':>8}"
          f"{'mAP50-95':>10}{'small':>7}{'Δ10ep':>8}{'horas':>7}  estado")
    print("-" * 100)
    for n, i in sorted(res.items(), key=lambda kv: (kv[1]["model"], kv[1]["train_size"])):
        if "map50_95" not in i:
            continue
        ep = f"{i.get('epochs_run', '?')}/{i['epochs_cap']}"
        d = i.get("ganancia_ult10ep")
        estado = "en meseta" if i.get("convergio") else "AÚN SUBÍA — cota inferior"
        print(f"{n:<24}{i['train_size']:>7}{ep:>9}{i.get('imgs_vistas', 0):>12,}"
              f"{i['map50']:>8}{i['map50_95']:>10}{i.get('map_small', '-'):>7}"
              f"{'   n/a' if d is None else f'{d:+7.2f}'}{i.get('horas', 0):>7.1f}  {estado}")
    print("=" * 100)
    if any("map50_95" in i and not i.get("convergio") for i in res.values()):
        print("AVISO: los puntos marcados 'AÚN SUBÍA' seguían mejorando al agotar el")
        print("techo. Su mAP es una COTA INFERIOR y la curva subestima ahí el")
        print("beneficio de más datos. Súbeles el techo y relanza con --force.")
    print("\nOJO con img-vistas: si no es aproximadamente constante entre puntos, la")
    print("comparación mezcla 'más datos' con 'más cómputo' y hay que decirlo.")
    print("Analiza la curva en experiments/curva_aprendizaje/analisis.ipynb")


if __name__ == "__main__":
    main()
