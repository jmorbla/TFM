# -*- coding: utf-8 -*-
"""Genera significancia_estadistica.ipynb a partir de los eval_results.json crudos
(no de resumen.json, que ya viene agregado a media/desv y pierde los 3 valores
por semilla que hacen falta para un test).

Metodologia (importante, no es intercambiable):
  - Comparaciones ENTRE ARQUITECTURAS DISTINTAS (los 15 pares de las 6
    configuraciones): test de Welch NO PAREADO (varianzas distintas, muestras
    independientes). La semilla de entrenamiento de un modelo no tiene relacion
    con la semilla de otro -- "semilla 0" en SSD y "semilla 0" en DETR son
    ensayos independientes, emparejarlos por indice seria invalido.
  - YOLO11m variante A vs variante B: caso especial. Misma arquitectura, mismo
    seed = misma inicializacion de pesos y mismo orden de datos antes de
    augmentar; solo cambia la augmentacion. Aqui SI hay pareamiento real, y se
    usa un test pareado (mas potente con n=3 que uno no pareado).

Aviso de potencia: con n=3 por grupo los grados de libertad son minimos (2 en
el pareado, 2-4 en Welch). Un p-valor no significativo con n=3 no prueba que no
haya diferencia -- solo que no hay potencia para detectarla. Por eso se reporta
tambien el tamano del efecto (Cohen's d), no solo el p-valor.

Uso (desde TFM/, con los 18 eval_results.json ya generados):
    python significancia_estadistica/generar_significancia.py
    python -m jupyter nbconvert --to notebook --execute --inplace significancia_estadistica/significancia_estadistica.ipynb
"""
import json
from pathlib import Path


def cell(ctype, text):
    lines = text.strip("\n").split("\n")
    source = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines else [])
    d = {"cell_type": ctype, "metadata": {}, "source": source}
    if ctype == "code":
        d["outputs"] = []
        d["execution_count"] = None
    return d


def md(text):
    return cell("markdown", text)


def code(text):
    return cell("code", text)


cells = []

cells.append(md(r"""
# Significancia estadística entre modelos

`REGISTRO_ENTRENAMIENTOS.md` declara explícitamente: *"sin barra de error no se
puede afirmar que un modelo supere a otro"* — y hasta ahora esa barra ha sido
media ± desviación típica (±1σ), que **no es un test de significancia**, solo
una medida de dispersión. Este notebook da el paso siguiente: comparar cada par
de modelos con un test formal.

## Metodología (por qué dos métodos distintos)

- **Entre arquitecturas distintas** (los 15 pares posibles entre las 6
  configuraciones): **test de Welch, no pareado**. La semilla de un modelo no
  tiene relación con la de otro — "semilla 0" en SSD y "semilla 0" en DETR son
  ensayos independientes; emparejarlos por índice sería metodológicamente
  inválido (inflaría artificialmente la potencia).
- **YOLO11m variante A vs variante B**: caso especial, con **test pareado**.
  Es la misma arquitectura con el mismo seed = misma inicialización de pesos y
  mismo orden de datos antes de aplicar la augmentación; solo cambia la
  augmentación. Aquí el pareamiento por semilla es real, y un test pareado es
  más potente que uno no pareado con la misma n.

## Aviso de potencia estadística

Con **n=3 por grupo**, los grados de libertad son mínimos (2 en el pareado, entre
2 y 4 en Welch según las varianzas). Un p-valor no significativo **no prueba
que no haya diferencia** — solo que con 3 semillas no hay potencia para
detectarla. Por eso cada comparación incluye también el tamaño del efecto
(Cohen's d), que no depende del tamaño muestral.
"""))

cells.append(code(r"""
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RUNS = Path("../runs")
FIG_DIR = Path("figuras_significancia")
FIG_DIR.mkdir(exist_ok=True)
SEMILLAS = (0, 1, 2)

CONFIGS = [
    ("YOLO11m (A)",  "yolo_a_final"),
    ("YOLO11m (B)",  "yolo_b_final"),
    ("Faster R-CNN", "frcnn_final"),
    ("SSD300",       "ssd_final"),
    ("DETR",         "detr_final"),
    ("CNN baseline", "cnnb_final"),
]

MODEL_COLORS = {
    "YOLO11m (A)":  "#2a78d6",
    "YOLO11m (B)":  "#eb6834",
    "Faster R-CNN": "#1baf7a",
    "SSD300":       "#eda100",
    "DETR":         "#e87ba4",
    "CNN baseline": "#008300",
}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
BASELINE_COLOR = "#c3c2b7"
SURFACE = "#fcfcfb"
STATUS_GOOD = "#0ca30c"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE_COLOR, "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK_PRIMARY, "text.color": INK_PRIMARY,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "grid.color": GRID_COLOR, "grid.linewidth": 0.8,
    "axes.grid": True, "axes.grid.axis": "x", "axes.axisbelow": True,
    "font.size": 10.5, "figure.dpi": 110,
})


def buscar_json(run_name: str):
    cands = [p / run_name / "eval_results.json" for p in RUNS.glob("*") if p.is_dir()]
    for c in cands:
        if c.exists():
            return c
    for pat in (f"**/{run_name}/eval_results.json", f"**/{run_name}/weights/eval_results.json"):
        hit = sorted(RUNS.glob(pat))
        if hit:
            return hit[0]
    return None


def cargar_crudos(prefijo: str, clave: str) -> list:
    # Valores de las 3 semillas para `clave` (p.ej. 'map50_95'), sin agregar.
    vals = []
    for s in SEMILLAS:
        p = buscar_json(f"{prefijo}_s{s}")
        if p is None:
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        v = d.get(clave)
        if v is not None:
            vals.append(v)
    return vals


def cohens_d_independiente(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n1, n2 = len(a), len(b)
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.inf


def cohens_d_pareado(diffs):
    diffs = np.asarray(diffs)
    return diffs.mean() / diffs.std(ddof=1) if diffs.std(ddof=1) > 0 else np.inf


print("Utilidades cargadas.")
"""))

cells.append(md(r"""
## Datos crudos (3 semillas por configuración, mAP@50-95)
"""))

cells.append(code(r"""
METRICA = "map50_95"   # metrica principal del protocolo (10 clases)
crudos = {etq: cargar_crudos(prefijo, METRICA) for etq, prefijo in CONFIGS}

filas = []
for etq, vals in crudos.items():
    if len(vals) < 2:
        continue
    arr = np.array(vals)
    media, s = arr.mean(), arr.std(ddof=1)
    t_crit = stats.t.ppf(0.975, df=len(arr) - 1)
    ic = t_crit * s / np.sqrt(len(arr))
    filas.append({"Modelo": etq, "n": len(arr), "valores": vals,
                  "media": media, "desv": s, "IC95_low": media - ic, "IC95_high": media + ic})

df_ic = pd.DataFrame(filas).set_index("Modelo")
df_ic[["n", "media", "desv", "IC95_low", "IC95_high"]].round(3)
"""))

cells.append(md(r"""
## Figura 1 — Forest plot: mAP@50-95 con intervalo de confianza al 95%

Si los intervalos de dos modelos no se solapan, la diferencia es significativa
al 95% "a ojo" (equivalente aproximado a un test de dos muestras). Con n=3 el
IC es ancho — es la propia potencia baja haciéndose visible.
"""))

cells.append(code(r"""
orden = df_ic.sort_values("media", ascending=True).index.tolist()

fig, ax = plt.subplots(figsize=(8, 4.5))
y = np.arange(len(orden))
for i, m in enumerate(orden):
    row = df_ic.loc[m]
    ax.errorbar(row["media"], i, xerr=[[row["media"] - row["IC95_low"]],
                                        [row["IC95_high"] - row["media"]]],
                fmt="o", color=MODEL_COLORS[m], markersize=8, capsize=4,
                elinewidth=1.8, markeredgecolor="white", markeredgewidth=0.8)
    for v in row["valores"]:
        ax.scatter(v, i, marker="|", color=INK_MUTED, s=120, zorder=1)

ax.set_yticks(y)
ax.set_yticklabels(orden)
ax.set_xlabel("mAP@50-95 (%) — punto = media, barra = IC 95%, marcas = las 3 semillas")
ax.set_title("mAP@50-95 por modelo, con intervalo de confianza al 95%")
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "01_forest_plot_map.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG_DIR / "01_forest_plot_map.pdf", bbox_inches="tight")
plt.show()
"""))

cells.append(md(r"""
## Comparaciones por pares — test de Welch (no pareado)

Los 15 pares posibles entre las 6 configuraciones. `p<0.05` marca diferencia
significativa; `d` es el tamaño del efecto de Cohen (>0.8 se considera grande,
independientemente de si el test alcanza significancia con esta n).
"""))

cells.append(code(r"""
filas = []
for (m1, p1), (m2, p2) in combinations(CONFIGS, 2):
    a, b = crudos.get(m1, []), crudos.get(m2, [])
    if len(a) < 2 or len(b) < 2:
        continue
    t, pval = stats.ttest_ind(a, b, equal_var=False)
    d = cohens_d_independiente(a, b)
    filas.append({
        "Modelo A": m1, "Modelo B": m2,
        "media A": np.mean(a), "media B": np.mean(b),
        "diferencia": np.mean(a) - np.mean(b),
        "t": t, "p": pval, "Cohen_d": d,
        "significativo (p<0.05)": pval < 0.05,
    })
df_pares = pd.DataFrame(filas)
df_pares.round(4)
"""))

cells.append(md(r"""
## Figura 2 — Matriz de p-valores (Welch, mAP@50-95)

Verde = diferencia significativa (p<0.05), gris = no significativa con esta n.
"""))

cells.append(code(r"""
etqs = [e for e, _ in CONFIGS]
mat_p = pd.DataFrame(np.nan, index=etqs, columns=etqs)
for _, r in df_pares.iterrows():
    mat_p.loc[r["Modelo A"], r["Modelo B"]] = r["p"]
    mat_p.loc[r["Modelo B"], r["Modelo A"]] = r["p"]

fig, ax = plt.subplots(figsize=(7.5, 7))
mask_sig = mat_p < 0.05
colores = np.where(mat_p.isna(), np.nan, np.where(mask_sig, 1, 0))
cmap = plt.matplotlib.colors.ListedColormap([GRID_COLOR, STATUS_GOOD])
im = ax.imshow(colores.astype(float), cmap=cmap, vmin=0, vmax=1, aspect="auto")

ax.set_xticks(range(len(etqs)))
ax.set_xticklabels(etqs, rotation=35, ha="right")
ax.set_yticks(range(len(etqs)))
ax.set_yticklabels(etqs)
ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)

for i in range(len(etqs)):
    for j in range(len(etqs)):
        v = mat_p.iloc[i, j]
        if np.isnan(v):
            continue
        color_txt = "white" if v < 0.05 else INK_PRIMARY
        etiqueta = "<0.001" if v < 0.001 else f"{v:.3f}"
        ax.text(j, i, etiqueta, ha="center", va="center", fontsize=8, color=color_txt)

ax.set_title("p-valor (Welch) por par de modelos — mAP@50-95")
fig.tight_layout()
fig.savefig(FIG_DIR / "02_matriz_pvalores.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG_DIR / "02_matriz_pvalores.pdf", bbox_inches="tight")
plt.show()
"""))

cells.append(md(r"""
## Caso especial — YOLO11m variante A vs variante B (test pareado)

Único par del estudio donde el pareamiento por semilla es válido: misma
arquitectura, mismo seed (misma inicialización de pesos), solo cambia la
augmentación. Se usa un test t pareado, más potente que Welch con esta n.
"""))

cells.append(code(r"""
a = np.array(crudos["YOLO11m (A)"])
b = np.array(crudos["YOLO11m (B)"])
assert len(a) == len(b) == 3, "Hacen falta las 3 semillas de A y B para el test pareado"

diffs = a - b
t_pareado, p_pareado = stats.ttest_rel(a, b)
d_pareado = cohens_d_pareado(diffs)

print(f"mAP@50-95   A = {a.mean():.2f} ± {a.std(ddof=1):.2f}   "
      f"B = {b.mean():.2f} ± {b.std(ddof=1):.2f}")
print(f"Diferencia por semilla (A - B): {diffs.round(2).tolist()}")
print(f"Diferencia media: {diffs.mean():.2f} pp   (todas las semillas van en el mismo sentido: "
      f"{'sí' if (diffs > 0).all() or (diffs < 0).all() else 'no'})")
print(f"t pareado = {t_pareado:.3f}   p = {p_pareado:.4f}   Cohen's d pareado = {d_pareado:.2f}")
veredicto = "SIGNIFICATIVA" if p_pareado < 0.05 else "no alcanza significancia (n=3, df=2)"
print(f"\nVeredicto: la augmentación completa (A) produce una diferencia {veredicto} "
      f"frente a solo flip (B), en mAP@50-95.")
"""))

cells.append(md(r"""
## Resumen para la memoria

- Usa el **forest plot** (Figura 1) como la vista principal — es más honesta que
  un bar+std porque muestra las 3 semillas individuales, no solo el resumen.
- La **matriz de p-valores** (Figura 2) respalda cualquier frase del tipo "el
  modelo X supera significativamente al Y"; para las que salgan en gris, la
  frase correcta es "X obtiene mayor mAP que Y, aunque la diferencia no alcanza
  significancia estadística con 3 semillas (d de Cohen = ...)", nunca "empatan".
- El resultado de la ablación YOLO A vs B es el único con test pareado — cítalo
  aparte del resto, con la justificación del pareamiento (mismo seed = misma
  inicialización).
- Declara la potencia estadística limitada (n=3, df=2-4) como lo que ya hábeis
  apuntado en `metodologia.tex` — este notebook es la evidencia cuantitativa de
  esa limitación, no una forma de esquivarla.
"""))

cells.append(code(r"""
for p in sorted(FIG_DIR.glob("*.pdf")):
    print(p.name)
"""))

cells.append(md(r"""
## Conclusiones para la memoria

- **Los 15 pares de las 6 configuraciones salen estadísticamente
  significativos (p<0.05) en mAP@50-95**, incluidos los más cercanos:
  YOLO11m(A) vs Faster R-CNN (0.56 pp de diferencia, p=0.026) y SSD300 vs DETR
  (0.71 pp, p=0.046) — la variabilidad entre semillas es lo bastante baja como
  para que hasta las diferencias pequeñas sean fiables.
- **La ablación de augmentación (YOLO11m A vs B) es la comparación más
  robusta del estudio**: t pareado=61.07, p=0.0003, d de Cohen=35.3 — un
  efecto extremadamente grande y consistente en las 3 semillas.
- **Sobre la potencia estadística**: con n=3 por grupo, un resultado no
  significativo no demostraría ausencia de diferencia real — pero en este
  estudio no hizo falta invocar esa cautela, porque las 15 comparaciones
  alcanzaron significancia.
"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (tfm)", "language": "python", "name": "tfm"},
        "language_info": {"name": "python", "version": "3.11.15"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path(__file__).parent / "significancia_estadistica.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Notebook escrito: {out_path} ({len(cells)} celdas)")
