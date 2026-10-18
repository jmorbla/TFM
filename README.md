# TFM

Reconocimiento de Objetos en Entornos Urbanos en Tiempo Real mediante Aprendizaje Profundo

Comparativa de cinco modelos de detección sobre BDD100K (10 clases): YOLO11m
(con y sin la augmentación de Ultralytics), Faster R-CNN, SSD300, DETR y una
CNN propia tipo RetinaNet (MobileNetV3-Small + FPN P2-P5). Cada configuración se
entrena con las semillas 0, 1 y 2: 18 entrenamientos en total.

## Protocolo de evaluación (idéntico en todos los modelos)

- 8 000 imágenes de train, 2 000 de validación y 2 000 de test, muestreadas con la semilla 42.
- `conf = 0.001` y `max_det = 100`; métricas en píxeles nativos (1280×720).
- Precision, Recall y F1 en el punto de máximo F1 (criterio de Ultralytics).
- FPS = forward + postproceso, con batch 1, sincronización CUDA y 10 imágenes de calentamiento.
- Parada temprana con `patience = 15`; `best.pt` se elige por mAP@50-95 de validación.

## Estructura

| Ruta | Contenido |
|---|---|
| `configs/` | Un YAML por modelo y la configuración del dataset (`bdd100k.yaml`) |
| `data/` | Preparación del dataset, `Dataset` de PyTorch y transforms |
| `models/` | Definición de cada modelo (y entrenamiento de YOLO11) |
| `train.py` | Entrenamiento de Faster R-CNN, SSD300, DETR y CNN propia |
| `evaluate.py` | Evaluación común a los cinco modelos; escribe `eval_results.json` junto al checkpoint |
| `consolidar.py` | Media ± desviación sobre las 3 semillas y filas LaTeX de las tablas |
| `stats_dataset.py` | Instancias y tamaño de objeto por clase |
| `run_evaluacion_final.ps1`, `run_evaluacion_test.ps1` | Evaluación de los 18 runs sobre val y sobre test |
| `graficas_resultados/`, `graficas_entrenamiento/`, `graficas_cualitativas/`, `curvas_pr_iou/`, `significancia_estadistica/` | Scripts `generar_*.py` que crean los notebooks de figuras |
| `experiments/curva_aprendizaje/` | Curva de aprendizaje (8k / 16k / 32k imágenes) |
| `experiments/cnn_baseline_anclas_coste/` | Cobertura de anchors y coste por componente de la CNN propia |
| `resumen.json`, `resumen_test.json` | Agregado de los 18 runs sobre val y sobre test |

## Entorno

```
conda create -n tfm python=3.11
conda activate tfm
pip install -r requirements.txt
```

Los entrenamientos se hicieron en una RTX 5070 Laptop de 8 GB.

## Reproducción

Todos los comandos se ejecutan desde la raíz del repositorio.

**1. Datos.** Descargar BDD100K (imágenes 100k y etiquetas de detección) en
`bdd100k_images_100k/100k/{train,val,test}` y `bdd100k_labels/100k/{train,val,test}`, y convertir:

```
python data/prepare_bdd100k.py
python data/build_yolo_dataset.py
```

**2. Entrenamiento** (una vez por semilla; el nombre del run debe terminar en `_s0`, `_s1` o `_s2`):

```
python train.py --model ssd          --config configs/ssd.yaml          --run_name ssd_final_s0  --seed 0
python train.py --model faster_rcnn  --config configs/faster_rcnn.yaml  --run_name frcnn_final_s0 --seed 0
python train.py --model detr         --config configs/detr.yaml         --run_name detr_final_s0  --seed 0
python train.py --model cnn_baseline --config configs/cnn_baseline.yaml --run_name cnnb_final_s0  --seed 0
python models/yolo11/train.py --config configs/yolo11m.yaml       --run_name yolo_a_final_s0 --seed 0
python models/yolo11/train.py --config configs/yolo11m_noaug.yaml --run_name yolo_b_final_s0 --seed 0
```

**3. Evaluación.** `run_evaluacion_final.ps1` evalúa los 18 runs seguidos sobre val
(el FPS solo es comparable si se mide así, con la máquina en reposo) y
`run_evaluacion_test.ps1` los evalúa sobre test. Para un solo run:

```
python evaluate.py --model ssd --config configs/ssd.yaml --weights runs/ssd/ssd_final_s0/best.pt [--split test]
```

**4. Tablas y figuras.**

```
python consolidar.py --json resumen.json
python consolidar.py --split test --json resumen_test.json
python graficas_resultados/generar_graficas.py
python -m jupyter nbconvert --to notebook --execute --inplace graficas_resultados/graficas_resultados.ipynb
```

Los demás directorios de figuras funcionan igual (`generar_*.py` + `nbconvert`).

## Qué hay en `runs/` y qué no

Solo se versionan los resultados de evaluación de los 18 runs finales
(`eval_results.json` en val y `eval_results_test.json` en test, junto a cada checkpoint).
Los checkpoints (`*.pt`), los logs de entrenamiento y los runs preliminares no están en el repositorio,
ni tampoco los datos de BDD100K (ver `.gitignore`).

Con esos JSON se pueden reproducir sin reentrenar: `consolidar.py` (tablas y `resumen*.json`),
`significancia_estadistica/` y `curvas_pr_iou/`. No se pueden reproducir `graficas_entrenamiento/`
(necesita los logs de cada run) ni `graficas_cualitativas/` (necesita los checkpoints).

Los JSON de YOLO11m conservan en el campo `weights` la ruta con la que se evaluaron
(`runs/detect/runs/yolo11/...`); `consolidar.py` los localiza por el nombre del run, no por ese campo.
