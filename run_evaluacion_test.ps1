# Evaluación sobre el held-out de test (comprobación de generalización).
# A diferencia de run_evaluacion_final.ps1, aquí el FPS no es el objetivo
# (la cifra oficial es la de la sesión de consolidación sobre val), así que
# no hace falta máquina en reposo ni tirada sin interrupciones.
# Cada línea usa --split test: escribe eval_results_test.json junto al
# checkpoint sin tocar el eval_results.json de val.
#
# Uso: con el entorno `tfm` activo, desde cualquier directorio:
#     conda activate tfm
#     .\run_evaluacion_test.ps1

Set-Location $PSScriptRoot

# --- 1. SSD300 ---
python evaluate.py --model ssd --config configs/ssd.yaml --weights runs/ssd/ssd_final_s0/best.pt --split test
python evaluate.py --model ssd --config configs/ssd.yaml --weights runs/ssd/ssd_final_s1/best.pt --split test
python evaluate.py --model ssd --config configs/ssd.yaml --weights runs/ssd/ssd_final_s2/best.pt --split test

# --- 2. YOLO11m variante A (augmentación completa) ---
python evaluate.py --model yolo11 --config configs/yolo11m.yaml --weights runs/yolo11/yolo_a_final_s0/weights/best.pt --split test
python evaluate.py --model yolo11 --config configs/yolo11m.yaml --weights runs/yolo11/yolo_a_final_s1/weights/best.pt --split test
python evaluate.py --model yolo11 --config configs/yolo11m.yaml --weights runs/yolo11/yolo_a_final_s2/weights/best.pt --split test

# --- 3. YOLO11m variante B (control de la ablación) ---
python evaluate.py --model yolo11 --config configs/yolo11m_noaug.yaml --weights runs/yolo11/yolo_b_final_s0/weights/best.pt --split test
python evaluate.py --model yolo11 --config configs/yolo11m_noaug.yaml --weights runs/yolo11/yolo_b_final_s1/weights/best.pt --split test
python evaluate.py --model yolo11 --config configs/yolo11m_noaug.yaml --weights runs/yolo11/yolo_b_final_s2/weights/best.pt --split test

# --- 4. CNN baseline ---
python evaluate.py --model cnn_baseline --config configs/cnn_baseline.yaml --weights runs/cnn_baseline/cnnb_final_s0/best.pt --split test
python evaluate.py --model cnn_baseline --config configs/cnn_baseline.yaml --weights runs/cnn_baseline/cnnb_final_s1/best.pt --split test
python evaluate.py --model cnn_baseline --config configs/cnn_baseline.yaml --weights runs/cnn_baseline/cnnb_final_s2/best.pt --split test

# --- 5. Faster R-CNN ---
python evaluate.py --model faster_rcnn --config configs/faster_rcnn.yaml --weights runs/faster_rcnn/frcnn_final_s0/best.pt --split test
python evaluate.py --model faster_rcnn --config configs/faster_rcnn.yaml --weights runs/faster_rcnn/frcnn_final_s1/best.pt --split test
python evaluate.py --model faster_rcnn --config configs/faster_rcnn.yaml --weights runs/faster_rcnn/frcnn_final_s2/best.pt --split test

# --- 6. DETR ---
python evaluate.py --model detr --config configs/detr.yaml --weights runs/detr/detr_final_s0/best.pt --split test
python evaluate.py --model detr --config configs/detr.yaml --weights runs/detr/detr_final_s1/best.pt --split test
python evaluate.py --model detr --config configs/detr.yaml --weights runs/detr/detr_final_s2/best.pt --split test

# --- Agregación sobre test ---
python consolidar.py --split test --json resumen_test.json
