# Sesión de consolidación de FPS -- ejecutar con la máquina en reposo
# (sin navegador, sin VS Code, nada más usando la GPU).
# Los 18 runs seguidos, de una sola tirada, para que el FPS sea comparable.
#
# Uso: con el entorno `tfm` activo, desde cualquier directorio:
#     conda activate tfm
#     .\run_evaluacion_final.ps1

Set-Location $PSScriptRoot

# --- 1. SSD300 ---
python evaluate.py --model ssd --config configs/ssd.yaml --weights runs/ssd/ssd_final_s0/best.pt
python evaluate.py --model ssd --config configs/ssd.yaml --weights runs/ssd/ssd_final_s1/best.pt
python evaluate.py --model ssd --config configs/ssd.yaml --weights runs/ssd/ssd_final_s2/best.pt

# --- 2. YOLO11m variante A (augmentación completa) ---
python evaluate.py --model yolo11 --config configs/yolo11m.yaml --weights runs/yolo11/yolo_a_final_s0/weights/best.pt
python evaluate.py --model yolo11 --config configs/yolo11m.yaml --weights runs/yolo11/yolo_a_final_s1/weights/best.pt
python evaluate.py --model yolo11 --config configs/yolo11m.yaml --weights runs/yolo11/yolo_a_final_s2/weights/best.pt

# --- 3. YOLO11m variante B (control de la ablación) ---
python evaluate.py --model yolo11 --config configs/yolo11m_noaug.yaml --weights runs/yolo11/yolo_b_final_s0/weights/best.pt
python evaluate.py --model yolo11 --config configs/yolo11m_noaug.yaml --weights runs/yolo11/yolo_b_final_s1/weights/best.pt
python evaluate.py --model yolo11 --config configs/yolo11m_noaug.yaml --weights runs/yolo11/yolo_b_final_s2/weights/best.pt

# --- 4. CNN baseline ---
python evaluate.py --model cnn_baseline --config configs/cnn_baseline.yaml --weights runs/cnn_baseline/cnnb_final_s0/best.pt
python evaluate.py --model cnn_baseline --config configs/cnn_baseline.yaml --weights runs/cnn_baseline/cnnb_final_s1/best.pt
python evaluate.py --model cnn_baseline --config configs/cnn_baseline.yaml --weights runs/cnn_baseline/cnnb_final_s2/best.pt

# --- 5. Faster R-CNN ---
python evaluate.py --model faster_rcnn --config configs/faster_rcnn.yaml --weights runs/faster_rcnn/frcnn_final_s0/best.pt
python evaluate.py --model faster_rcnn --config configs/faster_rcnn.yaml --weights runs/faster_rcnn/frcnn_final_s1/best.pt
python evaluate.py --model faster_rcnn --config configs/faster_rcnn.yaml --weights runs/faster_rcnn/frcnn_final_s2/best.pt

# --- 6. DETR ---
python evaluate.py --model detr --config configs/detr.yaml --weights runs/detr/detr_final_s0/best.pt
python evaluate.py --model detr --config configs/detr.yaml --weights runs/detr/detr_final_s1/best.pt
python evaluate.py --model detr --config configs/detr.yaml --weights runs/detr/detr_final_s2/best.pt

# --- Agregación: media +/- std sobre las 3 semillas, filas LaTeX y comprobaciones ---
python consolidar.py --json resumen.json
