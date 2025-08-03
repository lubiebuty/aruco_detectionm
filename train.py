
from ultralytics import YOLO

model = YOLO("yolov8n.pt")           # dowolny rozmiar: n, s, m, l, x
model.train(
    data="/Users/bartlomiejostasz/PYCH/aruco_detection/marker_dataset/marker.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device="mps",                    # Apple Silicon GPU; na PC użyj "0" (CUDA)
    workers=4,                       # szybkie wczytywanie na macOS
)