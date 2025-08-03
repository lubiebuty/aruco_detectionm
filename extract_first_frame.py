#!/usr/bin/env python3
"""
Zrzut pierwszej klatki z podglądem markera ArUco.

• Otwiera VIDEO_PATH
• Czyta pierwszą udaną klatkę
• Wykrywa znacznik(e) ArUco (DICT_6X6_1000)
• Rysuje zieloną obwiednię oraz środek (czerwona kropka)
• Zapisuje wynik do pliku PNG i wyświetla w oknie
"""

from __future__ import annotations
import cv2
import sys
from pathlib import Path
from ultralytics import YOLO

# ----------------- Ustawienia -----------------
VIDEO_PATH = "/Users/bartlomiejostasz/lot/n/3klatki/IMG_4699.MOV"
DICT_NAME  = "DICT_6X6_1000"
OUT_IMAGE  = "first_frame_aruco.png"
PIX_COLOR  = (0, 255, 0)     # zielony obrys
CENTER_COL = (0, 0, 255)     # czerwony środek
YOLO_MODEL_PATH = "runs/detect/train7/weights/best.pt"   # ścieżka do wytrenowanych wag YOLOv8
YOLO_CONF       = 0.05      # obniż próg, żeby zobaczyć wszystkie trzy klasy
YOLO_IMG_SIZE   = 640       # wymuś rozdzielczość wejściową dla YOLO
YOLO_COL        = (255, 255, 0)   # żółty bbox YOLO
# ----------------------------------------------

# słowniki OpenCV
ARUCO_DICTS = {
    name: getattr(cv2.aruco, name)
    for name in dir(cv2.aruco) if name.startswith("DICT_")
}

if DICT_NAME not in ARUCO_DICTS:
    sys.exit(f"❌ Nieznany słownik {DICT_NAME}")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    sys.exit("❌ Nie można otworzyć pliku wideo")

ok, frame = cap.read()   # pierwsza udana klatka
cap.release()

if not ok:
    sys.exit("❌ Nie udało się odczytać pierwszej klatki")

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[DICT_NAME])
detector = cv2.aruco.ArucoDetector(aruco_dict)

corners, ids, _ = detector.detectMarkers(frame)

if ids is None or len(ids) == 0:
    print("⚠️  Brak markerów ArUco w pierwszej klatce")
else:
    for pts in corners:
        pts = pts[0].astype(int)
        cv2.polylines(frame, [pts], True, PIX_COLOR, 2)
        center = pts.mean(axis=0).astype(int)
        cv2.circle(frame, tuple(center), 4, CENTER_COL, -1)

# ------------ YOLOv8 wykrywanie tych samych markerów --------------
try:
    model = YOLO(YOLO_MODEL_PATH)
    yolo_res = model.predict(
        source=frame,
        conf=YOLO_CONF,
        imgsz=YOLO_IMG_SIZE,
        device="mps" if cv2.cuda.getCudaEnabledDeviceCount() == 0 else "0",
        verbose=False
    )
    dets = yolo_res[0]
    class_names = {0: "Left", 1: "Center", 2: "Right"}

    for box, cls in zip(dets.boxes, dets.boxes.cls):
        cls_id = int(cls)
        if cls_id not in (0, 1, 2):
            continue            # ignoruj inne klasy
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), YOLO_COL, 2)
        label = f"Y{cls_id}:{class_names[cls_id]}"
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, YOLO_COL, 2)
        # środek bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
    found_cls = {int(c) for c in dets.boxes.cls}
    print("YOLO classes detected:", found_cls)
except Exception as e:
    print("⚠️  YOLO detection skipped:", e)

# zapis i podgląd
cv2.imwrite(OUT_IMAGE, frame)
print(f"✔ Zapisano {OUT_IMAGE}")

cv2.imshow("Pierwsza klatka z ArUco", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()