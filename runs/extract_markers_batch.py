import cv2
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
import argparse
import os
from cv2 import aruco
# słownik użytych znaczników – zakładam DICT_4X4_50 (zmień jeśli inny)
aruco_dict   = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
aruco_params = aruco.DetectorParameters()

parser = argparse.ArgumentParser(description="Extract markers with YOLO + BoT‑SORT")
parser.add_argument("--weights", default="/Users/bartlomiejostasz/PYCH/aruco_detection/runs/detect/train8/weights/best.pt",
                    help="Ścieżka do pliku .pt z wytrenowanymi wagami YOLO")
args = parser.parse_args()
if not os.path.isfile(args.weights):
    raise FileNotFoundError(f"❌ Nie znaleziono pliku wag YOLO: {args.weights}\n"
                            "   ➜ Podaj poprawną ścieżkę parametrem --weights /ścieżka/best.pt")

yolo_model = YOLO(args.weights)   # ścieżka do wytrenowanego modelu 3‑klasowego

# Kalman dla każdej roli (2=Left, 3=Center, 4=Right)
role_kf = {2: _new_kf(), 3: _new_kf(), 4: _new_kf()}

GATE_PX = 80      # maksymalna zmiana cx, aby zachować ten sam ID

kalmans = {}

def _new_kf():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32)*1
    return kf

def overlay_markers(video_path: str, csv_path: str, output_path: str):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0

    rid2color = {
        2: (0, 0, 255),   # Red
        3: (0, 255, 0),   # Green
        4: (255, 0, 0)    # Blue
    }

    records = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ----- YOLO z BoT‑SORT -----
        results = yolo_model.track(frame,
                                   conf=0.25,
                                   iou=0.4,
                                   persist=True,
                                   tracker="botsort.yaml")[0]

        frame_dets = []
        for box in results.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box.astype(int)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            frame_dets.append({"cx": cx, "cy": cy,
                               "bbox": (x1, y1, x2, y2)})

        # ---------- ArUco detect w TEJ klatce ----------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        ar_centers = {}
        if ids is not None:
            for cid, cset in zip(ids.flatten(), corners):
                # cid == 2/3/4 (Left/Center/Right)
                if cid in (2,3,4):
                    pts = cset[0]
                    ax = pts[:,0].mean()
                    ay = pts[:,1].mean()
                    ar_centers[cid] = (ax, ay)

        # ---------- role assignment ----------
        assigned = {}

        # (a) Hungarian na podstawie predykcji Kalmanów
        if frame_dets:
            pred_cx = np.array([[role_kf[r].predict()[0,0]] for r in (2,3,4)])  # 3×1
            det_cx  = np.array([[d["cx"]] for d in frame_dets]).T               # 1×N
            cost = abs(pred_cx - det_cx)                                        # 3×N
            row, col = linear_sum_assignment(cost)
            for r, c in zip(row, col):
                rid = [2,3,4][r]
                assigned[rid] = frame_dets[c]

        # (b) override, jeśli wzór ArUco leży w bboxie
        for rid_true, (ax, ay) in ar_centers.items():
            for det in frame_dets:
                x1, y1, x2, y2 = det["bbox"]
                if x1 <= ax <= x2 and y1 <= ay <= y2:
                    assigned[rid_true] = det
                    break   # wzór przypisany

        # ----- zapis + Kalman -----
        for rid in (2,3,4):
            det = assigned.get(rid)
            if det is None:
                continue    # brak detekcji w tej klatce; zostaw lukę
            cx, cy = det["cx"], det["cy"]
            x1,y1,x2,y2 = det["bbox"]

            role_kf[rid].correct(np.array([[cx],[cy]], np.float32))

            records.append((frame_idx, rid, cx, cy, x1, y1, x2, y2))

            # rysowanie
            color = {2:(0,0,255), 3:(0,255,0), 4:(255,0,0)}[rid]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (int(cx), int(cy)), 5, color, -1)
            cv2.putText(frame, f"ID {rid}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()