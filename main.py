#!/usr/bin/env python3
"""
Śledzenie markera ArUco + fallback CSRT
– bbox o połowę mniejszy,
– Δy dodatnie = ruch w górę, ujemne = w dół.
"""
from __future__ import annotations
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys, argparse, math, time
from typing import List, Tuple, Dict
from scipy.signal import savgol_filter

from ultralytics import YOLO

# ------------ Ustawienia -------------
VIDEO_PATH = "/Users/bartlomiejostasz/lot/n/3klatki/IMG_4699.MOV"
# plik kalibracyjny kamery (macierz K i dystorsja)
CALIB_PATH = '/Users/bartlomiejostasz/PYCH/LOT/1:5.npz'
ARUCO_DICT = "DICT_6X6_1000"   # ręcznie lub AUTO w przyszłości
MARKER_ID  = 2                 # None = pierwszy wykryty
# tylko ten marker ArUco będzie akceptowany (DICT_6X6_1000, ID 2)
TARGET_MARKER_ID = 2
OUT_VIDEO  = "out_annotated.mp4"
REFRESH_INTERVAL = 10
HOLD_FRAMES = 20        # ile klatek „trzymamy” ostatni punkt gdy tracker zgubi marker
PIXEL_MARGIN = 10          # stała: powiększamy bbox o 10 px z każdej strony

# ----------- Fusion validation constants ----------
EPSILON_PIX = 5         # tolerancja rogów – luźniej o 2 px
BETA_RATIO  = 0.40      # środek ArUco może odjechać do 40 % bbox

ENLARGE_FAC = 1.5      # factor to enlarge bbox when re‑initialising

# ----------- Bbox size limit -----------
MAX_BBOX_FRAC = 0.20   # bbox area may occupy max 20 % całej klatki

COLOR_ARUCO_BOX = (255, 0, 0)   # blue box around exact ArUco marker
COLOR_ARUCO_CENTER = (0, 0, 255)  # red dot at ArUco center
# -------------------------------------

# ścieżka do wytrenowanego modelu YOLOv8
YOLO_MODEL_PATH = "runs/detect/train7/weights/best.pt"   # ścieżka do wytrenowanego modelu YOLOv8
# -------------------------------------

# rzeczywiste parametry pomiarowe — iPhone 15 Pro
MARKER_SIZE_CM = 3.5
MARKER_DISTANCE_CM = 300.0
CAMERA_HFOV_DEG = 73.7     # pole widzenia poziome w stopniach
SENSOR_WIDTH_PX = 4032     # szerokość zdjęcia w px (pełne 48 MP)

# słowniki OpenCV
ARUCO_DICTS: Dict[str, int] = {
    name: getattr(cv2.aruco, name)
    for name in dir(cv2.aruco) if name.startswith("DICT_")
}

# ---------- Klasa główna -------------
class ArucoTracker:
    def __init__(self,
                 video_path: str | int,
                 dict_name: str,
                 out_path: str,
                 camera_matrix=None,
                 dist_coeffs=None) -> None:
        if dict_name not in ARUCO_DICTS:
            raise ValueError("Nieznany słownik ArUco")
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            sys.exit("❌ Nie można otworzyć wideo")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.w  = int(self.cap.get(3))
        self.h  = int(self.cap.get(4))

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict)

        # YOLOv8 – wykrywanie trzech markerów (marker_left, marker_center, marker_right)
        self.yolo = YOLO(YOLO_MODEL_PATH)
        # mapowanie indeksu klasy => marker_id (rid)
        self.class2rid = {0: 2, 1: 3, 2: 4}   # YOLO klasa 0→ID2, 1→ID3, 2→ID4

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(out_path, fourcc, self.fps, (self.w, self.h))

        try:
            self.tracker = cv2.TrackerCSRT_create()
        except AttributeError:
            self.tracker = cv2.legacy.TrackerCSRT_create()

        self.track_mode = False        # False = ArUco, True = CSRT
        self.last_bbox = None
        self.csrt_ready = False
        self.lost_counter = 0
        self.lost_thresh  = 1

        self.records: List[Tuple[float,float,float,float | None, float | None]] = []
        self.first_center: Tuple[float,float] | None = None

        self.K = camera_matrix
        self.D = dist_coeffs

        self.initialized = False
        self.cm_per_px_fixed = None
        self.centers: Dict[int, np.ndarray] = {}

        self.bboxes: Dict[int, Tuple[float,float,float,float]] = {}  # rid → (x,y,w,h)
        self.aruco_corners: Dict[int, np.ndarray] = {}              # rid → 4×2 corners

        self.trackers: Dict[int, cv2.Tracker] = {}
        self.last_seen = {}

    # ---------------------------------
    def _bbox_from_corners(self, corners: np.ndarray) -> Tuple[float,float,float,float]:
        """Tight bbox (+PIXEL_MARGIN) around 4×2 corners."""
        xs, ys = corners[:,0], corners[:,1]
        x_min = max(xs.min() - PIXEL_MARGIN, 0)
        y_min = max(ys.min() - PIXEL_MARGIN, 0)
        x_max = min(xs.max() + PIXEL_MARGIN, self.w)
        y_max = min(ys.max() + PIXEL_MARGIN, self.h)
        return (x_min, y_min, x_max - x_min, y_max - y_min)

    def run(self) -> None:
        frame_idx = 0
        start = time.time()
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            # Undistort frame if calibration is available
            if self.K is not None and self.D is not None:
                frame = cv2.undistort(frame, self.K, self.D)
            t = frame_idx / self.fps

            if not self.initialized:
                # --- YOLO wykrywa trzy markery ---
                yolo_res = self.yolo.predict(source=frame, conf=0.25, iou=0.5, device="mps", verbose=False)
                dets = yolo_res[0]
                # filtrowanie tylko klas 0..2
                found = {int(cls): box.xyxy[0].cpu().numpy() for cls, box in zip(dets.boxes.cls, dets.boxes)}
                if all(k in found for k in (0,1,2)):
                    required_ids = [2,3,4]   # trzy markery (L,C,R); odpowiadają klasom YOLO 0,1,2
                    self.trackers = {}
                    for cls_idx, rid in self.class2rid.items():
                        x1,y1,x2,y2 = found[cls_idx]
                        # powiększ bbox o stały margines 10 px na każdą stronę
                        x_min = max(x1 - PIXEL_MARGIN, 0)
                        y_min = max(y1 - PIXEL_MARGIN, 0)
                        x_max = min(x2 + PIXEL_MARGIN, self.w)
                        y_max = min(y2 + PIXEL_MARGIN, self.h)
                        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                        try:
                            tracker = cv2.legacy.TrackerCSRT_create()
                        except AttributeError:
                            tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame, bbox)
                        self.trackers[rid] = tracker
                        self.bboxes[rid] = bbox
                    self.initialized = True
                    self.track_mode = True
                    # start reference = środek markera 3 (YOLO klasa 1)
                    x1,y1,x2,y2 = found[1]
                    c3x = (x1 + x2)/2; c3y = (y1 + y2)/2
                    self.first_center = (c3x, c3y)
                    self.centers = {}
                else:
                    # Rysuj wykrycia YOLO (tylko podgląd)
                    for box, cls in zip(dets.boxes, dets.boxes.cls):
                        x1,y1,x2,y2 = box.xyxy[0]
                        cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,255), 2)
                        cv2.putText(frame, f"YOLO {int(cls)}", (int(x1), int(y1)-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            else:
                # Update trackers for each marker
                updated_centers = {}

                # ------ ArUco detection for precise center & overlay ------
                corners, ids, _ = self.detector.detectMarkers(frame)
                if ids is not None and len(ids):
                    ids_arr = ids.flatten().astype(int)
                    for marker_corners, marker_id in zip(corners, ids_arr):
                        if marker_id in (2, 3, 4):   # tylko nasze markery
                            pts = marker_corners[0].astype(int)
                            cv2.polylines(frame, [pts], True, COLOR_ARUCO_BOX, 2)
                            ar_center = pts.mean(axis=0)
                            cv2.circle(frame, tuple(ar_center.astype(int)), 4, COLOR_ARUCO_CENTER, -1)
                            # nadpisz center jeśli to dokładniejszy pomiar
                            updated_centers[marker_id] = tuple(ar_center)
                            self.aruco_corners[marker_id] = marker_corners[0]
                            # --- re‑init tracker from exact ArUco if missing ---
                            tight_bbox = self._bbox_from_corners(marker_corners[0])
                            if marker_id not in self.trackers:
                                try:
                                    tracker = cv2.legacy.TrackerCSRT_create()
                                except AttributeError:
                                    tracker = cv2.TrackerCSRT_create()
                                tracker.init(frame, tight_bbox)
                                self.trackers[marker_id] = tracker
                                self.bboxes[marker_id] = tight_bbox

                lost_trackers = []
                for rid, tracker in self.trackers.items():
                    ok, bbox = tracker.update(frame)
                    if ok:
                        x,y,w,h = bbox
                        p1 = (int(x), int(y))
                        p2 = (int(x + w), int(y + h))
                        cv2.rectangle(frame, p1, p2, (255,0,0), 2)
                        center = (x + w/2, y + h/2)
                        self.last_seen[rid] = {"frame": self.frame_idx, "center": center}
                        updated_centers[rid] = center
                        self.bboxes[rid] = bbox
                        cv2.putText(frame, f"ID: {rid}", (int(center[0]+5), int(center[1]-5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    else:
                        if rid in self.last_seen and self.frame_idx - self.last_seen[rid]["frame"] <= HOLD_FRAMES:
                            center = self.last_seen[rid]["center"]
                            updated_centers[rid] = center
                            x, y = center
                            last_bbox = self.bboxes.get(rid, (0,0,60,60))
                            w = last_bbox[2] * 0.5
                            h = last_bbox[3] * 0.5
                            p1 = (int(x - w/2), int(y - h/2))
                            p2 = (int(x + w/2), int(y + h/2))
                            cv2.rectangle(frame, p1, p2, (0,0,255), 2)
                            cv2.circle(frame, (int(x), int(y)), 4, (0,0,255), -1)
                            cv2.putText(frame, f"ID: {rid} (hold)", (int(x+5), int(y-5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                        else:
                            lost_trackers.append(rid)
                # For lost trackers or refresh interval, try YOLO fallback detection
                if lost_trackers or (self.frame_idx % REFRESH_INTERVAL == 0):
                    yolo_res = self.yolo.predict(source=frame, conf=0.25, iou=0.5, device="mps", verbose=False)
                    dets = yolo_res[0]
                    for cls_idx, rid in self.class2rid.items():
                        if rid in lost_trackers or self.frame_idx % REFRESH_INTERVAL == 0:
                            mask = (dets.boxes.cls.int() == cls_idx)
                            if mask.any():
                                box = dets.boxes[mask][0]
                                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                                x_min = max(x1 - PIXEL_MARGIN, 0)
                                y_min = max(y1 - PIXEL_MARGIN, 0)
                                x_max = min(x2 + PIXEL_MARGIN, self.w)
                                y_max = min(y2 + PIXEL_MARGIN, self.h)
                                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                                try:
                                    tracker = cv2.legacy.TrackerCSRT_create()
                                except AttributeError:
                                    tracker = cv2.TrackerCSRT_create()
                                tracker.init(frame, bbox)
                                self.trackers[rid] = tracker
                                self.bboxes[rid] = bbox
                                updated_centers[rid] = ((x1 + x2)/2, (y1 + y2)/2)

                # -------- fusion validation: ensure ArUco lies inside bbox --------
                for rid in (2,3,4):
                    if rid in self.bboxes and rid in self.aruco_corners:
                        bbox   = self.bboxes[rid]
                        x,y,w,h = bbox
                        x2,y2 = x+w, y+h
                        ar_corners = self.aruco_corners[rid]
                        # check all corners
                        corners_ok = all(
                            (x - EPSILON_PIX) <= cx <= (x2 + EPSILON_PIX) and
                            (y - EPSILON_PIX) <= cy <= (y2 + EPSILON_PIX)
                            for (cx,cy) in ar_corners
                        )
                        # check center deviation
                        if rid in updated_centers:
                            cx, cy = updated_centers[rid]
                            center_ok = (
                                abs((x + w/2) - cx) <= w * BETA_RATIO and
                                abs((y + h/2) - cy) <= h * BETA_RATIO
                            )
                        else:
                            center_ok = True
                        # limit bbox size; if too large, replace by tight bbox around ArUco
                        bbox_area = w * h
                        if bbox_area > MAX_BBOX_FRAC * self.w * self.h:
                            new_bbox = self._bbox_from_corners(ar_corners)
                            try:
                                tracker = cv2.legacy.TrackerCSRT_create()
                            except AttributeError:
                                tracker = cv2.TrackerCSRT_create()
                            tracker.init(frame, new_bbox)
                            self.trackers[rid] = tracker
                            self.bboxes[rid] = new_bbox
                            continue
                        if not (corners_ok and center_ok):
                            # re‑init tracker to tight bbox around ArUco (plus margin)
                            new_bbox = self._bbox_from_corners(ar_corners)
                            try:
                                tracker = cv2.legacy.TrackerCSRT_create()
                            except AttributeError:
                                tracker = cv2.TrackerCSRT_create()
                            tracker.init(frame, new_bbox)
                            self.trackers[rid] = tracker
                            self.bboxes[rid] = new_bbox

                self.centers = updated_centers

                # Record data only for marker 3 if available
                if 3 in self.centers:
                    cx, cy = self.centers[3]
                    if self.first_center is None:
                        self.first_center = (cx, cy)
                    # Calculate dy in pixels (positive up)
                    dy_px = self.first_center[1] - cy
                    # Convert to cm using fixed scale if available
                    z_cm = None
                    dy_cm = None
                    if self.cm_per_px_fixed is not None:
                        dy_cm = dy_px * self.cm_per_px_fixed
                    self.records.append((t, cx, cy, z_cm, dy_cm))

            self.writer.write(frame)
            cv2.imshow("tracker", frame)
            if cv2.waitKey(1)&0xFF==27: break
            frame_idx += 1
            self.frame_idx = frame_idx

        self.cap.release(); self.writer.release(); cv2.destroyAllWindows()
        # self._postprocess()
        print("✔ YOLO + CSRT tracking zakończone")

    # ---------------------------------
    def _postprocess(self)->None:
        if not self.records: return
        df = pd.DataFrame(self.records, columns=["t","cx","cy","z_cm","dy_cm"])
        # Δy: dodatnie w górę, ujemne w dół
        df["dy_px"] = self.first_center[1] - df["cy"]
        # Interpolacja rzeczywistego przemieszczenia pionowego
        df["dy_cm_interp"] = df["dy_cm"].interpolate(method="linear")

        # Eksport pełnych danych do CSV
        df.to_csv("positions_full.csv", index=False)

        plt.figure(figsize=(8,4))
        plt.plot(df["t"], df["dy_px"])
        plt.axhline(0, color="gray", lw=0.8)
        plt.xlabel("Czas [s]"); plt.ylabel("Δy [px]")
        plt.title("Wychylenie pionowe markera")
        plt.grid(); plt.tight_layout()
        plt.savefig("deflection_px.png", dpi=150)
        plt.show()

        if df["z_cm"].notna().any():
            plt.figure(figsize=(8,4))
            plt.plot(df["t"], df["z_cm"], label="Z [cm]")
            plt.xlabel("Czas [s]"); plt.ylabel("Z [cm]")
            plt.title("Przemieszczenie osi Z")
            plt.grid(); plt.tight_layout()
            plt.savefig("deflection_z_cm.png", dpi=150)
            plt.show()

        if df["dy_cm"].notna().any():
            plt.figure(figsize=(8,4))
            plt.plot(df["t"], df["dy_cm"], label="ΔY [cm]")
            plt.xlabel("Czas [s]"); plt.ylabel("ΔY [cm]")
            plt.title("Wychylenie pionowe markera (przybliżone)")
            plt.grid(); plt.tight_layout()
            plt.savefig("deflection_dy_cm.png", dpi=150)
            plt.show()

        # Nowy wykres dla interpolowanego przemieszczenia pionowego (dy_cm_interp)
        if df["dy_cm_interp"].notna().any():
            plt.figure(figsize=(8,4))
            plt.plot(df["t"], df["dy_cm_interp"], label="ΔY [cm] (interpolowane)", color="green")
            plt.axhline(0, color="gray", lw=0.8)
            plt.xlabel("Czas [s]"); plt.ylabel("ΔY [cm]")
            plt.title("Interpolowane wychylenie pionowe markera")
            plt.grid(); plt.tight_layout()
            plt.savefig("deflection_dy_cm_interp.png", dpi=150)
            plt.show()

        # Savitzky-Golay smoothing for dy_cm_interp
        if df["dy_cm_interp"].notna().sum() > 10:
            df["dy_cm_smooth"] = savgol_filter(df["dy_cm_interp"], window_length=21, polyorder=3)
        else:
            df["dy_cm_smooth"] = df["dy_cm_interp"]

        # Final plot for smoothed dy_cm
        if df["dy_cm_smooth"].notna().any():
            plt.figure(figsize=(8,4))
            plt.plot(df["t"], df["dy_cm_smooth"], label="ΔY [cm] (wygładzone)", color="darkgreen")
            plt.axhline(0, color="gray", lw=0.8)
            plt.xlabel("Czas [s]"); plt.ylabel("ΔY [cm]")
            plt.title("Wychylenie pionowe markera (cm, wygładzone)")
            plt.grid(); plt.tight_layout()
            plt.savefig("deflection_dy_cm_smooth.png", dpi=150)
            plt.show()

        print("✔ zapisano positions_full.csv i deflection_px.png")

# ------------- uruchomienie ----------
if __name__ == "__main__":
    # ---------- kalibracja ----------
    try:
        with np.load(CALIB_PATH) as calib:
            K = calib["camera_matrix"]
            D = calib["dist_coeffs"]
            print(f"✅ Załadowano kalibrację z {CALIB_PATH}")
    except Exception as e:
        print(f"❌ Brak / błąd kalibracji ({e}); kontynuuję bez niej")
        K, D = None, None

    ArucoTracker(
        video_path=VIDEO_PATH,
        dict_name=ARUCO_DICT,
        out_path=OUT_VIDEO,
        camera_matrix=K,
        dist_coeffs=D
    ).run()