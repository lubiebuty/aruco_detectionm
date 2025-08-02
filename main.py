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

# ------------ Ustawienia -------------
VIDEO_PATH = "/Users/bartlomiejostasz/lot/n/git/1.mov"
# plik kalibracyjny kamery (macierz K i dystorsja)
CALIB_PATH = '/Users/bartlomiejostasz/PYCH/LOT/1:5.npz'
ARUCO_DICT = "DICT_6X6_1000"   # ręcznie lub AUTO w przyszłości
MARKER_ID  = 2                 # None = pierwszy wykryty
# tylko ten marker ArUco będzie akceptowany (DICT_6X6_1000, ID 2)
TARGET_MARKER_ID = 2
OUT_VIDEO  = "out_annotated.mp4"
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

    # ---------------------------------
    def run(self) -> None:
        frame_idx = 0
        start = time.time()
        while True:
            ok, frame = self.cap.read()
            if not ok: break
            # Undistort frame if calibration is available
            if self.K is not None and self.D is not None:
                frame = cv2.undistort(frame, self.K, self.D)
            t = frame_idx / self.fps

            corners, ids, _ = self.detector.detectMarkers(frame)
            if ids is not None and len(ids):
                ids = ids.flatten()
                idx = 0
                # sprawdź tylko czy wśród wykrytych jest dokładnie marker o ID = 2
                matches = np.where(ids == TARGET_MARKER_ID)[0]
                if len(matches):
                    idx = int(matches[0])
                else:
                    ids = None  # nie znaleziono właściwego markera

            # ---- ArUco widoczny ----
            if ids is not None and len(ids):
                self.lost_counter = 0
                c = corners[idx][0]
                cx, cy = c.mean(axis=0)

                if self.first_center is None:
                    self.first_center = (cx, cy)

                dx_px = cx - self.first_center[0]
                dy_px = self.first_center[1] - cy
                # przeliczenie ruchu markera w px → cm na podstawie FOV i szerokości klatki
                scene_width_cm = 2 * MARKER_DISTANCE_CM * math.tan(math.radians(CAMERA_HFOV_DEG / 2))
                cm_per_px = scene_width_cm / SENSOR_WIDTH_PX
                dy_real_cm = dy_px * cm_per_px  # teraz: ujemne = w dół

                # --- pozycja 3‑D dzięki kalibracji ---
                if self.K is not None and self.D is not None:
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [c], 0.04, self.K, self.D)  # marker 4 cm
                    z_cm = tvec[0][0][2] * 100      # metr → cm
                else:
                    z_cm = None

                # -- pełny bbox z paddingiem --
                pad = 15
                x_min, y_min = np.min(c, axis=0)
                x_max, y_max = np.max(c, axis=0)
                x_min = max(0, int(x_min) - pad)
                y_min = max(0, int(y_min) - pad)
                x_max = min(self.w - 1, int(x_max) + pad)
                y_max = min(self.h - 1, int(y_max) + pad)
                bbox = (
                    x_min,
                    y_min,
                    max(2, x_max - x_min),
                    max(2, y_max - y_min),
                )
                self.last_bbox = bbox

                if self.track_mode or not self.csrt_ready:
                    try: self.tracker = cv2.TrackerCSRT_create()
                    except AttributeError: self.tracker = cv2.legacy.TrackerCSRT_create()
                    self.tracker.init(frame, bbox)
                    self.csrt_ready = True
                self.track_mode = False

                cv2.polylines(frame, [c.astype(int)], True, (0,255,0), 2)
                cv2.circle(frame, (int(cx),int(cy)), 4, (0,0,255), -1)
                self.records.append((t, cx, cy, z_cm, dy_real_cm))
            else:
                # ---- fallback CSRT ----
                self.lost_counter += 1
                if self.csrt_ready and self.lost_counter >= self.lost_thresh:
                    self.track_mode = True
                if self.track_mode and self.csrt_ready:
                    ok_t, nbbox = self.tracker.update(frame)
                    if ok_t:
                        cx, cy, w, h = nbbox[0] + nbbox[2]/2, nbbox[1] + nbbox[3]/2, nbbox[2], nbbox[3]
                        scale = 1.20
                        nw, nh = int(w * scale), int(h * scale)
                        x = int(cx - nw/2)
                        y = int(cy - nh/2)
                        w, h = nw, nh
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                        cv2.circle(frame,(int(cx),int(cy)),4,(255,0,0),-1)
                        # --- compute dy_px and dy_real_cm for fallback CSRT tracking ---
                        if self.first_center is not None:
                            dy_px = self.first_center[1] - cy
                            scene_width_cm = 2 * MARKER_DISTANCE_CM * math.tan(math.radians(CAMERA_HFOV_DEG / 2))
                            cm_per_px = scene_width_cm / SENSOR_WIDTH_PX
                            dy_real_cm = dy_px * cm_per_px
                        else:
                            dy_real_cm = None
                        self.records.append((t, cx, cy, None, dy_real_cm))
                    else:
                        cv2.putText(frame,"LOST",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                else:
                    cv2.putText(frame,"LOST",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

            self.writer.write(frame)
            cv2.imshow("tracker", frame)
            if cv2.waitKey(1)&0xFF==27: break
            frame_idx += 1

        self.cap.release(); self.writer.release(); cv2.destroyAllWindows()
        self._postprocess()

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