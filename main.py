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

# ------------ Ustawienia -------------
VIDEO_PATH = "/Users/bartlomiejostasz/lot/n/git/1.mov"
ARUCO_DICT = "DICT_6X6_1000"   # ręcznie lub AUTO w przyszłości
MARKER_ID  = 2                 # None = pierwszy wykryty
OUT_VIDEO  = "out_annotated.mp4"
# -------------------------------------

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
                 marker_id: int | None,
                 out_path: str) -> None:
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
        self.marker_id = marker_id

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

        self.records: List[Tuple[float,float,float]] = []
        self.first_center: Tuple[float,float] | None = None

    # ---------------------------------
    def run(self) -> None:
        frame_idx = 0
        start = time.time()
        while True:
            ok, frame = self.cap.read()
            if not ok: break
            t = frame_idx / self.fps

            corners, ids, _ = self.detector.detectMarkers(frame)
            if ids is not None and len(ids):
                ids = ids.flatten()
                idx = 0
                if self.marker_id is not None:
                    matches = np.where(ids == self.marker_id)[0]
                    if len(matches): idx = int(matches[0])
                    else: ids = None  # nie ten ID

            # ---- ArUco widoczny ----
            if ids is not None and len(ids):
                self.lost_counter = 0
                c = corners[idx][0]
                cx, cy = c.mean(axis=0)

                # -- mniejszy bbox (½ szer. & wys.), zachowujemy środek --
                pad = 15
                x_min, y_min = np.min(c, axis=0)
                x_max, y_max = np.max(c, axis=0)
                full_w = max(2, x_max - x_min)
                full_h = max(2, y_max - y_min)
                half_w = max(2, int(full_w * 0.5))
                half_h = max(2, int(full_h * 0.5))
                x_start = int(cx - half_w/2)
                y_start = int(cy - half_h/2)
                x_start = max(0, min(self.w - half_w, x_start))
                y_start = max(0, min(self.h - half_h, y_start))
                bbox = (x_start, y_start, half_w, half_h)
                self.last_bbox = bbox

                if self.track_mode or not self.csrt_ready:
                    try: self.tracker = cv2.TrackerCSRT_create()
                    except AttributeError: self.tracker = cv2.legacy.TrackerCSRT_create()
                    self.tracker.init(frame, bbox)
                    self.csrt_ready = True
                self.track_mode = False

                if self.first_center is None: self.first_center = (cx, cy)
                cv2.polylines(frame, [c.astype(int)], True, (0,255,0), 2)
                cv2.circle(frame, (int(cx),int(cy)), 4, (0,0,255), -1)
                self.records.append((t, cx, cy))
            else:
                # ---- fallback CSRT ----
                self.lost_counter += 1
                if self.csrt_ready and self.lost_counter >= self.lost_thresh:
                    self.track_mode = True
                if self.track_mode and self.csrt_ready:
                    ok_t, nbbox = self.tracker.update(frame)
                    if ok_t:
                        x,y,w,h = [int(v) for v in nbbox]
                        cx, cy = x + w/2, y + h/2
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                        cv2.circle(frame,(int(cx),int(cy)),4,(255,0,0),-1)
                        self.records.append((t, cx, cy))
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
        df = pd.DataFrame(self.records, columns=["t","cx","cy"])
        # Δy: dodatnie w górę, ujemne w dół
        df["dy_px"] = self.first_center[1] - df["cy"]
        df.to_csv("positions_px.csv", index=False)

        plt.figure(figsize=(8,4))
        plt.plot(df["t"], df["dy_px"])
        plt.axhline(0, color="gray", lw=0.8)
        plt.xlabel("Czas [s]"); plt.ylabel("Δy [px]")
        plt.title("Wychylenie pionowe markera")
        plt.grid(); plt.tight_layout()
        plt.savefig("deflection_px.png", dpi=150)
        plt.show()
        print("✔ zapisano positions_px.csv i deflection_px.png")

# ------------- uruchomienie ----------
if __name__ == "__main__":
    ArucoTracker(
        video_path=VIDEO_PATH,
        dict_name=ARUCO_DICT,
        marker_id=MARKER_ID,
        out_path=OUT_VIDEO
    ).run()