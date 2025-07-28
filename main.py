#!/usr/bin/env python3
"""Aruco-based trampoline deflection tracker.

Usage (terminal):
    python main.py \
        --video "/ścieżka/do/1.mov" \
        --dict  DICT_6X6_1000 \
        --marker-id 2

Domyślnie (bez parametrów) wczyta plik /Users/bartlomiejostasz/lot/n/git/1.mov.
Wymaga: opencv-contrib-python, pandas, matplotlib, numpy.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Stałe i funkcje pomocnicze
# ---------------------------------------------------------------------------

ARUCO_DICTS: Dict[str, int] = {
    name: getattr(cv2.aruco, name)
    for name in dir(cv2.aruco)
    if name.startswith("DICT_")
}


def auto_detect_dict_and_id(video_path: str | int) -> Tuple[str, int]:
    """Przeskanuj pierwszą klatkę i zwróć (dict_name, marker_id)."""
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError("Nie udało się odczytać pierwszej klatki.")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for name, code in ARUCO_DICTS.items():
        aruco_dict = cv2.aruco.getPredefinedDictionary(code)
        detector = cv2.aruco.ArucoDetector(aruco_dict)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None and len(ids):
            return name, int(ids[0][0])
    raise ValueError("Nie znaleziono markera w pierwszej klatce żadnym słownikiem.")


def compute_real_offset_cm(
    px: float,
    image_width: int,
    horizontal_fov_deg: float = 60.0,
    distance_cm: float = 120.0,
) -> float:
    deg_per_pixel = horizontal_fov_deg / image_width
    dx_pixels = px - (image_width / 2.0)
    angle_rad = math.radians(dx_pixels * deg_per_pixel)
    return distance_cm * math.tan(angle_rad)


# ---------------------------------------------------------------------------
# Główna klasa analizy
# ---------------------------------------------------------------------------

class ArucoTrampolineTracker:
    """Analizuje ruch trampoliny bazując na markerze ArUco."""

    def __init__(
        self,
        video_path: str | int,
        dict_name: str,
        marker_id: int | None,
        out_video: Path,
    ) -> None:
        if dict_name not in ARUCO_DICTS:
            raise ValueError(
                f"Nieznany słownik '{dict_name}'. Dostępne: {', '.join(ARUCO_DICTS)}"
            )
        self.video_path = video_path
        self.marker_id = marker_id
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])
        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dict,
            self._tuned_parameters(),
        )
        self.out_video = out_video
        # --- Tracker CSRT fallback ---
        try:
            self.tracker_csrt = cv2.TrackerCSRT_create()
        except AttributeError:
            self.tracker_csrt = cv2.legacy.TrackerCSRT_create()
        self.track_mode = False          # True = CSRT, False = ArUco
        self.lost_counter = 0
        self.lost_thresh = 1             # 1 klatka przerwy → CSRT
        self.last_bbox = None
        self.csrt_ready = False          # czy tracker został zainicjalizowany

    @staticmethod
    def _tuned_parameters() -> cv2.aruco.DetectorParameters:
        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshConstant = 7
        params.minMarkerPerimeterRate = 0.02
        return params

    # -----------------------------------------
    # Public API
    # -----------------------------------------

    def run(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            sys.exit("❌ Nie można otworzyć pliku wideo / kamery.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w, h = int(cap.get(3)), int(cap.get(4))
        writer = cv2.VideoWriter(
            str(self.out_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )

        records: List[Tuple[float, float, float]] = []
        frame_idx = 0
        first_center: Tuple[float, float] | None = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp = frame_idx / fps
            corners, ids, _ = self.detector.detectMarkers(frame)

            # --------- ArUco widoczny ---------
            if ids is not None and len(ids):
                self.lost_counter = 0
                ids = ids.flatten()
                chosen_idx = 0
                if self.marker_id is not None:
                    matches = np.where(ids == self.marker_id)[0]
                    if len(matches):
                        chosen_idx = int(matches[0])
                    else:
                        ids = None  # brak oczekiwanego ID

                if ids is not None and len(ids):
                    c = corners[chosen_idx][0]
                    cx, cy = c.mean(axis=0)

                    # bbox + margines
                    pad = 15
                    x_min, y_min = np.min(c, axis=0)
                    x_max, y_max = np.max(c, axis=0)
                    x_min = max(0, int(x_min) - pad)
                    y_min = max(0, int(y_min) - pad)
                    x_max = min(w - 1, int(x_max) + pad)
                    y_max = min(h - 1, int(y_max) + pad)
                    bbox = (x_min, y_min, max(2, x_max - x_min), max(2, y_max - y_min))
                    self.last_bbox = bbox

                    # Inicjalizacja CSRT tylko jeśli potrzeba
                    if self.track_mode or not self.csrt_ready:
                        try:
                            self.tracker_csrt = cv2.TrackerCSRT_create()
                        except AttributeError:
                            self.tracker_csrt = cv2.legacy.TrackerCSRT_create()
                        self.tracker_csrt.init(frame, bbox)
                        self.csrt_ready = True
                    self.track_mode = False

                    if first_center is None:
                        first_center = (cx, cy)
                    cv2.polylines(frame, [c.astype(int)], True, (0, 255, 0), 2)
                    cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                    records.append((timestamp, cx, cy))
                    cv2.putText(
                        frame, f"id={ids[chosen_idx]}",
                        (int(cx) + 10, int(cy) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                    )
                else:
                    ids = None  # przeskok do fallback

            # --------- ArUco niewidoczny ---------
            if ids is None:
                self.lost_counter += 1
                if self.last_bbox and self.lost_counter >= self.lost_thresh:
                    self.track_mode = True

                if self.track_mode and self.csrt_ready:
                    ok_track, new_bbox = self.tracker_csrt.update(frame)
                else:
                    ok_track = False

                if ok_track:
                    x, y, w_box, h_box = new_bbox
                    cx, cy = x + w_box / 2, y + h_box / 2
                    cv2.rectangle(
                        frame, (int(x), int(y)),
                        (int(x + w_box), int(y + h_box)), (255, 0, 0), 2
                    )
                    cv2.circle(frame, (int(cx), int(cy)), 4, (255, 0, 0), -1)
                    records.append((timestamp, cx, cy))
                else:
                    cv2.putText(frame, "LOST", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            writer.write(frame)
            cv2.imshow("Aruco tracker", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Esc
                break
            frame_idx += 1

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        if not records:
            sys.exit("❌  Nie odnotowano ani jednego wystąpienia markera.")
        self._postprocess(records, first_center, w)

    # ----------------------------------------- post-processing -----------------------------------------

    def _postprocess(
        self,
        records: List[Tuple[float, float, float]],
        first_center: Tuple[float, float] | None,
        image_width: int,
    ) -> None:
        df = pd.DataFrame(records, columns=["time_s", "cx", "cy"])
        df.to_csv("positions.csv", index=False)
        print("✔  Zapisano positions.csv (", len(df), "wierszy)")

        if first_center is not None:
            df["dy_px"] = df["cy"] - first_center[1]
            df["offset_cm"] = df["cx"].apply(
                lambda x: compute_real_offset_cm(x, image_width)
            )
            plt.figure(figsize=(10, 5))
            plt.plot(df["time_s"], df["dy_px"], label="Δy [px]")
            plt.xlabel("Czas [s]")
            plt.ylabel("Δy [px]")
            plt.title("Wychylenie pionowe markera w czasie")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("deflection_plot.png", dpi=150)
            plt.show()
            print("✔  Zapisano deflection_plot.png")


# ---------------------------------------------------------------------------
# Uruchamianie z linii poleceń
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automatyczny tracker markera ArUco")
    parser.add_argument(
        "--video",
        default="/Users/bartlomiejostasz/lot/n/git/1.mov",
        help="Ścieżka do pliku wideo (domyślnie ten plik); podaj 0, aby użyć kamerki",
    )
    parser.add_argument(
        "--dict",
        default="AUTO",
        help="Nazwa słownika ArUco (np. DICT_4X4_50) lub AUTO do samodetekcji",
    )
    parser.add_argument(
        "--marker-id",
        type=int,
        default=None,
        help="ID markera do śledzenia (jeśli puste – pierwszy wykryty)",
    )
    parser.add_argument(
        "--out",
        default="output_annotated.mp4",
        help="Ścieżka zapisu wideo z adnotacjami",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_source: str | int = int(
        args.video) if args.video.isdigit() else args.video
    if args.dict == "AUTO":
        detected_dict, detected_id = auto_detect_dict_and_id(video_source)
        print(f"🔍 AUTO: znaleziono {detected_dict} / id={detected_id}")
        args.dict = detected_dict
        if args.marker_id is None:
            args.marker_id = detected_id

    tracker = ArucoTrampolineTracker(
        video_path=video_source,
        dict_name=args.dict,
        marker_id=args.marker_id,
        out_video=Path(args.out),
    )
    tracker.run()


if __name__ == "__main__":
    main()