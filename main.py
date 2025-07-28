#!/usr/bin/env python3
"""Aruco‑based trampoline deflection tracker.

Usage (terminal):
    python aruco_trampoline_tracker.py \
        --video  "path/to/input.mov" \
        --dict   DICT_6X6_1000 \
        --marker-id 2

The script:
1.  Ładuje nagranie wideo (lub kamerę: --video 0).
2.  Wykrywa podany (lub pierwszy napotkany) znacznik ArUco.
3.  Śledzi jego środek klatka po klatce, zapisując wyniki do CSV.
4.  Rysuje obwiednię + środek i zapisuje wideo *output_annotated.mp4*.
5.  Po zakończeniu generuje wykres pionowego wychylenia.

Zależności:  opencv‑contrib‑python, pandas, matplotlib, numpy.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

# ---------------------------------------------------------------------------
# Stałe i funkcje pomocnicze
# ---------------------------------------------------------------------------

ARUCO_DICTS: Dict[str, int] = {
    name: getattr(cv2.aruco, name)
    for name in dir(cv2.aruco)
    if name.startswith("DICT_")
}


def compute_real_offset_cm(
    px: float,
    image_width: int,
    horizontal_fov_deg: float = 60.0,
    distance_cm: float = 120.0,
) -> float:
    """Przybliżone przesunięcie boczne w cm z pikseli.

    Zakładamy prostą geometrię perspektywy:  tan(theta) = offset / distance.
    """
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

    @staticmethod
    def _tuned_parameters() -> cv2.aruco.DetectorParameters:  # type: ignore[name-defined]
        """Zwrot parametrów detektora dostrojonych do słabego oświetlenia / małych markerów."""
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
            sys.exit("❌  Nie można otworzyć pliku wideo / kamery.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w, h = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.out_video), fourcc, fps, (w, h))

        records: List[Tuple[float, float, float]] = []  # (t, cx, cy)
        frame_idx = 0
        first_center: Tuple[float, float] | None = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp = frame_idx / fps
            corners, ids, _ = self.detector.detectMarkers(frame)

            if ids is not None:
                ids = ids.flatten()
                chosen_idx = 0
                if self.marker_id is not None:
                    indices = np.where(ids == self.marker_id)[0]
                    if len(indices):
                        chosen_idx = int(indices[0])
                    else:
                        # marker o podanym ID nieobecny w tej klatce
                        ids = None
                if ids is not None:
                    c = corners[chosen_idx][0]
                    cx, cy = c.mean(axis=0)
                    if first_center is None:
                        first_center = (cx, cy)
                    # rysuj
                    cv2.polylines(frame, [c.astype(int)], True, (0, 255, 0), 2)
                    cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                    records.append((timestamp, cx, cy))
                    cv2.putText(
                        frame,
                        f"id={ids[chosen_idx]}",
                        (int(cx) + 10, int(cy) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
            else:
                cv2.putText(
                    frame,
                    "LOST",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )

            writer.write(frame)
            cv2.imshow("Aruco tracker", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Esc
                break
            frame_idx += 1

        # Cleanup
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        if not records:
            sys.exit("❌  Nie odnotowano ani jednego wystąpienia markera.")

        self._postprocess(records, first_center, w)

    # -----------------------------------------
    # Post‑processing: CSV + wykres
    # -----------------------------------------

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
        default="0",
        help="Ścieżka do pliku wideo lub 0 (domyślnie) dla kamerki",
    )
    parser.add_argument(
        "--dict",
        default="DICT_6X6_1000",
        help="Nazwa słownika ArUco (np. DICT_4X4_50)",
    )
    parser.add_argument(
        "--marker-id",
        type=int,
        default=None,
        help="ID markera do śledzenia (jeśli pusty – pierwszy wykryty)",
    )
    parser.add_argument(
        "--out",
        default="output_annotated.mp4",
        help="Ścieżka zapisu wideo z adnotacjami",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_source: str | int = (
        int(args.video) if args.video.isdigit() else args.video
    )
    tracker = ArucoTrampolineTracker(
        video_path=video_source,
        dict_name=args.dict,
        marker_id=args.marker_id,
        out_video=Path(args.out),
    )
    tracker.run()


if __name__ == "__main__":
    main()
