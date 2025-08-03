#!/usr/bin/env python3
"""
overlay_markers.py
──────────────────
Generuje wideo z naniesionymi boksami i środkami trzech markerów ArUco
(Left/Center/Right) na podstawie wyników z markers.csv.

Użycie (przykład):
    python overlay_markers.py \
        --video "/ścieżka/IMG_4699.MOV" \
        --csv   markers.csv \
        --out   overlay.mp4 \
        --fps   30
"""

from __future__ import annotations
import cv2
import pandas as pd
import argparse
from pathlib import Path
import numpy as np

# ---------------- CLI ----------------
parser = argparse.ArgumentParser(description="Twórz wideo z overlay markerów")
parser.add_argument(
    "--video",
    default="/Users/bartlomiejostasz/lot/n/3klatki/IMG_4699.MOV",
    help="Ścieżka do pliku wideo (domyślnie testowy IMG_4699.MOV)"
)
parser.add_argument("--csv",   default="markers.csv", help="CSV z wynikami")
parser.add_argument("--out",   default="overlay.mp4", help="Plik wyjściowy MP4")
parser.add_argument("--fps",   type=float, default=None,
                    help="FPS wyjściowy (domyślnie taki jak wejściowy)")
args = parser.parse_args()

# ---------------- Dane ----------------
csv_path = Path(args.csv)
if not csv_path.is_file():
    raise FileNotFoundError(f"Nie znaleziono CSV: {csv_path}")

df = pd.read_csv(csv_path)

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise RuntimeError(f"Nie można otworzyć wideo: {args.video}")

w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_in  = cap.get(cv2.CAP_PROP_FPS) or 25.0
fps_out = args.fps or fps_in

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(args.out), fourcc, fps_out, (w, h))

# Kolory ID → kolor RGB
rid2color = {2: (0, 255,   0),   # Left   – zielony
             3: (0,   0, 255),   # Center – czerwony
             4: (255,255,   0)}   # Right  – cyjan

GATE_PX = 100  # maksymalna odległość, by uznać że to ten sam marker

def fix_marker_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Przydziela marker_id na podstawie historycznej pozycji + bramkowania.
    • Każdy wykryty bbox jest przypisany do ID, jeżeli |cx - last_cx[ID]| < GATE_PX.
    • Unikamy zamian: jeśli dwa bboxy wpadają do tej samej bramki, wybieramy bliższy,
      a drugi zostaje rozpatrzony dalej.
    • Brak dopasowania → bieżący bbox dostaje wolny ID najbliższy w osi X
      (ale nie zmienia istniejących).
    """
    global last_cx
    output = []

    for frame_id, grp in df.groupby("frame"):
        detections = grp.copy()
        detections["assigned"] = False
        id_taken = set()

        # 1) próbuj przypisać przez gating do poprzedniej pozycji
        for rid in (2,3,4):
            if rid not in last_cx:
                continue
            # znajdź detekcję w bramce
            candidates = detections[~detections.assigned &
                                    (abs(detections.cx_px - last_cx[rid]) < GATE_PX)]
            if len(candidates):
                # wybierz najbliższą
                idx = candidates.iloc[(abs(candidates.cx_px - last_cx[rid])).argmin()].name
                detections.at[idx, "marker_id"] = rid
                detections.at[idx, "assigned"] = True
                id_taken.add(rid)

        # 2) pozostałe detekcje → przydział według odległości do niewziętych ID
        remaining_ids = [rid for rid in (2,3,4) if rid not in id_taken]
        remaining_det = detections[~detections.assigned]
        if len(remaining_det):
            # sortuj pozostałe bboxy po X (left→right)
            remaining_det = remaining_det.sort_values("cx_px")
            for det_idx, rid in zip(remaining_det.index, remaining_ids):
                detections.at[det_idx, "marker_id"] = rid
                detections.at[det_idx, "assigned"] = True

        # 3) uaktualnij last_cx tylko dla przydzielonych
        for _, row in detections[detections.assigned].iterrows():
            last_cx[int(row.marker_id)] = row.cx_px

        output.append(detections.drop(columns="assigned"))

    return pd.concat(output, ignore_index=True)

print("🎞  Generuję overlay…")
frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break

    rows = df[df.frame == frame_idx]
    for _, row in rows.iterrows():
        rid          = int(row.marker_id)
        cx, cy       = int(row.cx_px),  int(row.cy_px)
        x1, y1, x2, y2 = map(int, (row.bbox_x1, row.bbox_y1,
                                   row.bbox_x2, row.bbox_y2))
        color = rid2color.get(rid, (255,255,255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle   (frame, (cx, cy), 4, color, -1)
        cv2.putText  (frame, f"ID{rid}", (x1, y1-6),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    writer.write(frame)
    frame_idx += 1

cap.release()
writer.release()
print(f"✔  Zapisano {args.out}  ({frame_idx} klatek)")