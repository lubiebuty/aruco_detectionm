# split_train_val.py  – uruchom w katalogu marker_dataset
import random, shutil
from pathlib import Path

root = Path("/Users/bartlomiejostasz/PYCH/aruco_detection/marker_dataset")           # albo pełna ścieżka
train_img_dir = root / "images/train"
train_lbl_dir = root / "labels/train"
val_img_dir   = root / "images/val"
val_lbl_dir   = root / "labels/val"

val_img_dir.mkdir(parents=True, exist_ok=True)
val_lbl_dir.mkdir(parents=True, exist_ok=True)

imgs = sorted(train_img_dir.glob("*.jpg")) + sorted(train_img_dir.glob("*.png"))
random.seed(0)
val_split = 0.2
n_val = int(len(imgs) * val_split)
val_imgs = random.sample(imgs, n_val)

for img_path in val_imgs:
    lbl_path = train_lbl_dir / f"{img_path.stem}.txt"
    # przeniesienie (mv) → użyj shutil.move
    shutil.move(img_path, val_img_dir / img_path.name)
    shutil.move(lbl_path, val_lbl_dir / lbl_path.name)

print(f"Przeniesiono {n_val} par do walidacji.")