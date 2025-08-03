from pathlib import Path
root = Path("/Users/bartlomiejostasz/PYCH/aruco_detection/marker_dataset")
for split in ("train", "val"):
    imgs = len(list((root/f"images/{split}").glob("*.*")))
    lbls = len(list((root/f"labels/{split}").glob("*.txt")))
    print(split, imgs, lbls)