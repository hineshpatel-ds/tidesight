from __future__ import annotations

import argparse
import random
import shutil

from ml.common import DATA_DIR, RUNS_DIR, SPLITS, ML_DIR, pick_device

SETTINGS = dict(model="yolo11s.pt", epochs=200, imgsz=1280, batch=4, patience=0, seed=0, val_frac=0.15,
                aug=dict(fliplr=0.5, flipud=0.0, degrees=0, mosaic=1.0, close_mosaic=20,
                         hsv_h=0.01, hsv_s=0.3, hsv_v=0.3))


def build_dataset(fold: str) -> tuple[str, int, int]:
    images = sorted(p for burst in SPLITS["folds"][fold]["train"]
                    for p in (DATA_DIR / burst).glob("*.png") if p.with_suffix(".txt").exists())
    if len(images) < 5:
        raise SystemExit(f"Only {len(images)} labelled frames for {fold}; run python -m ml.make_frames first.")
    random.Random(SETTINGS["seed"]).shuffle(images)
    n_val = max(2, int(len(images) * SETTINGS["val_frac"]))
    root = ML_DIR / "datasets" / fold
    if root.exists():
        shutil.rmtree(root)
    for split, items in (("val", images[:n_val]), ("train", images[n_val:])):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for p in items:
            shutil.copy(p, root / "images" / split / p.name)
            shutil.copy(p.with_suffix(".txt"), root / "labels" / split / p.with_suffix(".txt").name)
    yaml = root / "fish.yaml"
    yaml.write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: fish\n")
    return str(yaml), len(images) - n_val, n_val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fold", choices=list(SPLITS["folds"]))
    ap.add_argument("--epochs", type=int, default=SETTINGS["epochs"])
    a = ap.parse_args()
    from ultralytics import YOLO

    yaml, n_train, n_val = build_dataset(a.fold)
    print(f"{a.fold}: {n_train} train / {n_val} val frames from {SPLITS['folds'][a.fold]['train']}")
    model = YOLO(SETTINGS["model"])
    model.train(data=yaml, epochs=a.epochs, imgsz=SETTINGS["imgsz"], batch=SETTINGS["batch"],
                device=pick_device(), project=str(RUNS_DIR), name=a.fold, exist_ok=True,
                patience=SETTINGS["patience"], seed=SETTINGS["seed"], deterministic=True,
                plots=True, verbose=False, **SETTINGS["aug"])
    print("best weights:", RUNS_DIR / a.fold / "weights" / "best.pt")


if __name__ == "__main__":
    main()
