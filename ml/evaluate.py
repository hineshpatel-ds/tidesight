from __future__ import annotations

import json

import numpy as np

from ml.common import DATA_DIR, ML_DIR, RUNS_DIR, SPLITS, frame_name, pick_device

THRESHOLDS = [0.10, 0.15, 0.18, 0.20, 0.25]


def load_labels(txt, w, h):
    out = []
    for line in txt.read_text().split("\n"):
        p = line.split()
        if len(p) == 5:
            _, cx, cy, bw, bh = map(float, p)
            out.append((cx * w, cy * h, max(bw * w, bh * h)))
    return out


def match(preds, labels):
    pairs = sorted((np.hypot(px - lx, py - ly) / ls, i, j)
                   for i, (px, py) in enumerate(preds) for j, (lx, ly, ls) in enumerate(labels))
    used_p, used_l, n = set(), set(), 0
    for d, i, j in pairs:
        if d > 1.0:
            break
        if i not in used_p and j not in used_l:
            used_p.add(i)
            used_l.add(j)
            n += 1
    return n


def main() -> None:
    from ultralytics import YOLO

    device = pick_device()
    report = {}
    for fold in ("foldA", "foldB"):
        spec = SPLITS["folds"][fold]
        weights = RUNS_DIR / fold / "weights" / "best.pt"
        if not weights.exists():
            print(f"skip {fold}: {weights} missing (python -m ml.train {fold})")
            continue
        model = YOLO(str(weights))
        frames = sorted((DATA_DIR / spec["test_burst"]).glob("*.png"))
        per_t = {t: dict(pred=[], gt=[], tp=0, npred=0) for t in THRESHOLDS}
        for img in frames:
            res = model.predict(str(img), imgsz=1280, conf=min(THRESHOLDS), device=device, verbose=False)[0]
            h, w = res.orig_shape
            labels = load_labels(img.with_suffix(".txt"), w, h)
            xyxy, conf = res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()
            for t in THRESHOLDS:
                keep = xyxy[conf >= t]
                centres = [((a + c) / 2, (b + d) / 2) for a, b, c, d in keep]
                s = per_t[t]
                s["pred"].append(len(keep))
                s["gt"].append(len(labels))
                s["tp"] += match(centres, labels)
                s["npred"] += len(keep)
        hand_t = spec["test_hand_count"]
        hand_img = DATA_DIR / "test_handcounted" / f"{frame_name(int(hand_t))}.png"
        hc = model.predict(str(hand_img), imgsz=1280, conf=min(THRESHOLDS), device=device, verbose=False)[0]
        hconf = hc.boxes.conf.cpu().numpy()
        rows = []
        for t in THRESHOLDS:
            s = per_t[t]
            pred, gt = np.array(s["pred"]), np.array(s["gt"])
            rows.append(dict(
                conf=t, frames=len(gt), labelled=int(gt.sum()), predicted=int(pred.sum()),
                total_error_pct=round(100 * (pred.sum() - gt.sum()) / max(1, gt.sum()), 1),
                mae_per_frame=round(float(np.abs(pred - gt).mean()), 1),
                pearson_r=round(float(np.corrcoef(pred, gt)[0, 1]), 3) if pred.std() and gt.std() else None,
                precision=round(s["tp"] / max(1, s["npred"]), 3), recall=round(s["tp"] / max(1, gt.sum()), 3),
                hand_count_frame=f"t={hand_t}s", hand_count=SPLITS["hand_counts"][hand_t]["count"],
                model_count_on_hand_frame=int((hconf >= t).sum())))
        report[fold] = dict(trained_on=spec["train"], tested_on=spec["test_burst"], rows=rows)
        print(f"\n### {fold}: trained on {spec['train']}, tested on {spec['test_burst']} "
              f"({rows[0]['frames']} frames, {rows[0]['labelled']} labelled fish)")
        print("| conf | predicted | total err | MAE/frame | r | precision | recall | hand-count frame |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['conf']:.2f} | {r['predicted']} | {r['total_error_pct']:+.1f}% | {r['mae_per_frame']} | "
                  f"{r['pearson_r']} | {r['precision']} | {r['recall']} | "
                  f"{r['model_count_on_hand_frame']} vs {r['hand_count']} ({r['hand_count_frame']}) |")
    out = ML_DIR / "results"
    out.mkdir(exist_ok=True)
    (out / "heldout.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out / 'heldout.json'}")


if __name__ == "__main__":
    main()
