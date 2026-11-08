from __future__ import annotations

import sys
from pathlib import Path

import cv2

WIN = "label_fish  (click fish | n next | p prev | u undo | [ ] ; ' box size | q quit)"
MAX_W, MAX_H = 1600, 900


def load(txt: Path, w: int, h: int) -> list[list[float]]:
    boxes = []
    if txt.exists():
        for line in txt.read_text().split("\n"):
            p = line.split()
            if len(p) == 5:
                _, cx, cy, bw, bh = map(float, p)
                boxes.append([cx * w, cy * h, bw * w, bh * h])
    return boxes


def save(txt: Path, boxes, w: int, h: int) -> None:
    txt.write_text("".join(f"0 {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}\n"
                           for cx, cy, bw, bh in boxes))


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "labeling/train")
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".png"})
    if not images:
        sys.exit(f"No images in {folder}")
    idx = next((i for i, p in enumerate(images) if not p.with_suffix(".txt").exists()), 0)
    box_w, box_h = 56, 20
    state = {"boxes": [], "scale": 1.0}

    def on_mouse(event, x, y, flags, _):
        s = state["scale"]
        ix, iy = x / s, y / s
        if event == cv2.EVENT_LBUTTONDOWN:
            state["boxes"].append([ix, iy, box_w, box_h])
        elif event == cv2.EVENT_RBUTTONDOWN and state["boxes"]:
            j = min(range(len(state["boxes"])),
                    key=lambda k: (state["boxes"][k][0] - ix) ** 2 + (state["boxes"][k][1] - iy) ** 2)
            state["boxes"].pop(j)

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)
    while True:
        path = images[idx]
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        txt = path.with_suffix(".txt")
        state["boxes"] = load(txt, w, h)
        state["scale"] = min(1.0, MAX_W / w, MAX_H / h)
        while True:
            s = state["scale"]
            vis = cv2.resize(img, (int(w * s), int(h * s))) if s != 1.0 else img.copy()
            for cx, cy, bw, bh in state["boxes"]:
                cv2.rectangle(vis, (int((cx - bw / 2) * s), int((cy - bh / 2) * s)),
                              (int((cx + bw / 2) * s), int((cy + bh / 2) * s)), (0, 0, 255), 1)
            cv2.putText(vis, f"{idx + 1}/{len(images)}  {path.name}  fish: {len(state['boxes'])}  box {box_w}x{box_h}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow(WIN, vis)
            k = cv2.waitKey(30) & 0xFF
            if k in (ord("n"), ord(" ")):
                save(txt, state["boxes"], w, h); idx = min(idx + 1, len(images) - 1); break
            if k == ord("p"):
                save(txt, state["boxes"], w, h); idx = max(idx - 1, 0); break
            if k in (ord("q"), 27):
                save(txt, state["boxes"], w, h); cv2.destroyAllWindows()
                done = sum(p.with_suffix(".txt").exists() for p in images)
                print(f"Saved. {done}/{len(images)} images labelled."); return
            if k == ord("u") and state["boxes"]:
                state["boxes"].pop()
            if k == ord("c"):
                state["boxes"] = []
            if k == ord("["): box_w = max(8, box_w - 4)
            if k == ord("]"): box_w += 4
            if k == ord(";"): box_h = max(6, box_h - 2)
            if k == ord("'"): box_h += 2


if __name__ == "__main__":
    main()
