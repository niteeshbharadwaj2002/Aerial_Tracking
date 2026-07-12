import cv2
import random
from pathlib import Path

IMG_DIR = Path("data/yolo/images/train")
LBL_DIR = Path("data/yolo/labels/train")
OUT_DIR = Path("outputs/sanity_check")
N_SAMPLES = 10

CLASS_NAMES = ["pedestrian", "car", "van", "truck"]
COLORS = [(0, 255, 0), (255, 0, 0), (0, 165, 255), (0, 0, 255)]

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    img_paths = list(IMG_DIR.glob("*.jpg"))
    if not img_paths:
        print(f"No images found in {IMG_DIR}")
        return

    samples = random.sample(img_paths, min(N_SAMPLES, len(img_paths)))

    for img_path in samples:
        im = cv2.imread(str(img_path))
        if im is None:
            print(f"Could not read {img_path}")
            continue
        h, w = im.shape[:2]

        lbl_path = LBL_DIR / (img_path.stem + ".txt")
        if not lbl_path.exists():
            print(f"No label file for {img_path.name}, skipping")
            continue

        with open(lbl_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines:
            cls, xc, yc, bw, bh = map(float, line.split())
            cls = int(cls)
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            color = COLORS[cls % len(COLORS)]
            label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)

            cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
            cv2.putText(im, label, (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        out_path = OUT_DIR / f"sanity_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), im)
        print(f"Wrote {out_path} ({len(lines)} boxes)")

if __name__ == "__main__":
    main()