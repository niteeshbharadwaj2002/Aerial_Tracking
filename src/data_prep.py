import shutil
from pathlib import Path
from PIL import Image

# VisDrone class id -> your YOLO class id
CLASS_MAP = {
    1: 0,  # pedestrian
    4: 1,  # car
    5: 2,  # van
    6: 3,  # truck
}
YOLO_CLASS_NAMES = ["pedestrian", "car", "van", "truck"]

def convert_split(img_dir, ann_dir, out_img_dir, out_lbl_dir):
    img_dir, ann_dir = Path(img_dir), Path(ann_dir)
    out_img_dir, out_lbl_dir = Path(out_img_dir), Path(out_lbl_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    n_images, n_boxes, n_dropped = 0, 0, 0

    for img_path in sorted(img_dir.glob("*.jpg")):
        ann_path = ann_dir / (img_path.stem + ".txt")
        if not ann_path.exists():
            continue

        with Image.open(img_path) as im:
            w, h = im.size

        yolo_lines = []
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                x, y, bw, bh = map(float, parts[0:4])
                category = int(parts[5])

                if category not in CLASS_MAP:
                    n_dropped += 1
                    continue
                if bw <= 0 or bh <= 0:
                    n_dropped += 1
                    continue

                cls_id = CLASS_MAP[category]
                x_center = (x + bw / 2) / w
                y_center = (y + bh / 2) / h
                bw_norm = bw / w
                bh_norm = bh / h

                x_center = min(max(x_center, 0), 1)
                y_center = min(max(y_center, 0), 1)
                bw_norm = min(max(bw_norm, 0), 1)
                bh_norm = min(max(bh_norm, 0), 1)

                yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}")
                n_boxes += 1

        with open(out_lbl_dir / (img_path.stem + ".txt"), "w") as f:
            f.write("\n".join(yolo_lines))

        dst_img = out_img_dir / img_path.name
        if not dst_img.exists():
            shutil.copy2(img_path.resolve(), dst_img)

        n_images += 1

    print(f"{img_dir.name}: {n_images} images, {n_boxes} boxes kept, {n_dropped} boxes dropped")


if __name__ == "__main__":
    SRC_ROOT = Path("data/VisDrone2019-DET-split")
    OUT_ROOT = Path("data/yolo")

    for split in ["train", "val", "test"]:
        convert_split(
            img_dir=SRC_ROOT / split / "images",
            ann_dir=SRC_ROOT / split / "annotations",
            out_img_dir=OUT_ROOT / "images" / split,
            out_lbl_dir=OUT_ROOT / "labels" / split,
        )