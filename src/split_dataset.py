import random
import shutil
from pathlib import Path

random.seed(42)  # reproducible split

SRC_IMG_DIR = Path("data/DET-subset/images")
SRC_ANN_DIR = Path("data/DET-subset/annotations")

OUT_ROOT = Path("data/VisDrone2019-DET-split")
TRAIN_RATIO = 0.8
TEST_COUNT = 20  # 1% of 2000

def copy_pairs(pairs, split_name):
    out_img = OUT_ROOT / split_name / "images"
    out_ann = OUT_ROOT / split_name / "annotations"
    out_img.mkdir(parents=True, exist_ok=True)
    out_ann.mkdir(parents=True, exist_ok=True)
    for img_path, ann_path in pairs:
        shutil.copy2(img_path, out_img / img_path.name)
        shutil.copy2(ann_path, out_ann / ann_path.name)

def main():
    img_paths = sorted(SRC_IMG_DIR.glob("*.jpg"))
    pairs = [(p, SRC_ANN_DIR / (p.stem + ".txt")) for p in img_paths]
    pairs = [(img, ann) for img, ann in pairs if ann.exists()]

    missing = len(img_paths) - len(pairs)
    if missing:
        print(f"Warning: {missing} images had no matching annotation, skipped")

    random.shuffle(pairs)

    # --- Test set: 20 images, fully held out ---
    test_pairs = pairs[:TEST_COUNT]
    remaining_pairs = pairs[TEST_COUNT:]  # 1980 images

    # --- Remaining 1980 images: 80/20 train/val split ---
    split_idx = int(len(remaining_pairs) * TRAIN_RATIO)
    train_pairs = remaining_pairs[:split_idx]   # 1584
    val_pairs = remaining_pairs[split_idx:]     # 396

    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")
    copy_pairs(test_pairs, "test")

    print(f"Total source pairs: {len(pairs)}")
    print(f"Test (held out, NOT in train/val): {len(test_pairs)}")
    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")

if __name__ == "__main__":
    main()