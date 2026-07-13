import random
import shutil
from pathlib import Path

random.seed(42)  # reproducible split

SRC_IMG_DIR = Path("data/VisDrone2019-DET-train/images")
SRC_ANN_DIR = Path("data/VisDrone2019-DET-train/annotations")

OUT_ROOT = Path("data/VisDrone2019-DET-split")
TRAIN_RATIO = 0.8
TEST_COUNT = 20  # 1% of 2000

TEST_COUNT = 20
TRAIN_COUNT = 1584
VAL_COUNT = 396

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

    assert len(pairs) >= TEST_COUNT + TRAIN_COUNT + VAL_COUNT, "Not enough pairs for exact split"

    random.shuffle(pairs)

    test_pairs = pairs[:TEST_COUNT]
    train_pairs = pairs[TEST_COUNT:TEST_COUNT + TRAIN_COUNT]
    val_pairs = pairs[TEST_COUNT + TRAIN_COUNT:TEST_COUNT + TRAIN_COUNT + VAL_COUNT]

    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")
    copy_pairs(test_pairs, "test")

    print(f"Total source pairs: {len(pairs)}")
    print(f"Test: {len(test_pairs)} | Train: {len(train_pairs)} | Val: {len(val_pairs)}")

if __name__ == "__main__":
    main()