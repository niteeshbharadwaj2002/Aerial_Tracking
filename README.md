# Aerial_Tracking

**Detect and track pedestrians and vehicles from UAV aerial footage — optimized for small objects, high scene density, and edge deployment.**

---

## Demo

![Tracking demo — placeholder](outputs/gif/tracked_output_3.gif)
![Tracking demo — placeholder](outputs/gif/tracked_output_1.gif)
---

## Problem Statement

Aerial object detection identifies and localizes objects in images or video captured from an unmanned aerial vehicle (UAV), typically at oblique or nadir viewpoints hundreds of meters above the ground. It is substantially harder than generic object detection: targets occupy only a few pixels (small object scale), scenes contain dozens or hundreds of instances (high density), the camera platform moves (motion blur and shifting background), and appearance varies with altitude, angle, and lighting.

These constraints matter directly for aerospace and UAV applications — runway and landing-zone monitoring, perimeter surveillance, traffic flow analysis, and infrastructure inspection all depend on reliable perception from a moving aerial platform, often with strict latency and power budgets onboard.

---

## Aerospace / Real-World Relevance

This pipeline maps directly to UAV-based situational awareness: persistent surveillance over runways and landing zones, monitoring vehicle and pedestrian activity around airports or bases, and automated infrastructure inspection where a moving aerial platform must detect small objects in real time. The edge-deployment focus (YOLOv8n + ONNX) reflects the compute and power constraints of actual onboard flight hardware.

---

## Approach

### Dataset
- **VisDrone2019-DET** — 2,000-image subset (`data/VisDrone2019-DET-train/`), split 80 / 19.8 / 1 % into **1,584 train / 396 val / 20 test** (`src/split_dataset.py`).
- **VisDrone-MOT** — sequence subset for tracking evaluation (`data/MOT-subset/sequences/`).
- **Classes kept:** pedestrian, car, van, truck — the primary road-traffic actors for UAV surveillance.
- **Classes dropped:** people, bicycle, tricycle, awning-tricycle, bus, motor, others — rare, heavily overlapping, or weakly represented at aerial scale; dropping them reduces label noise and class imbalance while keeping the task focused on traffic monitoring.

### Detection
- **Model:** YOLOv8n (nano) — swap to `yolov8s.pt` in `src/train.py` if GPU memory allows.
- **Input size:** 512 px (configurable via `imgsz` in `src/train.py`).
- **Transfer learning:** COCO-pretrained weights (`models/yolov8n.pt`).
- **Training:** 100 epochs, batch 16, early stopping (`patience=15`), seed 42.

### Tracking
- **Algorithm:** SORT — Kalman-filter motion prediction + IOU cost matrix + Hungarian assignment (`src/track.py`).
- **Why SORT over DeepSORT:** No appearance embedding network → lower compute and memory, suitable for edge deployment. Aerial scenes often have moderate occlusion and spatial separation between instances, where motion-based association is sufficient for a baseline.

### Deployment
- **Export:** ONNX (FP32 and INT8 quantization) via `src/export_benchmark.py`.
- **Benchmarks:** Side-by-side latency, FPS, and model-size comparison across PyTorch FP32, ONNX FP32, and ONNX INT8 runtimes.

---

## Results

### Detection metrics

Best checkpoint from run `visdrone_baseline-6` (validation set, epoch 88 — highest mAP@0.5:0.95):

| Metric | Value |
|---|---|
| mAP@0.5 | 0.323 |
| mAP@0.5:0.95 | 0.178 |
| Precision | 0.467 |
| Recall | 0.343 |

**Test set (held-out, 20 images):**

| Metric | Value |
|---|---|
| mAP@0.5 | 0.314 |
| mAP@0.5:0.95 | 0.192 |
| Precision | 0.562 |
| Recall | 0.293 |

### Deployment benchmarks

| Config | Latency (ms) | FPS | Model size (MB) |
|---|---|---|---|
| PyTorch FP32 | 18.75 | 53.33 | 5.94 |
| ONNX FP32 | 19.18 | 52.14 | 11.64 |
| ONNX INT8 | 30.84 | 32.42 | 3.14 |

**Test hardware:** Apple M3 (Mac Air), CPU-only inference for all configs.  
All numbers above are from a development workstation. They do **not** represent onboard flight-hardware performance.

> **Note:** ONNX INT8 (dynamic quantization) is smallest on disk (~2x smaller than FP32) but *slower* here — Apple Silicon's ARM64 CPU lacks the fast native INT8 GEMM kernels that x86 (VNNI) has, so `onnxruntime` dequantizes weights back to FP32 at runtime before each matmul, adding overhead without a compute win. INT8 would likely show its expected speedup on x86 CPUs with VNNI support, or via a hardware-accelerated path (CoreML/ANE execution provider, TensorRT, etc.) rather than plain CPU dynamic quantization.

---

## Sample Outputs

### Detection — success case

![Detection success](docs/assets/detection_success.jpg)

> Dense traffic scene (test set) — 48 cars, 12 pedestrians, and 1 van correctly detected, showing the model holds up in high-density conditions where instances are packed closely together.

### Detection — failure case

![Detection failure](docs/assets/detection_failure.jpg)

> Dense pedestrian scene (test set) — pedestrian is the weakest class overall (Recall 0.128, mAP@0.5 0.134 on the test set), and this image illustrates why: several small/distant pedestrians in the crowd go undetected even though 24 were correctly found. Aerial altitude shrinks pedestrians to just a handful of pixels, which is the main driver of missed detections here.

---

## Limitations & Next Steps

| Area | Planned improvement |
|---|---|
| Class coverage | Expand beyond 4 classes (bus, bicycle, motor) |
| Tracking robustness | DeepSORT or ByteTrack for heavy occlusion |
| Sensors | Multi-camera / multi-drone fusion |
| Validation | Real onboard hardware benchmarking (Jetson, flight computer) |
| Export | Static-calibrated INT8 (vs. current dynamic quantization) for better accuracy/speed on x86 CPUs |

These are roadmap items, not blockers — the current baseline establishes a reproducible end-to-end path from VisDrone data to tracked aerial video.

---

## Repo Structure

```
Obj_Detection_and_Tracking/
├── data/
│   ├── visdrone.yaml                        # YOLO dataset config (paths + class names)
│   ├── VisDrone2019-DET-split/               # Raw DET subset, split by src/split_dataset.py
│   │   ├── train/{images,annotations}/
│   │   ├── val/{images,annotations}/
│   │   └── test/{images,annotations}/
│   ├── VisDrone2019-MOT-split/                # VisDrone-MOT sequences for tracking eval
│   │   ├── annotations/
│   │   └── sequences/                        # uav0000137_00458_v, uav0000268_05773_v, uav0000305_00000_v
│   └── yolo/                                 # YOLO-format images + labels (from src/data_prep.py)
│       ├── images/{train,val,test}/
│       └── labels/{train,val,test}/
├── models/
│   ├── yolov8n.pt                     # COCO-pretrained checkpoint (base for training)
│   └── yolo26n.pt
├── src/
│   ├── split_dataset.py               # 80/19.8/1 split of DET subset
│   ├── data_prep.py                   # VisDrone annotations → YOLO format
│   ├── train.py                       # YOLOv8 fine-tuning
│   ├── track.py                       # SORT tracker on MOT sequences
│   └── export_benchmark.py            # ONNX FP32/INT8 export + CPU latency/FPS/size benchmarks
├── tests/
│   ├── sanity_check.py                # Visualize random training labels
│   └── predict_test.py                # Test-set metrics + inference
├── docs/
│   └── assets/                        # README images (detection_success.jpg, detection_failure.jpg, ...)
├── outputs/                           # Generated artifacts (gitignored)
│   ├── runs/visdrone_baseline-6/      # Weights, curves, batch previews
│   ├── sanity_check/
│   ├── export_benchmark/              # benchmark_results.md + exported best.onnx / best_int8.onnx
│   ├── tracked/                       # tracked_output_1.mp4, tracked_output_2.mp4, tracked_output_3.mp4
│   └── gif/                           # tracked_output_1.gif, tracked_output_3.gif (README demo)
├── requirements.txt
├── pyproject.toml
└── README.md
```

> **Note:** `data/` is gitignored. Download VisDrone data separately (see Setup). Most of `outputs/` is generated locally by the scripts above; the demo GIFs under `outputs/gif/` are kept so they render in this README.

---

## Setup / Reproduce

### 1. Environment

```bash
python3 -m venv venv && source venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Data

Download [VisDrone2019](https://github.com/VisDrone/VisDrone-Dataset) and place subsets under `data/` as shown in the repo structure above. The 2,000-image DET subset is listed in `data/subset_list.txt`.

### 3. Data preparation

```bash
# Split 2000 images → train (1584) / val (396) / test (20)
python src/split_dataset.py

# Convert VisDrone bbox annotations → YOLO format (4 classes)
python src/data_prep.py

# Optional: verify labels visually
python tests/sanity_check.py
```

### 4. Train detector

```bash
python src/train.py
```

Weights saved to `outputs/runs/visdrone_baseline-6/weights/best.pt`.

### 5. Evaluate on test set

```bash
python tests/predict_test.py
```

### 6. Run tracking (SORT)

Edit the three paths at the top of `src/track.py` (`SEQ_DIR`, `MODEL_PATH`, `OUTPUT_PATH`), then:

```bash
python src/track.py
```

### 7. Export & benchmark

```bash
python src/export_benchmark.py
```

Or export manually with Ultralytics:

```bash
python -c "from ultralytics import YOLO; YOLO('outputs/runs/visdrone_baseline-6/weights/best.pt').export(format='onnx')"
```

---

## License

VisDrone dataset — see [VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset) for terms of use.  
YOLOv8 — [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
