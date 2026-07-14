import os
import time
import glob
import platform

import numpy as np
import cv2
import onnxruntime as ort
from ultralytics import YOLO

# Config
WEIGHTS_PT = "outputs/runs/visdrone_baseline-6/weights/best.pt"
WEIGHTS_ONNX_FP32 = "outputs/runs/visdrone_baseline-6/weights/best.onnx"
WEIGHTS_ONNX_INT8 = "outputs/runs/visdrone_baseline-6/weights/best_int8.onnx"

TEST_IMAGES_DIR = "data/yolo/images/test"
OUTPUT_DIR = "outputs/export_benchmark"
RESULTS_MD = os.path.join(OUTPUT_DIR, "benchmark_results.md")

IMGSZ = 512
N_WARMUP = 5     # iterations discarded before timing starts
N_RUNS = 30      # timed iterations per config (images are cycled if fewer exist)

# Export
def export_onnx_fp32():
    """Export best.pt -> ONNX FP32 via Ultralytics."""
    if os.path.exists(WEIGHTS_ONNX_FP32):
        print(f"[skip] {WEIGHTS_ONNX_FP32} already exists.")
        return WEIGHTS_ONNX_FP32

    print("Exporting ONNX FP32 ...")
    model = YOLO(WEIGHTS_PT)
    exported_path = model.export(format="onnx", imgsz=IMGSZ, simplify=True, opset=12)
    exported_path = str(exported_path)

    if exported_path != WEIGHTS_ONNX_FP32 and os.path.exists(exported_path):
        os.replace(exported_path, WEIGHTS_ONNX_FP32)

    print(f"Saved: {WEIGHTS_ONNX_FP32}")
    return WEIGHTS_ONNX_FP32


def export_onnx_int8():
    """Post-training dynamic quantization of the FP32 ONNX graph -> INT8."""""
    if os.path.exists(WEIGHTS_ONNX_INT8):
        print(f"[skip] {WEIGHTS_ONNX_INT8} already exists.")
        return WEIGHTS_ONNX_INT8

    from onnxruntime.quantization import quantize_dynamic, QuantType

    print("Quantizing ONNX FP32 -> INT8 (dynamic quantization) ...")
    quantize_dynamic(
        model_input=WEIGHTS_ONNX_FP32,
        model_output=WEIGHTS_ONNX_INT8,
        weight_type=QuantType.QUInt8,
    )
    print(f"Saved: {WEIGHTS_ONNX_INT8}")
    return WEIGHTS_ONNX_INT8


# Preprocessing (shared by both ONNX configs)
def preprocess(image_path, imgsz=IMGSZ):
    """Letterbox-resize + BGR->RGB + HWC->CHW + normalize to [0,1]."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    h, w = img.shape[:2]
    scale = imgsz / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh))

    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(chw, axis=0)  # (1, 3, imgsz, imgsz)


def get_test_images():
    images = sorted(glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg")))
    images += sorted(glob.glob(os.path.join(TEST_IMAGES_DIR, "*.png")))
    if not images:
        raise FileNotFoundError(
            f"No test images found in {TEST_IMAGES_DIR}. "
            "Run src/split_dataset.py and src/data_prep.py first."
        )
    return images


# Benchmarks
def benchmark_pytorch_fp32(image_paths):
    print("Benchmarking PyTorch FP32 (CPU) ...")
    model = YOLO(WEIGHTS_PT)
    model.to("cpu")

    cycle = (image_paths * ((N_WARMUP + N_RUNS) // len(image_paths) + 1))[: N_WARMUP + N_RUNS]

    for path in cycle[:N_WARMUP]:
        model.predict(path, imgsz=IMGSZ, device="cpu", verbose=False)

    latencies = []
    for path in cycle[N_WARMUP:]:
        t0 = time.perf_counter()
        model.predict(path, imgsz=IMGSZ, device="cpu", verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000)

    return summarize(latencies, WEIGHTS_PT)


def benchmark_onnx(model_path, image_paths, label):
    print(f"Benchmarking {label} (CPU) ...")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    cycle = (image_paths * ((N_WARMUP + N_RUNS) // len(image_paths) + 1))[: N_WARMUP + N_RUNS]
    tensors = [preprocess(p) for p in cycle]

    for t in tensors[:N_WARMUP]:
        session.run(None, {input_name: t})

    latencies = []
    for t in tensors[N_WARMUP:]:
        t0 = time.perf_counter()
        session.run(None, {input_name: t})
        latencies.append((time.perf_counter() - t0) * 1000)

    return summarize(latencies, model_path)


def summarize(latencies_ms, model_path):
    avg_latency = float(np.mean(latencies_ms))
    fps = 1000.0 / avg_latency
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    return {
        "latency_ms": round(avg_latency, 2),
        "fps": round(fps, 2),
        "size_mb": round(size_mb, 2),
    }


# Main
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    export_onnx_fp32()
    export_onnx_int8()

    image_paths = get_test_images()

    results = {
        "PyTorch FP32": benchmark_pytorch_fp32(image_paths),
        "ONNX FP32": benchmark_onnx(WEIGHTS_ONNX_FP32, image_paths, "ONNX FP32"),
        "ONNX INT8": benchmark_onnx(WEIGHTS_ONNX_INT8, image_paths, "ONNX INT8"),
    }

    hardware = f"{platform.processor() or platform.machine()}, {platform.system()} {platform.release()}"

    lines = [
        "# Deployment Benchmarks",
        "",
        f"**Test hardware:** {hardware} (CPU-only inference for all configs)",
        "",
        "| Config | Latency (ms) | FPS | Model size (MB) |",
        "|---|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(f"| {name} | {r['latency_ms']} | {r['fps']} | {r['size_mb']} |")

    report = "\n".join(lines) + "\n"
    with open(RESULTS_MD, "w") as f:
        f.write(report)

    print("\n" + report)
    print(f"Results written to {RESULTS_MD} — paste the table into README.md's Deployment benchmarks section.")


if __name__ == "__main__":
    main()
