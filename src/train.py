from ultralytics import YOLO
import torch

def main():
    print("CUDA available:", torch.cuda.is_available())

    model = YOLO("yolov8n.pt")  # nano baseline; swap to yolov8s.pt if GPU allows

    results = model.train(
        data="data/visdrone.yaml",
        epochs=100,
        patience=15,        # early stopping
        imgsz=512,
        batch=8,            # lower to 8/4 if CPU-only or low VRAM
        device=0 if torch.cuda.is_available() else "cpu",
        project="outputs/runs",
        name="visdrone_baseline",
        seed=42,
        val=True,
    )

if __name__ == "__main__":
    main()