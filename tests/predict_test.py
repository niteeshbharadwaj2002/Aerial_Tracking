from ultralytics import YOLO

model = YOLO('runs/detect/outputs/runs/visdrone_baseline-6/weights/best.pt')

# 1. Metrics on held-out test set
print("=== Test Set Metrics ===")
metrics = model.val(data='data/visdrone.yaml', split='test')
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)

# 2. Qualitative inference — save annotated images
print("\n=== Running Inference on Test Images ===")
model.predict(
    source='data/yolo/images/test',
    save=True,
    project='outputs',
    name='test_inference',
    conf=0.25
)
print("Done. Check outputs/test_inference/ for annotated images.")