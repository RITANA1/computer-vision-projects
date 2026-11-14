
from ultralytics import YOLO
from pathlib import Path

# 1. Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data.yaml"

# 2. Load a pretrained YOLO model
model = YOLO("yolo11n.pt")

# 3. Train the model
results = model.train(
    data=str(DATA_YAML),
    epochs=50,
    imgsz=320,
    batch=4,
    device="cpu",
    workers=0,
    patience=15,
    project=str(ROOT / "runs"),
    name="defect_detection",
    plots=True,
    exist_ok=True
)

print("\nTraining completed!")
print("Model saved inside:", ROOT / "runs" / "defect_detection" / "weights")
