
from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "runs" / "defect_detection" / "weights" / "best.pt"
DATA_YAML = ROOT / "data.yaml"

model = YOLO(str(MODEL_PATH))

metrics = model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=320,
    batch=4,
    device="cpu",
    workers=0,
    plots=True,
    project=str(ROOT / "runs"),
    name="evaluation",
    exist_ok=True
)

print("\n========== MODEL EVALUATION ==========")
print(f"Precision:  {metrics.box.mp:.4f}")
print(f"Recall:     {metrics.box.mr:.4f}")
print(f"mAP50:      {metrics.box.map50:.4f}")
print(f"mAP50-95:   {metrics.box.map:.4f}")

print("\nPer-class mAP50:")
for class_id, score in enumerate(metrics.box.ap50):
    print(f"{model.names[class_id]}: {score:.4f}")

print("\nEvaluation completed!")
print("Results saved in:", ROOT / "runs" / "evaluation")
