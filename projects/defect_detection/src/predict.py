
from ultralytics import YOLO
from pathlib import Path

# 1. Project paths
ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "runs" / "defect_detection" / "weights" / "best.pt"

# 2. Check model
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\nRun train.py first!"
    )

# 3. Load trained YOLO model
model = YOLO(str(MODEL_PATH))

# 4. Choose test images
IMAGE_DIR = ROOT / "data" / "test" / "images"

if not IMAGE_DIR.exists() or not any(IMAGE_DIR.iterdir()):
    raise FileNotFoundError("No test images found!")

# 5. Run prediction
results = model.predict(
    source=str(IMAGE_DIR),
    conf=0.25,
    imgsz=320,
    save=True,
    project=str(ROOT / "runs"),
    name="predictions",
    exist_ok=True
)

# 6. Display results
for result in results:
    print(f"\nImage: {Path(result.path).name}")

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        print(
            f"Defect: {class_name} | "
            f"Confidence: {confidence:.2%}"
        )

print("\nPrediction completed!")
