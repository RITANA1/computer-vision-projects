
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "emotion_cnn.pth"
FACE_MODEL_PATH = PROJECT_DIR / "models" / "face_detection_yunet_2023mar_int8.onnx"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Emotion CNN
class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# Load emotion model
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

classes = checkpoint["class_names"]

model = EmotionCNN(len(classes)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# Load YuNet face detector
detector = cv2.FaceDetectorYN.create(
    str(FACE_MODEL_PATH),
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam!")

print("Webcam started! Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam frame.")
        break

    height, width = frame.shape[:2]
    detector.setInputSize((width, height))

    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4].astype(int)

            # Keep coordinates inside the image
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width, x + w)
            y2 = min(height, y + h)

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            # Preprocess face
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (48, 48))

            image = gray.astype(np.float32) / 255.0
            image = (image - 0.5) / 0.5

            tensor = torch.from_numpy(image)
            tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)

            # Predict expression
            with torch.no_grad():
                output = model(tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = probabilities.max(dim=1)

            label = classes[predicted.item()]
            score = confidence.item() * 100

            # Draw face box and prediction
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            text = f"{label}: {score:.1f}%"

            cv2.putText(
                frame,
                text,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    cv2.imshow("VisionLab - Facial Expression Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Webcam stopped.")
