
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "emotion_cnn.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Same CNN architecture used during training
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


# Load trained model
checkpoint = torch.load(
    MODEL_PATH, map_location=device, weights_only=True
)

classes = checkpoint["class_names"]

model = EmotionCNN(len(classes)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# Prepare image
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# Change this path to your test image
image_path = PROJECT_DIR / "dataset" / "test" / "happy"

image_files = list(image_path.glob("*.jpg"))

if not image_files:
    raise FileNotFoundError("No test images found!")

image = Image.open(image_files[0])
image_tensor = transform(image).unsqueeze(0).to(device)


# Prediction
with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)
    confidence, predicted = probabilities.max(dim=1)

print("Image:", image_files[0].name)
print("Predicted expression:", classes[predicted.item()])
print(f"Model confidence: {confidence.item() * 100:.2f}%")
