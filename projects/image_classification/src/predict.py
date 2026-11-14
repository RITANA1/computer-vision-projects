import sys
import torch
from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms

from model import CNN

ROOT = Path("projects/image_classification")
MODEL_PATH = ROOT / "models/cat_dog_cnn.pth"
OUTPUT_DIR = ROOT / "outputs"

CLASSES = ["Cat", "Dog"]

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

def predict(image_path, model):

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted = probabilities.max(1)

    print(f"\nImage: {image_path}")
    print(f"Prediction: {CLASSES[predicted.item()]}")
    print(f"Confidence: {confidence.item()*100:.2f}%")

def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    model = CNN()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()

    if len(sys.argv) > 1 and sys.argv[1] != "--demo":
        predict(sys.argv[1], model)
        return

    test_data = datasets.CIFAR10(
        root=str(ROOT / "data/raw"),
        train=False,
        download=False
    )

    found = set()

    for image, label in test_data:

        if label not in [3, 5]:
            continue

        name = "cat" if label == 3 else "dog"

        if name in found:
            continue

        path = OUTPUT_DIR / f"test_{name}.jpg"
        image.save(path)

        predict(path, model)
        found.add(name)

        if len(found) == 2:
            break

if __name__ == "__main__":
    main()