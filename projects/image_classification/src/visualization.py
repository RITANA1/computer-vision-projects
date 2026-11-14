import cv2
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms

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

model = CNN()
model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
)
model.eval()

def classify(image_path):

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)

    confidence, predicted = probabilities.max(1)

    return CLASSES[predicted.item()], confidence.item() * 100

def visualize(image_path):

    label, confidence = classify(image_path)

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not load image!")
        return

    image = cv2.resize(image, (640, 480))

    text = f"{label}: {confidence:.2f}%"

    cv2.putText(
        image,
        text,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("CNN Image Classification", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    visualize(OUTPUT_DIR / "test_cat.jpg")
    visualize(OUTPUT_DIR / "test_dog.jpg")