import torch
from torch.utils.data import DataLoader

from dataset import CatDogDataset
from model import CNN

def evaluate():

    device = torch.device("cpu")

    test_data = CatDogDataset(train=False)

    loader = DataLoader(
        test_data,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    model = CNN().to(device)

    model.load_state_dict(
        torch.load(
            "projects/image_classification/models/cat_dog_cnn.pth",
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate()