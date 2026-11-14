import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CatDogDataset
from model import CNN

def train():

    device = torch.device("cpu")

    train_data = CatDogDataset(train=True)

    loader = DataLoader(
        train_data,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )

    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 5

    for epoch in range(epochs):

        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {total_loss/len(loader):.4f} "
            f"Accuracy: {accuracy:.2f}%"
        )

    torch.save(
        model.state_dict(),
        "projects/image_classification/models/cat_dog_cnn.pth"
    )

    print("\nModel saved successfully!")

if __name__ == "__main__":
    train()