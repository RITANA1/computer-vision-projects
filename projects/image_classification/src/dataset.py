import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

class CatDogDataset(Dataset):

    def __init__(self, train=True):

        self.data = datasets.CIFAR10(
            root="projects/image_classification/data/raw",
            train=train,
            download=False,
            transform=transform
        )

        self.indices = [
            i for i, label in enumerate(self.data.targets)
            if label in [3, 5]
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):

        image, label = self.data[self.indices[index]]

        # Cat = 0, Dog = 1
        label = 0 if label == 3 else 1

        return image, torch.tensor(label)