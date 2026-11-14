import sys

sys.path.append("projects/image_classification/src")

from train import train
from evaluate import evaluate

if __name__ == "__main__":

    print("Starting Image Classification Project...")

    print("\n1. Training CNN...")
    train()

    print("\n2. Evaluating CNN...")
    evaluate()

    print("\nProject completed!")