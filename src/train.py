import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Опять явно не вызываем автологгинг, чтобы не создавать вложенные запуски

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.fc1 = nn.Linear(26 * 26 * 16, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        return self.fc2(x)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False)
    for batch_idx, (data, target) in enumerate(loop):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        avg_loss = running_loss / (batch_idx + 1)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        loop.set_postfix(loss=avg_loss)


def test(model, device, test_loader, epoch):
    model.eval()
    correct = 0
    all_preds, all_targets = [], []
    loop = tqdm(test_loader, desc=f"Validation Epoch {epoch}", leave=False)
    with torch.no_grad():
        for data, target in loop:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    acc = correct / len(test_loader.dataset)
    mlflow.log_metric("val_accuracy", acc, step=epoch)
    return acc, all_preds, all_targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN on MNIST with MLflow logging")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")

    mlflow.set_experiment("cifar-experiment")

    mlflow.autolog()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        accuracy, preds, targets = test(model, device, test_loader, epoch)

    cm = confusion_matrix(targets, preds)
    fig, ax = plt.subplots()
    ax.matshow(cm)
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.pytorch.log_model(model, "model")
    print(f"Logged model with final accuracy: {accuracy}")
