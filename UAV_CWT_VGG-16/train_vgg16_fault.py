"""Train a VGG-16 fault diagnosis model on CWT images (Normal vs Fault)."""

# === Configuration ===
DATA_DIR = "E:/DL_Learn/FD_BSAEON_DATA/UAV_CWT_VGG-16/Image/Train"  # Folder with Train/Normal and Train/Fault
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 10
VAL_SPLIT = 0.2
SEED = 42
BEST_MODEL_PATH = "best_vgg16_fault_diagnosis.pth"

import random
from pathlib import Path

import matplotlib

# Use a non-interactive backend to avoid GUI issues in headless/remote runs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloaders(data_dir: Path):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, full_dataset.classes


def build_model(num_classes: int) -> models.VGG:
    model = models.vgg16(pretrained=True)

    # Freeze all feature extractor parameters
    for param in model.features.parameters():
        param.requires_grad = False

    # Unfreeze last two convolutional blocks (approximately features[24:] through features[30])
    for layer in list(model.features.children())[24:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Replace classifier for binary classification
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    val_loss = running_loss / len(loader.dataset)
    val_acc = correct / total if total > 0 else 0.0
    return val_loss, val_acc


def main():
    set_seed(SEED)
    data_dir = Path(DATA_DIR)
    if not data_dir.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {data_dir}")

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, classes = build_dataloaders(data_dir)
    print(f"Classes: {classes}")

    model = build_model(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    best_val_acc = 0.0
    acc_history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        acc_history.append(val_acc)

        print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save best weights
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  Saved new best model with Val Acc: {best_val_acc:.4f}")

    # Plot accuracy curve
    plt.figure()
    plt.plot(range(1, EPOCHS + 1), acc_history, marker="o")
    plt.title("Validation Accuracy vs. Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_curve.png")
    print("Training complete. Accuracy curve saved to accuracy_curve.png")


if __name__ == "__main__":
    main()
