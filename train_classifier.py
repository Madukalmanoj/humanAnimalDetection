import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from collections import Counter
import os

# ======================
# SETTINGS
# ======================

DATA_DIR = r"D:\HumanAnimalDetection\classification"
BASE_MODEL_PATH = "classifier_finetuned.pth"  # your existing model
CHECKPOINT_PATH = "classifier_resume_checkpoint.pth"

BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 1e-5   # Lower LR for fine-tuning

# ======================
# MAIN FUNCTION (WINDOWS SAFE)
# ======================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ======================
    # Transforms
    # ======================

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)

    # ======================
    # Train / Val Split
    # ======================

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # ======================
    # Compute Class Weights (Train Only)
    # ======================

    train_targets = [dataset.samples[i][1] for i in train_dataset.indices]
    class_counts = Counter(train_targets)

    print("Train class counts:", class_counts)

    class_weights = 1.0 / torch.tensor(
        [class_counts[i] for i in range(len(class_counts))],
        dtype=torch.float32
    )

    class_weights = class_weights.to(device)

    # ======================
    # Weighted Sampler
    # ======================

    sample_weights = [class_weights[label].item() for label in train_targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # ======================
    # DataLoaders
    # ======================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # ======================
    # Model
    # ======================

    model = efficientnet_b0(weights="DEFAULT")
    model.classifier[1] = nn.Linear(1280, 2)
    model.to(device)

    # ======================
    # Load Existing Base Model
    # ======================

    if os.path.exists(BASE_MODEL_PATH):
        print("Loading base classifier model...")
        model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=device))
        print("Base model loaded. Fine-tuning started.")
    else:
        print("No base model found. Training from scratch.")

    # ======================
    # Loss + Optimizer
    # ======================

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    scaler = torch.amp.GradScaler("cuda")

    # ======================
    # Training Loop
    # ======================

    for epoch in range(NUM_EPOCHS):

        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # ======================
        # Validation
        # ======================

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
            f"Loss: {total_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Val Acc: {val_acc:.4f}"
        )

        # Save checkpoint every epoch
        torch.save(model.state_dict(), CHECKPOINT_PATH)

    print("Training complete.")

    torch.save(model.state_dict(), "classifier_finetuned_v2.pth")
    print("Final model saved.")


# ======================
# ENTRY POINT (CRITICAL FOR WINDOWS)
# ======================

if __name__ == "__main__":
    main()
