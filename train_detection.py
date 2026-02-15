import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from datasets.voc_dataset import VOCHumanAnimal, get_transform
import os

# ==========================
# SETTINGS
# ==========================

CHECKPOINT_PATH = "detector_finetuned.pth"
ROOT = r"D:\HumanAnimalDetection\voc\VOC2012_train_val\VOC2012_train_val"
NUM_CLASSES = 3
BATCH_SIZE = 2
NUM_EPOCHS = 20
LR = 1e-5   # 🔥 Lower LR when resuming

def collate_fn(batch):
    return tuple(zip(*batch))

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ==========================
    # DATA
    # ==========================

    train_dataset = VOCHumanAnimal(
        root=ROOT,
        image_set="train",
        transforms=get_transform(train=True)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    # ==========================
    # MODEL ARCHITECTURE
    # ==========================

    model = fasterrcnn_resnet50_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, NUM_CLASSES
    )

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    start_epoch = 0

    # ==========================
    # LOAD EXISTING MODEL
    # ==========================

    if os.path.exists(CHECKPOINT_PATH):

        print("Loading existing model...")

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

        # If only model state dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1

            print(f"Resuming from epoch {start_epoch}")

        else:
            # Simple state dict
            model.load_state_dict(checkpoint)
            print("Model weights loaded (no optimizer state).")

    else:
        print("No checkpoint found. Training from scratch.")

    scaler = torch.cuda.amp.GradScaler()

    # ==========================
    # TRAIN LOOP
    # ==========================

    for epoch in range(start_epoch, NUM_EPOCHS):

        model.train()
        total_loss = 0

        for batch_idx, (images, targets) in enumerate(train_loader):

            images = [img.to(device) for img in images]
            targets = [
                {k: v.to(device) for k, v in t.items()}
                for t in targets
            ]

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += losses.item()

            if batch_idx % 20 == 0:
                print(
                    f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {losses.item():.4f}"
                )

        print(f"Epoch {epoch+1} Total Loss: {total_loss:.4f}")

        # Save checkpoint properly now
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, "detector_resume_checkpoint.pth")

        print("Checkpoint saved.")

    print("Training complete.")

if __name__ == "__main__":
    main()
