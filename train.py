"""
Training Pipeline for ResNet-50 Pulmonary Classifier.
Supports two-stage transfer learning: Feature Extraction & Fine-Tuning.
"""

import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import PulmonaryRadiographyDataset, get_data_transforms, TARGET_CLASSES
from model import ResNet50PulmonaryClassifier


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0
    epoch_acc = (correct / total) * 100 if total > 0 else 0
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0
    epoch_acc = (correct / total) * 100 if total > 0 else 0
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-50 on Radiographic Scans")
    parser.add_argument("--data_dir", type=str, default="data", help="Root directory containing train/ and val/ folders")
    parser.add_argument("--epochs_head", type=int, default=5, help="Epochs for training classification head only")
    parser.add_argument("--epochs_finetune", type=int, default=15, help="Epochs for fine-tuning top residual blocks")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr_head", type=float, default=1e-3, help="Learning rate for head training")
    parser.add_argument("--lr_finetune", type=float, default=1e-4, help="Learning rate for fine-tuning")
    parser.add_argument("--save_path", type=str, default="best_resnet50_pulmonary.pth", help="Model checkpoint output path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    # Data transforms and loaders
    transforms_dict = get_data_transforms(img_size=224, apply_clahe=True)

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"[!] Warning: '{train_dir}' or '{val_dir}' not found.")
        print("[*] Please structure your dataset or generate sample data via dataset.py.")
        return

    train_dataset = PulmonaryRadiographyDataset(train_dir, transform=transforms_dict["train"])
    val_dataset = PulmonaryRadiographyDataset(val_dir, transform=transforms_dict["val"])

    print(f"[*] Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
    print(f"[*] Classes: {TARGET_CLASSES}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Initialize Model
    model = ResNet50PulmonaryClassifier(num_classes=len(TARGET_CLASSES), pretrained=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # ==========================================
    # STAGE 1: Feature Extraction (Freeze Backbone)
    # ==========================================
    print("\n--- STAGE 1: Training Custom Classification Head ---")
    model.freeze_feature_extractor()
    optimizer_head = torch.optim.AdamW(model.resnet.fc.parameters(), lr=args.lr_head, weight_decay=1e-4)

    for epoch in range(1, args.epochs_head + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_head, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch}/{args.epochs_head}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"  [+] Saved new best model checkpoint (Val Acc: {best_val_acc:.2f}%)")

    # ==========================================
    # STAGE 2: Fine-Tuning Top Residual Blocks
    # ==========================================
    print("\n--- STAGE 2: Fine-Tuning Layer 4 & Classifier Head ---")
    model.unfreeze_top_layers()
    optimizer_finetune = torch.optim.AdamW([
        {"params": model.resnet.layer4.parameters(), "lr": args.lr_finetune * 0.1},
        {"params": model.resnet.fc.parameters(), "lr": args.lr_finetune}
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_finetune, T_max=args.epochs_finetune)

    total_stage2_epochs = args.epochs_finetune
    for epoch in range(1, total_stage2_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_finetune, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch}/{total_stage2_epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"  [+] Saved new best model checkpoint (Val Acc: {best_val_acc:.2f}%)")

    # Save training history
    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=4)
    print(f"\n[✓] Training complete. Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"[✓] Checkpoint saved at '{args.save_path}'")


if __name__ == "__main__":
    main()
