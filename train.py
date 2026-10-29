"""
Training Pipeline for ResNet-50 Pulmonary Classifier.
Supports two-stage transfer learning: Feature Extraction & Fine-Tuning.
Can be run via CLI or imported into Streamlit for live in-browser training.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
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


def run_training(
    data_dir="data",
    epochs_head=3,
    epochs_finetune=5,
    batch_size=16,
    lr_head=1e-3,
    lr_finetune=1e-4,
    save_path="best_resnet50_pulmonary.pth",
    epoch_callback=None
):
    """
    Executes 2-stage transfer learning with optional epoch_callback(info_dict)
    for real-time Streamlit visualization.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    # Data transforms and loaders
    transforms_dict = get_data_transforms(img_size=224, apply_clahe=True)

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        msg = f"Error: '{train_dir}' or '{val_dir}' not found."
        print(f"[!] {msg}")
        return False, msg, None

    train_dataset = PulmonaryRadiographyDataset(train_dir, transform=transforms_dict["train"])
    val_dataset = PulmonaryRadiographyDataset(val_dir, transform=transforms_dict["val"])

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        msg = f"Error: Training or validation folder is empty."
        print(f"[!] {msg}")
        return False, msg, None

    print(f"[*] Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
    print(f"[*] Classes: {TARGET_CLASSES}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize Model
    # Always prefer loading from local checkpoint (avoids internet download).
    # Fall back to pretrained=False (random init) if no checkpoint exists.
    if os.path.exists(save_path):
        print(f"[*] Loading from local checkpoint '{save_path}' (no download needed)")
        model = ResNet50PulmonaryClassifier(num_classes=len(TARGET_CLASSES), pretrained=False)
        try:
            model.load_state_dict(torch.load(save_path, map_location=device), strict=False)
            print(f"[*] Checkpoint loaded successfully.")
        except Exception as e:
            print(f"[!] Could not load checkpoint ({e}). Initializing from scratch.")
    else:
        print("[*] No local checkpoint found. Initializing with random weights (pretrained=False).")
        model = ResNet50PulmonaryClassifier(num_classes=len(TARGET_CLASSES), pretrained=False)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    total_epochs = epochs_head + epochs_finetune
    current_global_epoch = 0

    # ==========================================
    # STAGE 1: Feature Extraction (Freeze Backbone)
    # ==========================================
    print("\n--- STAGE 1: Training Custom Classification Head ---")
    model.freeze_feature_extractor()
    optimizer_head = torch.optim.AdamW(model.resnet.fc.parameters(), lr=lr_head, weight_decay=1e-4)

    for epoch in range(1, epochs_head + 1):
        current_global_epoch += 1
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_head, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{current_global_epoch}/{total_epochs}] (Stage 1: {epoch}/{epochs_head}) | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  [+] Saved new best model checkpoint (Val Acc: {best_val_acc:.2f}%)")

        if epoch_callback:
            epoch_callback({
                "stage": 1,
                "stage_name": "Stage 1: Classification Head Warm-up",
                "epoch": epoch,
                "stage_epochs": epochs_head,
                "global_epoch": current_global_epoch,
                "total_epochs": total_epochs,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_acc": best_val_acc,
                "history": history
            })

    # ==========================================
    # STAGE 2: Fine-Tuning Top Residual Blocks
    # ==========================================
    print("\n--- STAGE 2: Fine-Tuning Layer 4 & Classifier Head ---")
    model.unfreeze_top_layers()
    optimizer_finetune = torch.optim.AdamW([
        {"params": model.resnet.layer4.parameters(), "lr": lr_finetune * 0.1},
        {"params": model.resnet.fc.parameters(), "lr": lr_finetune}
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_finetune, T_max=max(1, epochs_finetune))

    for epoch in range(1, epochs_finetune + 1):
        current_global_epoch += 1
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_finetune, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{current_global_epoch}/{total_epochs}] (Stage 2: {epoch}/{epochs_finetune}) | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  [+] Saved new best model checkpoint (Val Acc: {best_val_acc:.2f}%)")

        if epoch_callback:
            epoch_callback({
                "stage": 2,
                "stage_name": "Stage 2: ResNet-50 Layer 4 Fine-Tuning",
                "epoch": epoch,
                "stage_epochs": epochs_finetune,
                "global_epoch": current_global_epoch,
                "total_epochs": total_epochs,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_acc": best_val_acc,
                "history": history
            })

    # Save training history
    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=4)
    print(f"\n[+] Training complete. Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"[+] Checkpoint saved at '{save_path}'")
    return True, "Training finished successfully", history


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

    run_training(
        data_dir=args.data_dir,
        epochs_head=args.epochs_head,
        epochs_finetune=args.epochs_finetune,
        batch_size=args.batch_size,
        lr_head=args.lr_head,
        lr_finetune=args.lr_finetune,
        save_path=args.save_path
    )


if __name__ == "__main__":
    main()
