"""
Comprehensive Performance Evaluation for ResNet-50 Pulmonary Classification.
Calculates Accuracy, Precision, Recall, F1-Score, ROC-AUC, and generates visual plots.
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from dataset import PulmonaryRadiographyDataset, get_data_transforms, TARGET_CLASSES
from model import ResNet50PulmonaryClassifier


def run_evaluation(model_path="best_resnet50_pulmonary.pth", data_dir="data/test", output_dir="evaluation_results"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running evaluation on device: {device}")

    # Load test dataset
    transforms_dict = get_data_transforms(img_size=224, apply_clahe=True)
    test_dataset = PulmonaryRadiographyDataset(data_dir, transform=transforms_dict["test"])
    if len(test_dataset) == 0:
        print(f"[!] Test dataset at '{data_dir}' is empty or not found.")
        return

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Load model
    model = ResNet50PulmonaryClassifier(num_classes=len(TARGET_CLASSES), pretrained=False)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[*] Loaded weights from '{model_path}'")
    else:
        print(f"[!] Warning: Model file '{model_path}' not found. Using initialized weights for dry run.")

    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.append(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_probs = np.vstack(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 1. Classification Report (Precision, Recall, F1-Score)
    report = classification_report(all_labels, all_preds, target_names=TARGET_CLASSES, output_dict=True, zero_division=0)
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=TARGET_CLASSES, zero_division=0))

    # Save metrics JSON
    with open(os.path.join(output_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    # 2. Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=TARGET_CLASSES, yticklabels=TARGET_CLASSES)
    plt.title("ResNet-50 Pulmonary Diagnostic Confusion Matrix", fontsize=14, pad=12)
    plt.ylabel("True Diagnosis", fontsize=12)
    plt.xlabel("Predicted Diagnosis", fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[+] Confusion Matrix saved to '{cm_path}'")

    # 3. ROC Curves & AUC Calculation (One-vs-Rest)
    y_test_bin = label_binarize(all_labels, classes=list(range(len(TARGET_CLASSES))))
    plt.figure(figsize=(9, 7))

    for i, cls_name in enumerate(TARGET_CLASSES):
        if y_test_bin.shape[1] > 1:
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{cls_name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title("Multi-Class ROC-AUC Curves (ResNet-50)", fontsize=14, pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "roc_auc_curves.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[+] ROC-AUC Curves saved to '{roc_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="best_resnet50_pulmonary.pth")
    parser.add_argument("--data_dir", type=str, default="data/test")
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    args = parser.parse_args()

    run_evaluation(args.model_path, args.data_dir, args.output_dir)
