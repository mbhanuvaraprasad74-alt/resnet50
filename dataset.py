"""
Dataset Loading, Preprocessing, and Augmentation Pipeline
for Pulmonary Condition Classification from Radiographic Scans (X-Ray & CT).
"""

import os
try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Default 4 target classes as defined in the project abstract
TARGET_CLASSES = ["Normal", "COVID-19", "Pneumonia", "Tuberculosis"]

class MedicalCLAHE:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to enhance soft-tissue contrast in radiographic scans.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img_pil):
        if cv2 is not None:
            # Convert PIL Image to Grayscale / OpenCV array
            img_np = np.array(img_pil.convert("L"))
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            enhanced = clahe.apply(img_np)
            # Convert back to 3-channel RGB for ResNet50
            enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(enhanced_rgb)
        else:
            equalized = ImageOps.equalize(img_pil.convert("L"))
            return equalized.convert("RGB")


def get_data_transforms(img_size=224, apply_clahe=True):
    """
    Constructs PyTorch image transformations for training, validation, and testing.
    """
    train_transforms_list = []
    val_transforms_list = []

    if apply_clahe:
        train_transforms_list.append(MedicalCLAHE(clip_limit=2.0, tile_grid_size=(8, 8)))
        val_transforms_list.append(MedicalCLAHE(clip_limit=2.0, tile_grid_size=(8, 8)))

    # Training transforms with conservative medical data augmentation
    train_transforms_list.extend([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Validation/Testing transforms (deterministic)
    val_transforms_list.extend([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return {
        "train": transforms.Compose(train_transforms_list),
        "val": transforms.Compose(val_transforms_list),
        "test": transforms.Compose(val_transforms_list)
    }


class PulmonaryRadiographyDataset(Dataset):
    """
    Custom PyTorch Dataset for loading X-Ray / CT scan images.
    Expects directory structure:
        data_dir/
            Normal/
            COVID-19/
            Pneumonia/
            Tuberculosis/
    """
    def __init__(self, root_dir, classes=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = classes if classes is not None else TARGET_CLASSES
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        for cls_name in self.classes:
            class_folder = os.path.join(root_dir, cls_name)
            if not os.path.exists(class_folder):
                continue
            for fname in os.listdir(class_folder):
                ext = os.path.splitext(fname)[1].lower()
                if ext in valid_extensions:
                    full_path = os.path.join(class_folder, fname)
                    self.samples.append((full_path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, img_path


def create_mock_dataset(base_dir="dataset", num_samples_per_class=10):
    """
    Generates a synthetic mock dataset structure for immediate local testing and dry runs.
    """
    for split in ["train", "val", "test"]:
        for cls_name in TARGET_CLASSES:
            dir_path = os.path.join(base_dir, split, cls_name)
            os.makedirs(dir_path, exist_ok=True)
            for i in range(num_samples_per_class if split == "train" else max(3, num_samples_per_class // 3)):
                # Generate a dummy synthetic grayscale radiograph-like pattern
                img_array = np.random.randint(50, 200, (224, 224), dtype=np.uint8)
                # Draw mock lung ellipses
                cv2.ellipse(img_array, (80, 112), (35, 60), 0, 0, 360, 230, -1)
                cv2.ellipse(img_array, (144, 112), (35, 60), 0, 0, 360, 230, -1)
                cv2.imwrite(os.path.join(dir_path, f"sample_{i:03d}.png"), img_array)
    print(f"[+] Mock dataset successfully generated under '{base_dir}/'")


if __name__ == "__main__":
    create_mock_dataset("sample_data", num_samples_per_class=5)
