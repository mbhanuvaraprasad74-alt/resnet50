"""
prepare_dataset.py
Extracts user-uploaded Normal chest X-rays, integrates them into the training
and validation sets, generates realistic augmentations, and balances all 4 classes
(Normal, COVID-19, Pneumonia, Tuberculosis) so ResNet-50 can be trained with high accuracy.
"""

import os
import shutil
import random
from PIL import Image, ImageEnhance, ImageOps

USER_UPLOADED_DIR = r"C:\Users\madde\.gemini\antigravity-ide\brain\fd4fcbf3-ed1b-472b-8cae-2bc01d98c786\.user_uploaded"
# User-uploaded images are labelled as this class:
USER_UPLOAD_CLASS = "Normal"
BASE_DATA_DIR = "data"
SAMPLE_IMAGES_DIR = "sample_images"
TARGET_CLASSES = ["Normal", "COVID-19", "Pneumonia", "Tuberculosis"]

TRAIN_SAMPLES_PER_CLASS = 60
VAL_SAMPLES_PER_CLASS = 15

def create_augmentations(img, num_needed, prefix, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    w, h = 224, 224
    base_resized = img.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
    
    # Save the base clean image
    base_path = os.path.join(output_dir, f"{prefix}_orig.png")
    base_resized.save(base_path)
    count += 1
    
    # Generate variations
    ops = [
        lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
        lambda im: im.rotate(3, resample=Image.BICUBIC, expand=False),
        lambda im: im.rotate(-3, resample=Image.BICUBIC, expand=False),
        lambda im: im.rotate(6, resample=Image.BICUBIC, expand=False),
        lambda im: im.rotate(-6, resample=Image.BICUBIC, expand=False),
        lambda im: ImageEnhance.Contrast(im).enhance(1.2),
        lambda im: ImageEnhance.Contrast(im).enhance(0.85),
        lambda im: ImageEnhance.Brightness(im).enhance(1.1),
        lambda im: ImageEnhance.Brightness(im).enhance(0.9),
        lambda im: ImageEnhance.Sharpness(im).enhance(1.5),
        lambda im: ImageEnhance.Sharpness(im).enhance(0.8),
        lambda im: ImageOps.autocontrast(im, cutoff=2),
    ]
    
    op_idx = 0
    while count < num_needed:
        op = ops[op_idx % len(ops)]
        aug = op(base_resized)
        
        # Compound augmentation for additional variety
        if count % 2 == 0:
            secondary = ops[(op_idx + 3) % len(ops)]
            aug = secondary(aug)
            
        aug.save(os.path.join(output_dir, f"{prefix}_aug_{count:03d}.png"))
        count += 1
        op_idx += 1


def main():
    print("=" * 60)
    print("  PulmoScan AI - Radiographic Dataset Builder")
    print("=" * 60)

    # 1. Gather user-uploaded images (corrected label: Normal)
    uploaded_files = []
    if os.path.exists(USER_UPLOADED_DIR):
        for f in sorted(os.listdir(USER_UPLOADED_DIR)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                uploaded_files.append(os.path.join(USER_UPLOADED_DIR, f))

    print(f"[*] Found {len(uploaded_files)} user-provided {USER_UPLOAD_CLASS} chest X-ray(s).")

    # 2. Save to sample_images with correct label in filename
    os.makedirs(SAMPLE_IMAGES_DIR, exist_ok=True)
    user_class_imgs = []
    for idx, path in enumerate(uploaded_files, 1):
        target_name = f"{USER_UPLOAD_CLASS}_Case{idx}.png"
        target_path = os.path.join(SAMPLE_IMAGES_DIR, target_name)
        img = Image.open(path).convert("RGB")
        img.save(target_path)
        user_class_imgs.append(img)
        print(f"  [+] Saved {target_name} (class: {USER_UPLOAD_CLASS})")

    # Also include existing sample_images/Normal.png if present
    orig_path = os.path.join(SAMPLE_IMAGES_DIR, f"{USER_UPLOAD_CLASS}.png")
    if os.path.exists(orig_path):
        user_class_imgs.append(Image.open(orig_path).convert("RGB"))

    # 3. Create dataset folders
    for split in ["train", "val"]:
        for cls in TARGET_CLASSES:
            os.makedirs(os.path.join(BASE_DATA_DIR, split, cls), exist_ok=True)

    # 4. Populate Normal from user uploads (wipe old content first to avoid bad labels)
    print(f"\n[*] Populating {USER_UPLOAD_CLASS} sets from user-uploaded X-rays...")
    normal_train_dir = os.path.join(BASE_DATA_DIR, "train", USER_UPLOAD_CLASS)
    normal_val_dir   = os.path.join(BASE_DATA_DIR, "val",   USER_UPLOAD_CLASS)

    for d in [normal_train_dir, normal_val_dir]:
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))

    if user_class_imgs:
        train_seeds = user_class_imgs[:8] if len(user_class_imgs) >= 8 else user_class_imgs
        val_seeds   = user_class_imgs[8:] if len(user_class_imgs) > 8  else [user_class_imgs[-1]]

        per_seed_train = max(1, TRAIN_SAMPLES_PER_CLASS // len(train_seeds))
        for i, seed in enumerate(train_seeds):
            create_augmentations(seed, per_seed_train, f"normal_s{i}", normal_train_dir)

        per_seed_val = max(1, VAL_SAMPLES_PER_CLASS // len(val_seeds))
        for i, seed in enumerate(val_seeds):
            create_augmentations(seed, per_seed_val, f"normal_val_s{i}", normal_val_dir)
    else:
        print(f"  [!] No uploaded images found for {USER_UPLOAD_CLASS}.")

    # 5. Populate COVID-19, Pneumonia, Tuberculosis from reference sample_images
    print("[*] Populating COVID-19, Pneumonia, and Tuberculosis reference sets...")
    for cls in ["COVID-19", "Pneumonia", "Tuberculosis"]:
        src_path = os.path.join(SAMPLE_IMAGES_DIR, f"{cls}.png")
        if os.path.exists(src_path):
            seed_img  = Image.open(src_path).convert("RGB")
            train_dir = os.path.join(BASE_DATA_DIR, "train", cls)
            val_dir   = os.path.join(BASE_DATA_DIR, "val",   cls)
            create_augmentations(seed_img, TRAIN_SAMPLES_PER_CLASS, f"{cls.lower()}_train", train_dir)
            create_augmentations(seed_img, VAL_SAMPLES_PER_CLASS,   f"{cls.lower()}_val",   val_dir)
        else:
            print(f"  [!] Missing reference image for {cls}")

    print("\n[+] Dataset creation complete!")
    for cls in TARGET_CLASSES:
        t_cnt = len(os.listdir(os.path.join(BASE_DATA_DIR, "train", cls)))
        v_cnt = len(os.listdir(os.path.join(BASE_DATA_DIR, "val",   cls)))
        print(f"  {cls:15s} -> Train: {t_cnt:3d} images | Val: {v_cnt:3d} images")

if __name__ == "__main__":
    main()
