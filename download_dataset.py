"""
Dataset Download & Setup Script for PulmoScan AI
Downloads the COVID-19 Radiography Database from Kaggle and structures it
for training the ResNet-50 pulmonary classifier.

Dataset: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
Classes: Normal, COVID-19, Lung_Opacity (Pneumonia), Viral Pneumonia / Tuberculosis

SETUP:
  1. Create a Kaggle account at https://www.kaggle.com
  2. Go to Account -> API -> Create New Token -> download kaggle.json
  3. Place kaggle.json in: C:\Users\<YourName>\.kaggle\kaggle.json
  4. Run: python download_dataset.py
"""

import os, sys, shutil, random
from pathlib import Path

TARGET_CLASSES = ["Normal", "COVID-19", "Pneumonia", "Tuberculosis"]
KAGGLE_DATASET = "tawsifurrahman/covid19-radiography-database"
DATA_DIR = "data"
SAMPLES_PER_CLASS_TRAIN = 300
SAMPLES_PER_CLASS_VAL   = 75

def check_kaggle():
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("\nkaggle.json not found!")
        print("Steps: 1) go to kaggle.com -> Account -> API -> Create New Token")
        print(f"       2) copy kaggle.json to {kaggle_json}")
        return False
    return True

def download_from_kaggle():
    print(f"\nDownloading dataset: {KAGGLE_DATASET}")
    import subprocess
    r = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download",
                        "-d", KAGGLE_DATASET, "--unzip", "-p", "kaggle_raw"])
    if r.returncode != 0:
        print("Download failed. Install kaggle: pip install kaggle")
        sys.exit(1)
    print("Download complete.")

def structure_dataset(raw_dir="kaggle_raw"):
    folder_map = {
        "Normal": "Normal", "COVID": "COVID-19", "COVID-19": "COVID-19",
        "Lung_Opacity": "Pneumonia", "Viral Pneumonia": "Tuberculosis",
        "Tuberculosis": "Tuberculosis",
    }
    for split in ["train", "val"]:
        for cls in TARGET_CLASSES:
            os.makedirs(os.path.join(DATA_DIR, split, cls), exist_ok=True)

    raw_path = Path(raw_dir)
    for folder in raw_path.rglob("*"):
        if not folder.is_dir(): continue
        mapped = folder_map.get(folder.name)
        if not mapped: continue
        images = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
        random.shuffle(images)
        selected = images[:SAMPLES_PER_CLASS_TRAIN + SAMPLES_PER_CLASS_VAL]
        for i, img in enumerate(selected):
            split = "train" if i < SAMPLES_PER_CLASS_TRAIN else "val"
            dst = os.path.join(DATA_DIR, split, mapped, img.name)
            if not os.path.exists(dst): shutil.copy2(str(img), dst)

    print("\nDataset structured:")
    for cls in TARGET_CLASSES:
        t = len(list(Path(DATA_DIR, "train", cls).glob("*")))
        v = len(list(Path(DATA_DIR, "val",   cls).glob("*")))
        print(f"  {cls:20s}  train={t}  val={v}")

def verify_dataset():
    ok = True
    for split in ["train", "val"]:
        for cls in TARGET_CLASSES:
            p = Path(DATA_DIR) / split / cls
            if not p.exists() or len(list(p.glob("*"))) == 0:
                ok = False
    return ok

if __name__ == "__main__":
    print("=" * 55)
    print("  PulmoScan AI - Dataset Setup")
    print("=" * 55)
    if verify_dataset():
        print("Dataset already ready!")
        print("Run: python train.py --data_dir data")
        sys.exit(0)
    if not check_kaggle(): sys.exit(1)
    try: import kaggle
    except ImportError: os.system(f"{sys.executable} -m pip install kaggle")
    download_from_kaggle()
    structure_dataset()
    if verify_dataset():
        print("\nAll done! Now run:")
        print("  python train.py --data_dir data --epochs_head 5 --epochs_finetune 15")
