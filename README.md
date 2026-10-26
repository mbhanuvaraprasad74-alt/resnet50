# ResNet50-Powered Approach for Classifying Pulmonary Conditions from Radiographic Scans

**Student:** M. Akhil Kumar (Roll No: 25G31D5819)  
**Institution:** SJCET, Department of Computer Science & Engineering  
**Domain:** Deep Learning / Computer Vision / Medical Image Processing  

---

## 📌 Project Overview
This repository contains the complete end-to-end implementation for the Major Project proposal:
> **"ResNet50-Powered Approach for Classifying Pulmonary Conditions from Radiographic Scans"**

The framework classifies **Chest X-Ray** and **Computed Tomography (CT)** scans into 4 clinical categories:
1. **Normal** (Healthy Lung Tissue)
2. **COVID-19** (SARS-CoV-2 Ground-Glass Opacities)
3. **Pneumonia** (Bacterial & Viral Infiltrates / Consolidation)
4. **Tuberculosis** (Mycobacterial Cavitary Lesions & Apical Fibrosis)

Additionally, this framework includes **Grad-CAM (Gradient-weighted Class Activation Mapping)** for Explainable AI (XAI) and clinical decision support, alongside a full web interface built with **Streamlit**.

---

## 🗂 Project Architecture & Directory Layout

```
resnet/
├── app.py                      # Month-9: Streamlit Web Diagnostic GUI & Grad-CAM visualizer
├── dataset.py                  # Month-2 & 3: Dataset loader, CLAHE preprocessor, augmentations
├── model.py                    # Month-4: ResNet-50 transfer learning & Grad-CAM hook module
├── train.py                    # Month-5 & 6: 2-stage training (head warm-up + top block fine-tuning)
├── evaluate.py                 # Month-7: Accuracy, Precision, Recall, F1, ROC-AUC & Confusion Matrix
├── benchmark_comparison.py     # Month-8: Baseline ML (SVM, Random Forest) vs ResNet-50
├── requirements.txt            # Python dependencies
├── sample_data/                # Auto-generated mock dataset for rapid testing
├── evaluation_results/         # Generated plots (Confusion Matrix, ROC curves, Benchmarks)
└── README.md                   # Full documentation & viva guide
```

---

## 🚀 Quick Setup & Installation

### 1. Prerequisites
- **OS**: Windows 10 / 11
- **Python**: 3.9, 3.10, or 3.11
- **Hardware**: Core i5/i7, 16GB RAM (NVIDIA GPU recommended, but CPU is supported)

### 2. Environment Setup
Open PowerShell or Command Prompt in this folder:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 🔬 Dataset Structure

Organize your collected datasets (from Kaggle, NIH, RSNA, COVIDx) as follows:

```
data/
├── train/
│   ├── Normal/
│   ├── COVID-19/
│   ├── Pneumonia/
│   └── Tuberculosis/
├── val/
│   ├── Normal/
│   ├── COVID-19/
│   ├── Pneumonia/
│   └── Tuberculosis/
└── test/
    ├── Normal/
    ├── COVID-19/
    ├── Pneumonia/
    └── Tuberculosis/
```

### Quick Synthetic Data Generation (For Immediate Testing):
If you have not yet downloaded the multi-gigabyte datasets, run:
```powershell
python dataset.py
```
This generates mock sample scans in `sample_data/` so you can verify the entire pipeline in seconds.

---

## 🏋️‍♂️ Execution Guide

### Step 1: Train the ResNet-50 Model (Months 5 & 6)
```powershell
python train.py --data_dir data --epochs_head 5 --epochs_finetune 15 --batch_size 16
```
This executes:
- **Phase 1**: Freezes ResNet-50 feature extractor and trains custom classifier head (5 epochs).
- **Phase 2**: Unfreezes `layer4` and fine-tunes with Cosine Annealing learning rate (15 epochs).
- Saves best model weights to `best_resnet50_pulmonary.pth`.

### Step 2: Comprehensive Evaluation (Month 7)
```powershell
python evaluate.py --model_path best_resnet50_pulmonary.pth --data_dir data/test
```
Produces:
- `evaluation_results/confusion_matrix.png`
- `evaluation_results/roc_auc_curves.png`
- `evaluation_results/classification_report.json`

### Step 3: Traditional ML Benchmarking (Month 8)
```powershell
python benchmark_comparison.py
```
Extracts GLCM texture and HOG statistical descriptors, trains **SVM** and **Random Forest**, and plots comparison graphs against **ResNet-50**.

### Step 4: Launch Web Diagnostic System (Month 9)
```powershell
streamlit run app.py
```
Opens an interactive browser interface at `http://localhost:8501`:
- Upload radiographic scan (X-Ray or CT).
- View instant condition classification & probability percentages.
- View Grad-CAM attention heatmaps highlighting suspicious pulmonary regions.

---

## 📊 Alignment with the 12-Month Work Plan

| Month | Proposal Milestone | Implementation in this Codebase |
|---|---|---|
| **Month 1** | Literature Review | Documented in Project Report Artifact |
| **Month 2** | Dataset Collection | Kaggle / RSNA dataset mapping in documentation |
| **Month 3** | Data Preprocessing | `dataset.py` (CLAHE, Normalization, Resizing) |
| **Month 4** | Model Familiarization | `model.py` (ResNet-50 residual blocks & skip connections) |
| **Month 5** | Phase I: X-Ray Training | `train.py` (2-stage transfer learning) |
| **Month 6** | Phase II: CT Training | `train.py` with modality flag |
| **Month 7** | Performance Evaluation | `evaluate.py` (Accuracy, Precision, Recall, F1, ROC-AUC) |
| **Month 8** | Comparison with Traditional ML | `benchmark_comparison.py` (SVM, Random Forest vs ResNet-50) |
| **Month 9** | System Integration | `app.py` (Interactive Streamlit prototype with Grad-CAM) |
| **Month 10** | Validation & Robustness | Noise sensitivity & cross-dataset verification |
| **Month 11** | Paper & Report Writing | Comprehensive structured research paper template |
| **Month 12** | Presentation & Viva | Presentation outline and defense Q&A guide |
