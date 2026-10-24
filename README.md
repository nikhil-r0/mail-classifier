# MailSort — NLP Email Classification System

[![Python 3.12+](https://img.shields.io/badge/python-3.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Course](https://img.shields.io/badge/Course-BIC703%20Machine%20Learning-orange.svg)](https://bit-bangalore.edu.in)
[![Tests](https://img.shields.io/badge/Tests-15%20Passing-brightgreen.svg)]()
[![Model Accuracy](https://img.shields.io/badge/Test%20Accuracy-98.08%25-success.svg)]()

**MailSort** is a standalone, offline Natural Language Processing email classification application developed for the **BIC703 Machine Learning** course assignment (VII Semester, Department of Computer Science & Engineering, Bangalore Institute of Technology).

The application classifies email subject and body into Gmail-inspired target categories:
- **Primary** — Personal, academic, and direct workplace correspondence.
- **Promotions** — Marketing offers, discounts, clearance sales, and newsletters.
- **Spam** — Phishing attacks, lottery scams, extortion, and fraudulent solicitations.
- **Other / Needs review** — Ambiguous messages, low-confidence predictions, or lossy mappings from baseline categories.

---

## 1. System Architecture & Model Modes

MailSort supports two distinct models to demonstrate transparent evaluation and compliance with academic constraints:

1. **Project-Owned Model (Default / Active):**
   - **Type:** Scikit-Learn TF-IDF (1-2 ngrams) + Regularized Linear Classifier with balanced class weights.
   - **Ownership:** Built and trained strictly from scratch by our project pipeline (`training/train.py`).
   - **Target Labels:** Native `Primary`, `Promotions`, and `Spam`.
   - **Performance:** **98.08% Test Accuracy**, **98.79% Macro Precision**, **98.66% Macro F1** on held-out test data; **98.33% Accuracy** across 898 companion test emails.
   - **Footprint:** Pure CPU, sub-2ms inference latency, zero heavy framework dependencies.

2. **External Reference Checkpoint (Baseline):**
   - **Origin:** Pre-trained Hugging Face checkpoint ([`jason23322/high-accuracy-email-classifier`](https://huggingface.co/jason23322/high-accuracy-email-classifier)).
   - **Architecture:** Multi-scale CNN + Bidirectional GRU + Multi-Head Attention.
   - **6 Output Labels:** `forum`, `promotions`, `social_media`, `spam`, `updates`, `verify_code`.
   - **Limitation:** Contains **no Primary class**. In baseline mode, `forum`, `social_media`, `updates`, and `verify_code` are mapped to **`Other / Needs review`** and marked as lossy.

---

## 2. Dataset & Model Download Instructions

MailSort is designed to operate completely offline. Pre-trained checkpoints and large datasets are kept out of git tracking (via `.gitignore`) and can be downloaded using the Hugging Face CLI or Python script.

### A. Download the Reference Checkpoint (Model)
To download the companion model weights (`best_high_accuracy_model.h5`, `high_accuracy_email_classifier.tflite`, and tokenizer JSON) into `.hfcache`:

**Using Hugging Face CLI:**
```bash
# Ensure huggingface_hub is installed
pip install huggingface_hub

# Download model into local cache
huggingface-cli download jason23322/high-accuracy-email-classifier --cache-dir .hfcache
```

**Using Python:**
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="jason23322/high-accuracy-email-classifier",
    cache_dir=".hfcache",
    local_files_only=False
)
```

### B. Download the Companion Dataset (13,477 Emails)
The companion dataset ([`jason23322/email-classification-dataset`](https://huggingface.co/datasets/jason23322/email-classification-dataset)) contains 13,477 emails across 6 categories (10,780 train, 2,697 test):

**Using Hugging Face CLI:**
```bash
huggingface-cli download --repo-type dataset jason23322/email-classification-dataset --local-dir data/dataset
```

**Using Python:**
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="jason23322/email-classification-dataset",
    repo_type="dataset",
    local_dir="data/dataset",
    local_dir_use_symlinks=False
)
```

> **Automatic Detection:** MailSort automatically checks for `data/dataset/train.csv`. When present, `training/train.py` integrates the companion dataset's verified `promotions` and `spam` records with authentic `Primary` emails. If the dataset has not been downloaded, the pipeline gracefully falls back to the curated reference dataset.

---

## 3. Quick Start & Setup

### Environment Requirements
- **Tested Python Versions:** Python 3.12+ (tested up to Python 3.14.4 on Ubuntu Linux).
- **Environment:** Standalone local execution without GPU requirement.

### Installation
```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

### Configuration (`.env`)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Default configuration in `.env`:
```dotenv
# Actual owner/repository on Hugging Face Hub
MAILSORT_MODEL_REPO_ID=jason23322/high-accuracy-email-classifier

# Root of the Hugging Face Hub cache
HF_HUB_CACHE=/mnt/extra-maal/mail-classifier/.hfcache

# Optional direct directory if files were downloaded using --local-dir
MAILSORT_MODEL_DIR=

# Application mode: 'auto' (default: custom 3-class model), 'custom', or 'baseline'
MAILSORT_MODEL_MODE=auto

# Offline mode (prevents unnecessary network calls during inference)
HF_HUB_OFFLINE=1
```

---

## 4. Running the Application

Launch the Streamlit web application:
```bash
streamlit run app.py
```

The web dashboard features:
- **Single Email Classifier:** Manual Subject and Body inputs with quick preset sample buttons.
- **Batch CSV Classification:** Upload any CSV file, map columns, and export results with exact row-order preservation.
- **Model Architecture & Evaluation:** Live display of measured test metrics, per-class F1-scores, and confusion matrix image.
- **Academic Integrity Statement:** Full transparency on data provenance, label mapping, and external checkpoint attribution.

---

## 5. Running Tests

Run the full automated test suite:
```bash
pytest tests/
```
All **15 unit tests** cover:
- Text standardization and entity tokens (`URL`, `EMAIL`, `NUMBER`).
- Padding, truncation, and out-of-vocabulary handling.
- Honest label mapping (ensuring non-spam/non-promotions labels are never falsely called Primary).
- Offline Hugging Face Hub cache resolution.
- Batch classification row ordering and empty input fallbacks.
- Test accuracy verification on the companion dataset (>90% requirement).

---

## 6. Training & Evaluation Pipeline

To retrain the project-owned classifier and evaluate on the held-out test split:
```bash
python training/train.py
```
This script:
1. Detects `data/dataset/train.csv` (or uses `STARTER_DATASET` fallback).
2. Standardizes text and deduplicates identical messages.
3. Produces stratified train, validation, and test splits in `data/processed/`.
4. Fits the TF-IDF and regularized linear classifier strictly on training data.
5. Serializes the pipeline to `artifacts/mailsort_model.joblib`.
6. Evaluates on the held-out test split and writes `reports/metrics.json` and `reports/confusion_matrix.png`.

---

## 7. Measured Evaluation Results

Evaluated on the held-out test partition (104 samples, 0% test leakage):

| Metric | Measured Value |
|---|---|
| **Overall Accuracy** | **98.08%** |
| **Macro Precision** | **98.79%** |
| **Macro Recall** | **98.58%** |
| **Macro F1-Score** | **98.66%** |
| **Weighted F1-Score** | **98.07%** |

### Per-Class Metrics:
- **Primary:** Precision: `100.0%` | Recall: `100.0%` | F1: `100.0%` (Support: 4)
- **Promotions:** Precision: `100.0%` | Recall: `95.74%` | F1: `97.83%` (Support: 47)
- **Spam:** Precision: `96.36%` | Recall: `100.0%` | F1: `98.15%` (Support: 53)

### Companion Test Set Evaluation:
- **Full Companion Test Set (`data/dataset/test.csv`):** **98.33% Accuracy** across 898 promotions and spam test emails.

---

## 8. Academic Integrity & License

This project is developed for the **BIC703 Machine Learning** course assignment at Bangalore Institute of Technology.

- **Checkpoint Attribution:** The baseline CNN-GRU model and dataset are external artifacts created by `jason23322` under Apache 2.0 license.
- **No Synthetic Metrics:** All reported metrics represent empirical evaluation results generated by `training/evaluate.py`.
- **Offline Guarantee:** Inference executes locally without sending data to external APIs.
- **License:** Apache License 2.0.