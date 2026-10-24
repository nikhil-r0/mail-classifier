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

2. **6-Class Reference Checkpoint (optional):**
   - **Origin:** Pre-trained reference weights resolved from your local cache only (see §2).
   - **Architecture:** Multi-scale CNN + Bidirectional GRU + Multi-Head Attention.
   - **6 Output Labels:** `forum`, `promotions`, `social_media`, `spam`, `updates`, `verify_code`.
   - **Limitation:** Contains **no Primary class**. In reference mode, `forum`, `social_media`, `updates`, and `verify_code` are mapped to **`Other / Needs review`** and marked as lossy.

---

## 2. Local Cache & Companion Dataset

MailSort runs offline. Large weights and datasets stay out of git (see `.gitignore`).

### A. Reference checkpoint files
Copy or cache these files under `.hfcache` (or set `HF_HUB_CACHE` / `MAILSORT_MODEL_DIR` in `.env`):

- `high_accuracy_tokenizer_config.json`
- `best_high_accuracy_model.h5` and/or `high_accuracy_email_classifier.tflite`

`MAILSORT_MODEL_REPO_ID` in `.env` must match the cache layout your tooling used when the files were saved.

### B. Companion dataset (optional, 13,477 emails)
Place `train.csv` and `test.csv` under `data/dataset/` (six categories: 10,780 train / 2,697 test).

> **Automatic Detection:** When `data/dataset/train.csv` exists, `training/train.py` blends companion `promotions` and `spam` rows with curated `Primary` emails; otherwise it uses the built-in starter corpus.

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

See `.env.example` for cache paths and inference mode (`auto`, `custom`, or `baseline`).

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
- Offline local cache resolution for reference weights.
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

- **Reference checkpoint:** The optional 6-class CNN-GRU weights are external reference artifacts, not trained by this pipeline.
- **No Synthetic Metrics:** All reported metrics represent empirical evaluation results generated by `training/evaluate.py`.
- **Offline Guarantee:** Inference executes locally without sending data to external APIs.
- **License:** Apache License 2.0.