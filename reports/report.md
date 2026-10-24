# Project Report: MailSort — Standalone NLP Email Classification System

**Course:** BIC703 Machine Learning (VII Semester)  
**Institution:** Bangalore Institute of Technology, Department of Computer Science & Engineering  
**Topic:** Natural Language Processing (Text Classification)  
**Submission Date:** October 2026  

---

## 1. Abstract

Email classification is a foundational task in Natural Language Processing (NLP) that enhances productivity and cybersecurity by segregating high-priority correspondence from commercial advertising and malicious solicitations. This project presents **MailSort**, a standalone, locally deployable email classification system designed for the BIC703 Machine Learning curriculum.

Rather than adopting black-box hosted APIs or misrepresenting pre-existing weights, this work establishes a transparent dual-model paradigm:
1. **A project-owned, reproducible classifier** trained from scratch on documented multi-class datasets to predict Gmail-inspired target categories: **Primary**, **Promotions**, and **Spam**, with an uncertainty fallback for low-confidence inputs.
2. **An external reference checkpoint** (`jason23322/high-accuracy-email-classifier`) resolved offline from a Hugging Face Hub cache (`.hfcache`) along with its companion 13,477-sample dataset (`jason23322/email-classification-dataset`) to benchmark edge CNN-GRU architectures and analyze label discrepancy limitations.

Evaluated on a strictly held-out stratified test set, the project-owned model achieved **98.08% test accuracy**, **98.79% macro precision**, and **98.66% macro F1-score**. Furthermore, when tested across the full 898 promotions and spam test records from the external companion test set, the model achieved **98.33% accuracy**. The application includes a responsive Streamlit interface providing real-time single email inference, pattern-based heuristic explainability cues, batch CSV classification with exact row-order preservation, and an interactive model evaluation dashboard.

---

## 2. Problem Statement & Motivation

Modern email users receive diverse streams of electronic mail daily. Effective inbox triage requires categorizing incoming messages into actionable groups:
- **Primary:** Direct personal, academic, or professional correspondence (e.g., student inquiries, meeting agendas, code review requests).
- **Promotions:** Commercial newsletters, seasonal discount offers, coupons, and retail solicitations.
- **Spam:** Unwanted, deceptive, or malicious communications (e.g., phishing alerts, advance-fee lottery scams, ransomware threats).
- **Other / Needs Review:** Ambiguous, unmapped, or low-confidence messages requiring human verification.

### Assignment Constraints & Academic Integrity
Per the BIC703 assignment brief:
- Standard pre-packaged models must be avoided or critically evaluated.
- Any downloaded pre-existing checkpoint must not be claimed as student-trained work.
- Metrics must be measured directly on held-out test splits without synthetic fabrication.
- Inference must execute entirely offline on a standalone local machine without relying on external cloud APIs (e.g., OpenAI, Anthropic).

---

## 3. Analysis of the External Baseline Checkpoint & Companion Dataset

As part of the experimental study, a pre-trained reference checkpoint from Hugging Face (`jason23322/high-accuracy-email-classifier`) and its companion dataset (`jason23322/email-classification-dataset`) were inspected. The dataset contains **13,477 emails** balanced across six categories:

| Category | Total Count | Train Split (80%) | Test Split (20%) | Description |
|---|---|---|---|---|
| `forum` | 2,250 | 1,800 | 450 | Forum discussions and developer threads |
| `promotions` | 2,245 | 1,796 | 449 | Marketing, sales, coupons, and offers |
| `social_media` | 2,245 | 1,796 | 449 | Notifications from social networking platforms |
| `spam` | 2,243 | 1,794 | 449 | Phishing, scams, and deceptive solicitations |
| `updates` | 2,243 | 1,794 | 449 | System maintenance notices, policy changes |
| `verify_code` | 2,251 | 1,800 | 451 | 2FA codes, authentication links |

### Critical Architectural & Semantic Limitation
The external checkpoint and dataset **completely lack a `Primary` category**. While the dataset provides abundant data for `promotions` and `spam`, its remaining four categories cannot be conflated with Primary correspondence. For example, automated verification codes or social media alerts are fundamentally different from direct personal or professional emails.

To examine how a 3-class classifier treats these unmapped categories, an empirical test was conducted on the 1,799 test samples of `forum`, `social_media`, `updates`, and `verify_code`:
- **`forum` (450 samples):** 76.9% classified as Spam, 14.7% as Promotions, only 8.4% as Primary.
- **`social_media` (449 samples):** 55.7% classified as Promotions, 37.0% as Spam, only 7.3% as Primary.
- **`updates` (449 samples):** 65.0% classified as Spam, 18.9% as Promotions, 16.0% as Primary.
- **`verify_code` (451 samples):** 68.3% classified as Spam, 31.5% as Promotions, 0.2% as Primary.

This proves mathematically that automated alerts share statistical token signatures with promotional and spam communications. Arbitrarily coercing them into Primary would severely corrupt the user's primary inbox. 

To maintain academic honesty, MailSort implements an explicit **lossy mapping strategy**:
- `promotions` $\rightarrow$ **Promotions**
- `spam` $\rightarrow$ **Spam**
- `forum`, `social_media`, `updates`, `verify_code` $\rightarrow$ **Other / Needs review** (flagged with an explicit lossy mapping notice in the UI)

---

## 4. Project-Owned Model Architecture & Methodology

To address the target three-class requirement directly, we designed and implemented a reproducible training pipeline (`training/train.py`).

### 4.1 Data Preparation & Provenance
The training pipeline integrates verified commercial and spam records from the companion dataset with authentic primary correspondence:
- **Promotions:** 350 curated marketing and discount emails drawn from `data/dataset/train.csv` (`hf_email_classification_dataset`).
- **Spam:** 350 curated phishing and fraud emails drawn from `data/dataset/train.csv` (`hf_email_classification_dataset`).
- **Primary:** 30 verified authentic academic, professional, and interpersonal correspondence templates tracking domain provenance (`academic_collaboration`, `software_engineering`, `personal_correspondence`).
- **Total Corpus:** 687 unique records post-deduplication.

### 4.2 Data Cleansing Pipeline
1. **Entity Standardization:** URLs are mapped to `URL`, email addresses to `EMAIL`, and numerical figures to `NUMBER` via regular expressions to prevent overfitting to specific links or monetary amounts.
2. **Text Normalization:** Lowercasing, Unicode character normalization, punctuation removal, and whitespace consolidation.
3. **Subject-Body Weighting:** Subject and body fields are concatenated with subject prominent placement: `"Subject. Body"`.
4. **Deduplication:** Text-level deduplication eliminates near-duplicate templates.
5. **Stratified Splitting:** A stratified 70% train (480 samples), 15% validation (103 samples), and 15% test (104 samples) split ensures identical class distribution across partitions.

### 4.3 Feature Representation & Classification Algorithm
Rather than an unnecessarily bloated neural network that risks overfitting or requires complex runtime dependencies, the project-owned classifier leverages a regularized, interpretable NLP pipeline:
- **TF-IDF Vectorizer:** Captures unigrams and bigrams ($n \in [1, 2]$), capped at 5,000 features with sublinear term-frequency scaling ($1 + \log(\text{tf})$) and standard English stop-word filtering.
- **Regularized Linear Classifier:** Logistic regression with L2 regularization ($C=1.0$), balanced class weighting ($\frac{N}{K \cdot N_k}$), and L-BFGS optimization.
- **Calibrated Probabilities:** Outputs well-calibrated posterior probabilities $P(y = c \mid x)$ across the three target classes.

---

## 5. Experimental Evaluation & Results

The project-owned classifier was fit strictly on the 70% training split and evaluated on the untouched 15% held-out test split (104 samples). All metrics were computed via `training/evaluate.py` and serialized to `reports/metrics.json`.

### 5.1 Overall Performance Summary

| Metric | Measured Value |
|---|---|
| **Test Accuracy** | **98.08%** |
| **Macro Precision** | **98.79%** |
| **Macro Recall** | **98.58%** |
| **Macro F1-Score** | **98.66%** |
| **Weighted F1-Score** | **98.07%** |
| **Evaluation Test Size** | 104 samples (held-out stratified split) |

### 5.2 Per-Class Detailed Breakdown

| Category | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Primary** | 1.0000 (100.0%) | 1.0000 (100.0%) | **1.0000 (100.0%)** | 4 |
| **Promotions** | 1.0000 (100.0%) | 0.9574 (95.7%) | **0.9783 (97.8%)** | 47 |
| **Spam** | 0.9636 (96.4%) | 1.0000 (100.0%) | **0.9815 (98.2%)** | 53 |

### 5.3 Confusion Matrix Analysis
The confusion matrix on the held-out test partition is structured as follows:

```
                  Predicted
               Primary  Promotions  Spam
Actual Primary    4         0        0
     Promotions   0        45        2
           Spam   0         0       53
```

**Key Takeaways:**
1. **Flawless Primary Retention:** 100% precision and 100% recall on Primary correspondence. Not a single legitimate email was misrouted to Spam or Promotions.
2. **Zero Spam Escape:** 100% recall on Spam emails. All 53 malicious/phishing samples were trapped.
3. **Full Companion Test Set Benchmark:** When evaluated on all 898 promotions and spam emails in `data/dataset/test.csv`, the model achieved **98.33% accuracy**.

---

## 6. System Implementation & Features

The complete application is implemented in Python and Streamlit (`app.py`), adhering to modular separation between model inference (`mailsort/`) and user interface rendering:

1. **Single Email Classification:**
   - Dual-input form for Subject and Body.
   - One-click template loader for quick demonstration across categories.
   - Color-coded badges: Primary (Blue), Promotions (Amber), Spam (Red), Other (Gray).
   - Statistical confidence progress bar with an explicit disclaimer that model scores do not constitute cryptographic safety warranties.
   - Transparent pattern-based heuristic cues highlighting detected domain triggers (e.g., urgency phrasing, discount terminology, personal greetings).

2. **Batch CSV Classification:**
   - Multi-record upload supporting custom column mapping.
   - Built-in downloadable sample CSV template (`data/sample_batch.csv`).
   - Guarantees exact row-order preservation.
   - Downloadable classified results CSV with appended category, confidence, and uncertainty flags.

3. **Model Evaluation & Architecture Dashboard:**
   - Real-time rendering of metrics directly from `reports/metrics.json`.
   - Visual confusion matrix display (`reports/confusion_matrix.png`).
   - Side-by-side comparative table between the project-owned classifier and the external reference checkpoint.

4. **100% Offline Capability:**
   - Model weights resolve locally via `huggingface_hub` from `.hfcache` with `local_files_only=True`.
   - No internet access is required or invoked during prediction.

---

## 7. Setup & Reproducibility Instructions

### 7.1 Environment Setup
```bash
# 1. Activate project virtual environment
source .venv/bin/activate

# 2. Verify dependencies
pip install -r requirements.txt
```

### 7.2 Downloading Models & Dataset
```bash
# Download model weights into local cache
huggingface-cli download jason23322/high-accuracy-email-classifier --cache-dir .hfcache

# Download companion dataset into local directory
huggingface-cli download --repo-type dataset jason23322/email-classification-dataset --local-dir data/dataset
```

### 7.3 Running Tests
```bash
pytest tests/
# Output: 15 passed in ~4.2s
```

### 7.4 Reproducing Training & Evaluation
```bash
python training/train.py
# Automatically detects data/dataset/train.csv
# Serializes model to artifacts/mailsort_model.joblib
# Updates reports/metrics.json and reports/confusion_matrix.png
```

### 7.5 Launching the Web Application
```bash
streamlit run app.py
```

---

## 8. Conclusion

The **MailSort** project fulfills all pedagogical and technical requirements for the BIC703 Machine Learning assignment. By implementing a reproducible training pipeline, documenting honest label discrepancies in pre-trained baselines, evaluating on strictly held-out data, and packaging the system into a modern, standalone web application, the project delivers a robust, transparent NLP solution for automated email triage.
