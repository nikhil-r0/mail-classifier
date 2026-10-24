# AGENTS.md — MailSort (NLP Email Classification)

## 1. Project objective

Build **MailSort**, a polished, standalone email-classification application for the BIC703 Machine Learning assignment (VII semester, Bangalore Institute of Technology; topic: Natural Language Processing).

The application should accept an email subject and body, classify it, explain the predicted category and confidence, and let a user classify a batch of sample emails from CSV. The app must run locally and must not call a hosted AI API during inference.

The intended Gmail-inspired target categories are:
- **Primary** — ordinary personal or direct correspondence that is not promotional or spam.
- **Promotions** — offers, marketing, discounts, newsletters and advertisements.
- **Spam** — unwanted, suspicious or fraudulent email.

A fourth **Other / Needs review** outcome is allowed when an available model cannot distinguish Primary reliably or the prediction is uncertain. Do not force every non-spam email into Primary.

## 2. Assignment constraints — follow these

The assignment brief says online datasets may be used, the work must be an application using the specified ML technique, it should be demonstrable on a standalone system, standard models should be avoided, and the submitted report must be accompanied by code and dependencies. The deadline in the supplied brief is **25 October 2026**.

Consequently:
1. Do not present an externally downloaded, already-trained checkpoint as a model trained by our team.
2. Create a reproducible training pipeline and train a project-owned classifier on documented, labelled public data. Prefer a compact model whose parameters are randomly initialized and trained by our pipeline. A small CNN/GRU/BiLSTM is acceptable if feasible; avoid pretrained LLMs and hosted inference APIs.
3. The supplied Hugging Face model may be integrated as a **baseline/reference model** and used to verify that offline model loading works. Report its origin and results honestly. Do not claim its stated accuracy as our experimental result.
4. Evaluate the project-owned model on a held-out test split and report measured accuracy, precision, recall, macro F1, per-class results and a confusion matrix. Do not invent results or hard-code metrics.
5. If time or data prevents a reliable Primary/Promotions/Spam model, clearly document the limitation and deliver the working six-class baseline rather than mislabelling its outputs or pretending the requirement was met.
6. Include setup instructions, dependencies, data provenance/licences, and a concise report draft suitable for submission.

## 3. Important limitation of the supplied Hugging Face checkpoint

The model card supplied by the user describes a CNN + bidirectional GRU + multi-head attention model with six labels:
- `Social Media`
- `Promotions`
- `Forum`
- `Spam`
- `Verify Code`
- `Updates`

It does **not** include a `Primary` label. The checkpoint can predict Promotions and Spam directly, but its other four labels are not equivalent to Primary. In a six-label/baseline mode, map those four labels to **Other** for the Gmail-inspired interface, and label that mapping as lossy. Do not silently map them to Primary.

For the requested three target classes, train a separate model with clearly documented training data and labels. If useful, show a baseline comparison with the downloaded six-class model. Never state that the downloaded model was retrained unless the code actually updates its weights and the training run is recorded.

The pasted model card's cited repository URL uses a placeholder (`your-username`). Verify the real model repository/revision and licence from the actual downloaded metadata or Hugging Face page before citing it. Treat claims such as “98%+ accuracy” as claims from the model card until independently reproduced.

## 4. Load the downloaded model through Hugging Face libraries where possible

The user downloaded the model with the Hugging Face CLI into a `.hfcache` directory on a **different drive**. Prefer resolving cached files through `huggingface_hub` instead of hard-coding a snapshot path into application code. Do not move or duplicate large model files just to make the app work.

Important distinction: Hugging Face Hub is the download/cache interface; it does not make every model compatible with `transformers.AutoModel`. The supplied model card describes a custom TensorFlow/Keras CNN + bidirectional GRU + attention model with `.h5` and TFLite artefacts plus a custom tokenizer JSON. Inspect the actual repository files first:

- If the repository contains a Transformers-compatible configuration and model weights, use the appropriate `transformers` `from_pretrained(...)` API with `local_files_only=True` for cached/offline operation.
- If the available artefacts are the described `.h5`/`.tflite` model, use `huggingface_hub.hf_hub_download(...)` or `snapshot_download(...)` to resolve the cached file, then load it with TensorFlow/Keras or TFLite. Do not try to load a custom Keras `.h5` using `AutoModel` unless the repository explicitly implements a compatible Transformers model class.
- A file downloaded using `hf download --local-dir ...` may be in a normal directory rather than the Hub cache layout. Support a direct local-directory fallback for this case.

### Discovery requirements

1. Inspect the current workspace and available mounted filesystems.
2. Inspect the user's `.hfcache` directory and Hugging Face cache metadata. Standard cache directories commonly contain a repository folder such as `models--OWNER--REPO/snapshots/<commit>/...`; don't assume files are directly inside `.hfcache`.
3. Determine and verify the real Hugging Face `repo_id` and revision from CLI metadata/cache folder names or ask the user if ambiguous. The pasted model card includes a placeholder URL (`your-username`), so do not invent the repository identifier.
4. Check for the expected filenames, if they are present in the actual repo:
   - `best_high_accuracy_model.h5`
   - `high_accuracy_email_classifier.tflite`
   - `high_accuracy_tokenizer_config.json`
5. If the assets cannot be resolved through the cache or current accessible filesystem, ask for the absolute cache/local-directory path and actual repo ID. Never fabricate a path or download a duplicate without need.
6. Do not perform expensive recursive scans over the entire drive on every app launch. Use a one-time discovery helper or explicit configuration.

### Configuration

Provide `.env.example` with a Hugging Face-first configuration. Values below are placeholders, not real user paths:

```dotenv
# Actual owner/repository on Hugging Face Hub, verified from cache metadata or repo page.
MAILSORT_MODEL_REPO_ID=owner/repository

# Root of the Hugging Face Hub cache on the other drive.
HF_HUB_CACHE=/absolute/path/to/.hfcache

# Optional direct directory if files were downloaded using --local-dir, or if hub resolution is unavailable.
MAILSORT_MODEL_DIR=

# Optional application mode: custom, baseline, or auto.
MAILSORT_MODEL_MODE=auto

# Set to 1 after all required assets are cached and offline operation is desired.
HF_HUB_OFFLINE=1
```

In Python, prefer `hf_hub_download(repo_id=..., filename=..., cache_dir=..., local_files_only=True)` for known individual files or `snapshot_download(repo_id=..., cache_dir=..., local_files_only=True)` for a complete cached snapshot. Hugging Face Hub returns the resolved local filename/directory; use that return value rather than hard-coding cache snapshot paths. A custom `HF_HUB_CACHE` must point to the actual Hub cache root, not automatically to a nested `models--.../snapshots/...` folder.

The app must not depend on internet access during normal prediction. If the repo has not been cached, display an actionable error telling the user to download the exact verified repo/revision first; don't silently start a network download. Keep any direct local model directory as a fallback, not as a required hard-coded path.

Never commit a real private drive path to source control. Add `.env`, caches, datasets, large model weights and generated temporary files to `.gitignore`; commit `.env.example`, not the real `.env`.

## 5. Verify the external checkpoint before using it

Do not assume the model card's filenames, category order, input dtype or preprocessing are correct until they have been inspected.

- Check which of the `.h5`, `.tflite` and tokenizer JSON files actually exist.
- Read the tokenizer JSON and use its stored `categories`, `word_index` and `max_len` (or the actual equivalent fields). Do not guess output class order.
- Inspect model input/output shapes and dtypes before inference.
- Apply preprocessing consistently with the checkpoint's documented training pipeline: lowercase, URL/email/number standardization, punctuation handling, token conversion and sequence padding/truncation as appropriate. Test edge cases.
- Do not assume TFLite input tensors are `float32`; use the interpreter's reported input dtype and shape.
- Load Keras models for inference without compiling them when supported (`compile=False`). Do not enable unsafe deserialization options just to bypass a load error. Inspect custom layers/configuration first.
- Add a smoke test that loads the assets and predicts on at least one example per known class. If only TFLite loads successfully, it is acceptable to use it for the baseline.
- Make the application display the six actual baseline categories in a model-info panel or make the `Other` mapping explicit. Do not misreport baseline scores as the custom model's scores.

## 6. Data and training plan for the project-owned model

Build a repeatable data-preparation/training pipeline. Use publicly accessible datasets whose terms permit the intended academic use, and record source URLs, licence/attribution and the original label mapping in the README/report.

Suggested starting sources:
- A multiclass email dataset with Personal/Business, Promotions/Newsletter and related labels for candidate Primary and Promotions examples.
- Enron1 or another documented ham/spam email dataset for additional legitimate and spam examples.
- The supplied six-category email dataset/checkpoint may be used for a baseline or category mapping, but it does not itself provide a Primary class.

Before training:
1. Inspect the downloaded data schema and label counts instead of assuming column names or counts.
2. Create a documented mapping from source labels to `Primary`, `Promotions`, and `Spam`. Put genuinely ambiguous/unmapped messages into a reviewed exclusion set or `Other`; do not arbitrarily label them Primary.
3. Deduplicate by normalized subject/body and inspect for near-duplicates or generated templates that could leak across splits.
4. Keep provenance/source identifiers so metrics can be inspected by source.
5. Use a stratified train/validation/test split (or a group/source-aware split where duplicates or templates require it). Fit vocabulary and other learned preprocessing on training data only.
6. Do not train and test on duplicated messages or report training accuracy as test accuracy.
7. Set and record random seeds; save the label order, tokenizer/vocabulary, training configuration, test metrics and confusion matrix alongside the trained model.

A modest scratch-trained neural model is preferred for the assignment. Choose a model size that can train within the available environment. Only add complexity when validation results show a reason to do so.

## 7. Application requirements

Prefer Python + Streamlit for a quick, locally runnable application. Keep model logic separate from UI code.

Minimum features:
- A chat-like/manual email input form with separate Subject and Body fields.
- A classify button and clear category result.
- Confidence/probability display, clearly described as model confidence rather than a guarantee that an email is safe.
- A short explanation of the predicted category based on the label and useful text patterns; do not claim the explanation is a faithful explanation of the neural network unless it is implemented as such.
- CSV upload for batch classification, with a downloadable CSV containing input columns, predicted category and scores.
- A model-info/evaluation page showing the actual model type, training data, label mapping, test metrics and confusion matrix.
- A helpful low-confidence/uncertain outcome rather than confidently guessing Primary.
- Robust input validation, empty-input handling, sensible text-length limits and exceptions with actionable messages.
- No automatic connection to a real Gmail account, no email sending/deleting, and no data upload to a server. Use uploaded/sample emails only.
- No network calls during normal prediction. The app must function offline after required dependencies and files are installed.

A simple, readable UI is more important than unnecessary animations. Make it suitable for a classroom demonstration and include sample messages for each category.

## 8. Suggested repository layout

Adapt to an existing repository rather than overwriting it. A sensible target layout is:

```text
MailSort/
├── AGENTS.md
├── app.py
├── mailsort/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── model_loader.py
│   ├── inference.py
│   └── data_prep.py
├── training/
│   ├── train.py
│   └── evaluate.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_label_mapping.py
│   └── test_model_loading.py
├── artifacts/                 # small generated metadata only; don't duplicate external weights
├── reports/
│   ├── metrics.json
│   └── report.md
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

Do not create empty placeholder implementations just to match this layout. Keep the actual project structure coherent.

## 9. Test and quality requirements

Run whatever tests are possible in the current environment and report exactly what passed or was blocked.

At minimum verify:
- Correct, deterministic source-label mapping.
- Empty strings, long text, Unicode and punctuation handling.
- Unknown/OOV words and padding/truncation behaviour.
- The model's output count matches the stored category order.
- Probabilities are finite and have the expected shape; if the model outputs logits, apply the correct activation.
- Batch classification preserves row order and includes every input row.
- Missing/wrong model path yields a clear error, not an opaque stack trace.
- The UI starts with the intended local command and a sample prediction works offline.
- Metrics reported in the UI/report are read from the real evaluation output, not hard-coded.

Do not claim offline operation unless it has been tested with network access unnecessary for inference.

## 10. Expected commands and documentation

Provide a concise README with:
- Exact Python version tested.
- Environment setup and dependency installation.
- How to set `MAILSORT_MODEL_REPO_ID` and `HF_HUB_CACHE`, how Hugging Face resolves cached files, and how to use `MAILSORT_MODEL_DIR` only as a direct-folder fallback on Linux/macOS and Windows.
- How to launch the app, e.g. `streamlit run app.py`.
- How to run tests and how to train/evaluate the custom model.
- Dataset sources, licenses, preprocessing, label mapping and limitations.
- Which model is used by default, and whether it is the externally downloaded baseline or the project-owned trained model.

Keep `requirements.txt` limited to dependencies actually imported. Avoid installing a large framework stack if the app can use the available TFLite model for baseline inference. Clearly separate optional training dependencies from inference dependencies if that saves installation size.

## 11. Work order for the coding agent

1. Inspect the existing repository, Python environment and mounted drives. Do not overwrite existing work before understanding it.
2. Resolve the model via `huggingface_hub` using the verified repo ID and configured cache; use a direct local-directory fallback only when needed. Record paths only in local environment configuration, not committed source.
3. Implement a minimal inference smoke test and document the actual six-label outputs.
4. Inspect the candidate training datasets and create a justified label mapping for the target classes.
5. Implement reproducible training/evaluation of a project-owned model. If time/data constraints block this, record the blocker and retain the six-class baseline with honest labels.
6. Implement the Streamlit application, sample data, CSV batch inference and model-info view.
7. Test, fix errors, update README and create a concise report draft based on measured results.
8. Finish with a short summary of completed files, exact run command, model path configuration required from the user, evaluation results, and any limitations. Do not claim any task was completed if it was not run or verified.

## 12. Do not do these things

- Do not assume or invent the external drive/mount path.
- Do not move/copy large model files unnecessarily.
- Do not hard-code local file paths or credentials.
- Do not mislabel the model's `Social Media`, `Forum`, `Verify Code` or `Updates` predictions as `Primary`.
- Do not claim the checkpoint's advertised 98%+ accuracy as our result.
- Do not claim the external model was trained by our team.
- Do not claim that a confidence score proves an email is legitimate or safe.
- Do not use an online LLM/API to classify email content.
- Do not silently skip model/data-loading errors or return fabricated fallback predictions.
