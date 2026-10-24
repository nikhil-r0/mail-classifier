import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from mailsort.config import (
    BASE_DIR,
    ARTIFACTS_DIR,
    REPORTS_DIR,
    CUSTOM_MODEL_PATH,
    TARGET_CATEGORIES
)
from mailsort.data_prep import load_or_create_dataset
from training.evaluate import evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def build_classifier_pipeline(random_state: int = 42) -> Pipeline:
    """
    Constructs a reproducible, compact NLP classification pipeline:
    - TF-IDF Vectorizer (unigrams + bigrams, sublinear TF scaling)
    - Regularized Logistic Classifier with balanced class weights
    """
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True,
                stop_words="english",
                strip_accents="unicode"
            )
        ),
        (
            "clf",
            LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
                solver="lbfgs"
            )
        )
    ])

def train_and_save(
    data_csv: str = None,
    random_state: int = 42
) -> Pipeline:
    """
    Executes reproducible training pipeline:
    1. Loads / prepares stratified dataset
    2. Fits pipeline strictly on training data
    3. Saves trained pipeline to artifacts/mailsort_model.joblib
    4. Evaluates on held-out test split and writes reports/metrics.json
    """
    logger.info("Initializing dataset splits...")
    train_df, val_df, test_df = load_or_create_dataset(
        custom_csv_path=data_csv,
        random_state=random_state
    )

    X_train = train_df["cleaned_text"].values
    y_train = train_df["label"].values

    logger.info(f"Training pipeline on {len(X_train)} samples across classes: {sorted(list(set(y_train)))}...")
    pipeline = build_classifier_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)

    # Ensure output directory exists
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, CUSTOM_MODEL_PATH)
    logger.info(f"Successfully serialized project-owned model to {CUSTOM_MODEL_PATH}")

    # Evaluate on held-out test split
    logger.info("Evaluating on held-out test split...")
    metrics = evaluate_model(
        pipeline=pipeline,
        test_df=test_df,
        output_metrics_path=REPORTS_DIR / "metrics.json",
        output_cm_path=REPORTS_DIR / "confusion_matrix.png"
    )

    return pipeline

if __name__ == "__main__":
    custom_csv = sys.argv[1] if len(sys.argv) > 1 else None
    train_and_save(data_csv=custom_csv)
