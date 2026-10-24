import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from mailsort.config import TARGET_CATEGORIES, REPORTS_DIR

logger = logging.getLogger(__name__)

def evaluate_model(
    pipeline,
    test_df: pd.DataFrame,
    output_metrics_path: Optional[Path] = None,
    output_cm_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Evaluates a trained model pipeline on a held-out test split.
    Calculates measured accuracy, precision, recall, macro F1, per-class metrics,
    and plots the confusion matrix.
    """
    X_test = test_df["cleaned_text"].values
    y_test = test_df["label"].values

    y_pred = pipeline.predict(X_test)
    labels = sorted(list(set(TARGET_CATEGORIES)))

    # Compute overall metrics
    acc = float(accuracy_score(y_test, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    # Compute per-class metrics
    p_per_class, r_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, zero_division=0
    )

    per_class_results = {}
    for i, label in enumerate(labels):
        per_class_results[label] = {
            "precision": round(float(p_per_class[i]), 4),
            "recall": round(float(r_per_class[i]), 4),
            "f1_score": round(float(f1_per_class[i]), 4),
            "support": int(support[i])
        }

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_dict = {
        "labels": labels,
        "matrix": cm.tolist()
    }

    metrics = {
        "model_name": "MailSort Project-Owned Classifier",
        "architecture": "TF-IDF (1-2 ngrams) + Regularized Linear Classifier",
        "evaluation_split_size": len(y_test),
        "overall": {
            "accuracy": round(acc, 4),
            "macro_precision": round(float(p_macro), 4),
            "macro_recall": round(float(r_macro), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_f1": round(float(f1_weighted), 4)
        },
        "per_class": per_class_results,
        "confusion_matrix": cm_dict
    }

    # Save metrics JSON
    metrics_path = output_metrics_path or (REPORTS_DIR / "metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved test metrics to {metrics_path}")

    # Plot & save confusion matrix
    cm_path = output_cm_path or (REPORTS_DIR / "confusion_matrix.png")
    cm_path.parent.mkdir(parents=True, exist_ok=True)
    if HAS_PLOT:
        try:
            plt.figure(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels
            )
            plt.title("Confusion Matrix — MailSort Held-Out Test Set")
            plt.ylabel("Actual Category")
            plt.xlabel("Predicted Category")
            plt.tight_layout()
            plt.savefig(cm_path, dpi=200)
            plt.close()
            logger.info(f"Saved confusion matrix plot to {cm_path}")
        except Exception as e:
            logger.warning(f"Could not render confusion matrix plot: {e}")
    else:
        logger.info("Plotting libraries (matplotlib/seaborn) not installed; confusion matrix preserved in metrics.json.")

    return metrics
