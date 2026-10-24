import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from mailsort.config import (
    MAILSORT_MODEL_REPO_ID,
    HF_HUB_CACHE,
    CUSTOM_MODEL_PATH,
    TARGET_CATEGORIES,
    FALLBACK_CATEGORY
)
from mailsort.model_loader import resolve_cached_model_file, load_baseline_model, load_custom_model
from mailsort.inference import classify_email, classify_batch

def test_resolve_cached_model_file():
    tok_path = resolve_cached_model_file(
        MAILSORT_MODEL_REPO_ID, "high_accuracy_tokenizer_config.json"
    )
    assert tok_path.is_file()
    assert tok_path.name == "high_accuracy_tokenizer_config.json"

def test_load_baseline_tokenizer():
    baseline_wrapper = load_baseline_model()
    assert len(baseline_wrapper.categories) == 6
    assert "spam" in baseline_wrapper.categories
    assert "promotions" in baseline_wrapper.categories
    assert len(baseline_wrapper.word_index) > 100

def test_load_custom_model():
    if not CUSTOM_MODEL_PATH.is_file():
        pytest.skip("Project-owned model not yet trained. Run training/train.py first.")
    
    model = load_custom_model()
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    assert set(model.classes_) == set(TARGET_CATEGORIES)

def test_classify_email_custom():
    if not CUSTOM_MODEL_PATH.is_file():
        pytest.skip("Project-owned model not yet trained.")

    res = classify_email("Team sync meeting", "Let's review the assignment draft tomorrow morning.")
    assert "predicted_category" in res
    assert "confidence" in res
    assert "all_probabilities" in res
    assert "explanation" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert sum(res["all_probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert res["predicted_category"] in TARGET_CATEGORIES or res["predicted_category"] == FALLBACK_CATEGORY

def test_classify_empty_input():
    res = classify_email("", "")
    assert res["predicted_category"] == FALLBACK_CATEGORY
    assert res["confidence"] == 0.0
    assert res["is_uncertain"] is True
    assert "Empty email" in res["explanation"][0]

def test_classify_batch_order_preservation():
    if not CUSTOM_MODEL_PATH.is_file():
        pytest.skip("Project-owned model not yet trained.")

    df = pd.DataFrame([
        {"id": 1, "subject": "Meeting", "body": "Discuss project deliverables."},
        {"id": 2, "subject": "Sale 50% Off", "body": "Buy now and save on all products!"},
        {"id": 3, "subject": "Winner", "body": "Claim your lottery prize now!"},
    ])

    out_df = classify_batch(df, subject_col="subject", body_col="body")
    assert len(out_df) == 3
    assert list(out_df["id"]) == [1, 2, 3]
    assert "predicted_category" in out_df.columns
    assert "confidence" in out_df.columns
    assert "is_uncertain" in out_df.columns
    assert "model_used" in out_df.columns

def test_companion_dataset_predictions():
    test_csv_path = Path("data/dataset/test.csv")
    if not test_csv_path.is_file():
        pytest.skip("Companion dataset test.csv not present.")

    if not CUSTOM_MODEL_PATH.is_file():
        pytest.skip("Custom model not trained.")

    test_df = pd.read_csv(test_csv_path)
    # Sample 20 promotions and 20 spam from the real companion test set
    promos = test_df[test_df["category"] == "promotions"].sample(20, random_state=42)
    spams = test_df[test_df["category"] == "spam"].sample(20, random_state=42)
    sample_df = pd.concat([promos, spams], ignore_index=True)

    classified = classify_batch(sample_df, subject_col="subject", body_col="body")
    
    correct = 0
    for _, row in classified.iterrows():
        expected = row["category"].capitalize()
        if row["predicted_category"] == expected:
            correct += 1

    accuracy = correct / len(classified)
    # The classifier must achieve at least 90% accuracy on real companion test samples
    assert accuracy >= 0.90, f"Expected >= 90% accuracy on companion dataset sample, got {accuracy:.2%}"

