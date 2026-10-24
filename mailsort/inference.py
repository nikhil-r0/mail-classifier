import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from mailsort.config import (
    TARGET_CATEGORIES,
    BASELINE_CATEGORIES,
    BASELINE_TO_TARGET_MAP,
    FALLBACK_CATEGORY,
    CONFIDENCE_THRESHOLD,
    MAILSORT_MODEL_MODE
)
from mailsort.preprocessing import (
    clean_text,
    combine_subject_and_body,
    text_to_padded_sequence,
    extract_explanation_cues
)
from mailsort.model_loader import load_custom_model, load_baseline_model, BaselineModelWrapper

logger = logging.getLogger(__name__)

# Cached model singletons
_CUSTOM_MODEL = None
_BASELINE_MODEL = None

def get_custom_model():
    global _CUSTOM_MODEL
    if _CUSTOM_MODEL is None:
        try:
            _CUSTOM_MODEL = load_custom_model()
        except Exception as e:
            logger.info(f"Custom model not available: {e}")
    return _CUSTOM_MODEL

def get_baseline_model():
    global _BASELINE_MODEL
    if _BASELINE_MODEL is None:
        try:
            _BASELINE_MODEL = load_baseline_model()
        except Exception as e:
            logger.info(f"Baseline model not available: {e}")
    return _BASELINE_MODEL

def classify_email(
    subject: str,
    body: str,
    mode: Optional[str] = None
) -> Dict[str, Any]:
    """
    Classifies an email given subject and body.
    
    Mode options:
    - 'custom': Use project-owned 3-class model (Primary, Promotions, Spam)
    - 'baseline': Use 6-class external reference model (with lossy mapping to Other)
    - 'auto': Use custom model if available, otherwise baseline
    """
    target_mode = (mode or MAILSORT_MODEL_MODE).lower()
    
    # Validation: empty input check
    s = (subject or "").strip()
    b = (body or "").strip()
    if not s and not b:
        return {
            "predicted_category": FALLBACK_CATEGORY,
            "confidence": 0.0,
            "all_probabilities": {c: 0.0 for c in TARGET_CATEGORIES},
            "explanation": ["Empty email content provided. Unable to determine classification."],
            "model_used": "Input Validation",
            "is_uncertain": True,
            "lossy_mapping_note": None
        }

    combined = combine_subject_and_body(s, b)
    cleaned = clean_text(combined)

    # 1. Custom model branch
    custom_model = get_custom_model() if target_mode in ["auto", "custom"] else None

    use_custom = False
    if target_mode == "custom":
        if custom_model is None:
            raise RuntimeError("Project-owned model not found. Please train it first using 'python training/train.py'.")
        use_custom = True
    elif target_mode == "baseline":
        use_custom = False
    else:  # 'auto'
        use_custom = (custom_model is not None)

    baseline_model = get_baseline_model() if not use_custom else None

    if use_custom and custom_model is not None:
        # Predict with scikit-learn pipeline
        classes = list(custom_model.classes_)
        probs = custom_model.predict_proba([cleaned])[0]
        pred_idx = int(np.argmax(probs))
        pred_cat = classes[pred_idx]
        conf = float(probs[pred_idx])
        
        prob_dict = {cat: float(p) for cat, p in zip(classes, probs)}
        for cat in TARGET_CATEGORIES:
            if cat not in prob_dict:
                prob_dict[cat] = 0.0

        is_uncertain = conf < CONFIDENCE_THRESHOLD
        effective_cat = pred_cat if not is_uncertain else FALLBACK_CATEGORY
        explanation = extract_explanation_cues(combined, pred_cat)

        return {
            "predicted_category": effective_cat,
            "raw_prediction": pred_cat,
            "confidence": conf,
            "all_probabilities": prob_dict,
            "explanation": explanation,
            "model_used": "Project-Owned Classifier (3-Class: Primary/Promotions/Spam)",
            "is_uncertain": is_uncertain,
            "lossy_mapping_note": None
        }

    # 2. Baseline reference model branch
    if baseline_model is not None and baseline_model.is_ready():
        seq = text_to_padded_sequence(
            cleaned,
            word_index=baseline_model.word_index,
            max_len=baseline_model.max_len
        )
        
        if baseline_model.keras_model is not None:
            raw_out = baseline_model.keras_model.predict(seq, verbose=0)[0]
        else:
            interp = baseline_model.interpreter
            input_idx = interp.get_input_details()[0]["index"]
            output_idx = interp.get_output_details()[0]["index"]
            interp.set_tensor(input_idx, seq)
            interp.invoke()
            raw_out = interp.get_tensor(output_idx)[0]

        pred_idx = int(np.argmax(raw_out))
        raw_cat = baseline_model.categories[pred_idx]
        raw_conf = float(raw_out[pred_idx])

        # Map to Gmail-inspired categories
        mapped_cat = BASELINE_TO_TARGET_MAP.get(raw_cat, FALLBACK_CATEGORY)
        is_lossy = raw_cat not in ["promotions", "spam"]
        lossy_note = (
            f"The baseline model predicted '{raw_cat}'. This label was mapped to '{mapped_cat}' "
            f"because the 6-class baseline model does not have a native 'Primary' category."
            if is_lossy else None
        )

        explanation = extract_explanation_cues(combined, mapped_cat)
        prob_dict = {cat: float(p) for cat, p in zip(baseline_model.categories, raw_out)}

        return {
            "predicted_category": mapped_cat,
            "raw_prediction": raw_cat,
            "confidence": raw_conf,
            "all_probabilities": prob_dict,
            "explanation": explanation,
            "model_used": f"Baseline Reference Checkpoint ({baseline_model.model_type}, 6-Class)",
            "is_uncertain": raw_conf < CONFIDENCE_THRESHOLD or is_lossy,
            "lossy_mapping_note": lossy_note
        }

    # If neither model is ready, return actionable message
    return {
        "predicted_category": FALLBACK_CATEGORY,
        "confidence": 0.0,
        "all_probabilities": {c: 0.0 for c in TARGET_CATEGORIES},
        "explanation": [
            "Neither the trained custom model nor the baseline model could be executed in the current environment.",
            "Run 'python training/train.py' to generate the project-owned classifier."
        ],
        "model_used": "None",
        "is_uncertain": True,
        "lossy_mapping_note": None
    }

def classify_batch(
    df: pd.DataFrame,
    subject_col: str = "subject",
    body_col: str = "body",
    mode: Optional[str] = None
) -> pd.DataFrame:
    """
    Classifies a batch of emails from a DataFrame while preserving exact row ordering.
    Appends 'predicted_category', 'confidence', 'is_uncertain', and 'model_used' columns.
    """
    out_df = df.copy()
    results = []
    
    for _, row in out_df.iterrows():
        s = str(row.get(subject_col, "")) if pd.notna(row.get(subject_col)) else ""
        b = str(row.get(body_col, "")) if pd.notna(row.get(body_col)) else ""
        res = classify_email(s, b, mode=mode)
        results.append({
            "predicted_category": res["predicted_category"],
            "confidence": round(res["confidence"], 4),
            "is_uncertain": res["is_uncertain"],
            "model_used": res["model_used"]
        })

    res_df = pd.DataFrame(results, index=out_df.index)
    return pd.concat([out_df, res_df], axis=1)
