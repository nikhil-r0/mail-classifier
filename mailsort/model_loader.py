import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from mailsort.config import (
    BASE_DIR,
    BASELINE_CATEGORIES,
    CUSTOM_MODEL_PATH,
    HF_HUB_CACHE,
    MAILSORT_MODEL_DIR,
    MAILSORT_MODEL_REPO_ID,
)


def resolve_cached_model_file(
    repo_id: str, filename: str, cache_dir: Optional[str] = None
) -> Path:
    """Resolve a model asset from a local cache directory (offline, no network)."""
    from huggingface_hub import hf_hub_download

    if MAILSORT_MODEL_DIR and os.path.isdir(MAILSORT_MODEL_DIR):
        direct_path = Path(MAILSORT_MODEL_DIR) / filename
        if direct_path.is_file():
            return direct_path

    primary_cache = cache_dir or HF_HUB_CACHE or str(BASE_DIR / ".hfcache")
    candidates = [
        primary_cache,
        str(Path(primary_cache) / "huggingface" / "hub" / "hub"),
        str(Path(primary_cache) / "huggingface" / "hub"),
        str(BASE_DIR / ".hfcache"),
        "/nikhil/.hfcache/huggingface/hub/hub",
    ]

    seen = set()
    unique_candidates = []
    for c in candidates:
        if c and c not in seen and os.path.exists(c):
            seen.add(c)
            unique_candidates.append(c)

    for candidate in unique_candidates:
        try:
            resolved = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=candidate,
                local_files_only=True,
            )
            if os.path.isfile(resolved):
                return Path(resolved)
        except Exception:
            continue

    raise FileNotFoundError(
        f"Could not locate cached model file '{filename}' for '{repo_id}'.\n"
        f"Checked cache directories: {unique_candidates}.\n"
        f"Set MAILSORT_MODEL_DIR to a folder containing the file, or HF_HUB_CACHE in .env."
    )


# Backward-compatible alias for tests and callers
resolve_hf_file = resolve_cached_model_file


class BaselineModelWrapper:
    """Wrapper for the 6-class reference checkpoint (local cache only)."""

    def __init__(
        self,
        tokenizer_config: Dict[str, Any],
        keras_model=None,
        tflite_interpreter=None,
    ):
        self.categories = tokenizer_config.get("categories", BASELINE_CATEGORIES)
        self.word_index = tokenizer_config.get("word_index", {})
        self.max_len = tokenizer_config.get("max_len", 250)
        self.keras_model = keras_model
        self.interpreter = tflite_interpreter
        self.model_type = (
            "Keras H5"
            if keras_model
            else ("TFLite" if tflite_interpreter else "Unavailable")
        )

    def is_ready(self) -> bool:
        return self.keras_model is not None or self.interpreter is not None


def load_baseline_model() -> BaselineModelWrapper:
    """Load 6-class neural checkpoint assets from the configured local cache."""
    tok_path = resolve_cached_model_file(
        MAILSORT_MODEL_REPO_ID, "high_accuracy_tokenizer_config.json"
    )
    with open(tok_path, "r", encoding="utf-8") as f:
        tok_config = json.load(f)

    keras_model = None
    tflite_interpreter = None

    try:
        h5_path = resolve_cached_model_file(
            MAILSORT_MODEL_REPO_ID, "best_high_accuracy_model.h5"
        )
        try:
            import tensorflow as tf

            keras_model = tf.keras.models.load_model(str(h5_path), compile=False)
        except ImportError:
            try:
                import keras

                keras_model = keras.models.load_model(str(h5_path), compile=False)
            except Exception:
                pass
    except Exception:
        pass

    if keras_model is None:
        try:
            tflite_path = resolve_cached_model_file(
                MAILSORT_MODEL_REPO_ID, "high_accuracy_email_classifier.tflite"
            )
            try:
                import ai_edge_litert.interpreter as tflite

                interp = tflite.Interpreter(model_path=str(tflite_path))
                interp.allocate_tensors()
                tflite_interpreter = interp
            except Exception:
                pass
        except Exception:
            pass

    return BaselineModelWrapper(
        tokenizer_config=tok_config,
        keras_model=keras_model,
        tflite_interpreter=tflite_interpreter,
    )


def load_custom_model(model_path: Optional[Path] = None):
    """Loads the project-owned 3-class classifier trained by our pipeline."""
    path = model_path or CUSTOM_MODEL_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Project-owned model not found at '{path}'.\n"
            f"Run 'python training/train.py' to train and save the model."
        )
    return joblib.load(path)
