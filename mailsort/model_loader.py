import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import joblib

from mailsort.config import (
    MAILSORT_MODEL_REPO_ID,
    HF_HUB_CACHE,
    MAILSORT_MODEL_DIR,
    CUSTOM_MODEL_PATH,
    BASELINE_CATEGORIES,
    BASE_DIR
)

logger = logging.getLogger(__name__)

def resolve_hf_file(repo_id: str, filename: str, cache_dir: Optional[str] = None) -> Path:
    """
    Resolves a file from Hugging Face Hub cache using huggingface_hub.
    Supports candidate cache paths and raises actionable errors if not found.
    """
    from huggingface_hub import hf_hub_download
    
    # 1. Direct local directory override
    if MAILSORT_MODEL_DIR and os.path.isdir(MAILSORT_MODEL_DIR):
        direct_path = Path(MAILSORT_MODEL_DIR) / filename
        if direct_path.is_file():
            return direct_path

    # 2. Try configured cache_dir and common layouts
    primary_cache = cache_dir or HF_HUB_CACHE or str(BASE_DIR / ".hfcache")
    candidates = [
        primary_cache,
        str(Path(primary_cache) / "huggingface" / "hub" / "hub"),
        str(Path(primary_cache) / "huggingface" / "hub"),
        str(BASE_DIR / ".hfcache"),
        "/nikhil/.hfcache/huggingface/hub/hub",
    ]

    # Deduplicate while preserving order
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c and c not in seen and os.path.exists(c):
            seen.add(c)
            unique_candidates.append(c)

    last_error = None
    for candidate in unique_candidates:
        try:
            resolved = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=candidate,
                local_files_only=True
            )
            if os.path.isfile(resolved):
                return Path(resolved)
        except Exception as e:
            last_error = e

    raise FileNotFoundError(
        f"Could not locate cached Hugging Face asset '{filename}' for repository '{repo_id}'.\n"
        f"Checked cache directories: {unique_candidates}.\n"
        f"To fix: Ensure the repository has been downloaded via Hugging Face CLI or set HF_HUB_CACHE in .env."
    )

class BaselineModelWrapper:
    """
    Wrapper for the 6-class external reference checkpoint.
    """
    def __init__(self, tokenizer_config: Dict[str, Any], keras_model=None, tflite_interpreter=None):
        self.categories = tokenizer_config.get("categories", BASELINE_CATEGORIES)
        self.word_index = tokenizer_config.get("word_index", {})
        self.max_len = tokenizer_config.get("max_len", 250)
        self.keras_model = keras_model
        self.interpreter = tflite_interpreter
        self.model_type = "Keras H5" if keras_model else ("TFLite" if tflite_interpreter else "Unavailable")

    def is_ready(self) -> bool:
        return self.keras_model is not None or self.interpreter is not None

def load_baseline_model() -> BaselineModelWrapper:
    """
    Attempts to load the baseline model assets (tokenizer JSON and H5/TFLite model).
    """
    # 1. Resolve tokenizer config
    tok_path = resolve_hf_file(MAILSORT_MODEL_REPO_ID, "high_accuracy_tokenizer_config.json")
    with open(tok_path, "r", encoding="utf-8") as f:
        tok_config = json.load(f)

    keras_model = None
    tflite_interpreter = None

    # 2. Try loading Keras .h5 model
    try:
        h5_path = resolve_hf_file(MAILSORT_MODEL_REPO_ID, "best_high_accuracy_model.h5")
        try:
            import tensorflow as tf
            keras_model = tf.keras.models.load_model(str(h5_path), compile=False)
            logger.info("Successfully loaded baseline H5 model via TensorFlow.")
        except ImportError:
            try:
                import keras
                keras_model = keras.models.load_model(str(h5_path), compile=False)
                logger.info("Successfully loaded baseline H5 model via Keras.")
            except Exception as e:
                logger.warning(f"Could not load H5 model via Keras: {e}")
    except Exception as e:
        logger.warning(f"Could not resolve or load best_high_accuracy_model.h5: {e}")

    # 3. If Keras model not loaded, try TFLite
    if keras_model is None:
        try:
            tflite_path = resolve_hf_file(MAILSORT_MODEL_REPO_ID, "high_accuracy_email_classifier.tflite")
            try:
                import ai_edge_litert.interpreter as tflite
                interp = tflite.Interpreter(model_path=str(tflite_path))
                interp.allocate_tensors()
                tflite_interpreter = interp
                logger.info("Successfully loaded baseline TFLite interpreter.")
            except Exception as tflite_err:
                logger.warning(f"TFLite interpreter allocation failed (may require Flex delegate): {tflite_err}")
        except Exception as e:
            logger.warning(f"Could not resolve TFLite asset: {e}")

    return BaselineModelWrapper(
        tokenizer_config=tok_config,
        keras_model=keras_model,
        tflite_interpreter=tflite_interpreter
    )

def load_custom_model(model_path: Optional[Path] = None):
    """
    Loads the project-owned 3-class classifier trained by our pipeline.
    """
    path = model_path or CUSTOM_MODEL_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Project-owned model not found at '{path}'.\n"
            f"Run 'python training/train.py' to train and save the model."
        )
    return joblib.load(path)
