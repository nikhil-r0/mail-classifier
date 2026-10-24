import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Hugging Face Configuration
MAILSORT_MODEL_REPO_ID = os.getenv("MAILSORT_MODEL_REPO_ID", "jason23322/high-accuracy-email-classifier")
HF_HUB_CACHE = os.getenv("HF_HUB_CACHE", str(BASE_DIR / ".hfcache"))
MAILSORT_MODEL_DIR = os.getenv("MAILSORT_MODEL_DIR", "")
MAILSORT_MODEL_MODE = os.getenv("MAILSORT_MODEL_MODE", "auto").lower()  # 'custom', 'baseline', or 'auto'
HF_HUB_OFFLINE = os.getenv("HF_HUB_OFFLINE", "1") == "1"

# Project Artifacts Paths
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "data"

CUSTOM_MODEL_PATH = ARTIFACTS_DIR / "mailsort_model.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"
REPORT_MD_PATH = REPORTS_DIR / "report.md"

# Categories
TARGET_CATEGORIES = ["Primary", "Promotions", "Spam"]
FALLBACK_CATEGORY = "Other / Needs review"

BASELINE_CATEGORIES = [
    "forum",
    "promotions",
    "social_media",
    "spam",
    "updates",
    "verify_code"
]

# Lossy mapping from the 6-class external baseline to the Gmail-inspired categories
BASELINE_TO_TARGET_MAP = {
    "promotions": "Promotions",
    "spam": "Spam",
    "forum": "Other / Needs review",
    "social_media": "Other / Needs review",
    "updates": "Other / Needs review",
    "verify_code": "Other / Needs review",
}

# Confidence threshold below which prediction is flagged as uncertain
# (For 3 classes with uniform baseline of 0.333, < 0.40 indicates low confidence)
CONFIDENCE_THRESHOLD = 0.40
