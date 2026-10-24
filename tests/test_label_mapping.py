import pytest
from mailsort.config import (
    BASELINE_CATEGORIES,
    BASELINE_TO_TARGET_MAP,
    TARGET_CATEGORIES,
    FALLBACK_CATEGORY
)

def test_baseline_categories_definition():
    expected_baseline = ["forum", "promotions", "social_media", "spam", "updates", "verify_code"]
    assert sorted(BASELINE_CATEGORIES) == sorted(expected_baseline)
    assert "primary" not in [c.lower() for c in BASELINE_CATEGORIES]

def test_baseline_mapping_honesty():
    """
    CRITICAL RULE from AGENTS.md:
    Do not mislabel Social Media, Forum, Updates, or Verify Code as Primary.
    They must map to 'Other / Needs review' to indicate lossy mapping.
    """
    assert BASELINE_TO_TARGET_MAP["promotions"] == "Promotions"
    assert BASELINE_TO_TARGET_MAP["spam"] == "Spam"

    # Non-promotions/spam categories MUST NOT map to Primary
    assert BASELINE_TO_TARGET_MAP["forum"] == FALLBACK_CATEGORY
    assert BASELINE_TO_TARGET_MAP["social_media"] == FALLBACK_CATEGORY
    assert BASELINE_TO_TARGET_MAP["updates"] == FALLBACK_CATEGORY
    assert BASELINE_TO_TARGET_MAP["verify_code"] == FALLBACK_CATEGORY

    for cat in ["forum", "social_media", "updates", "verify_code"]:
        assert BASELINE_TO_TARGET_MAP[cat] != "Primary"

def test_target_categories():
    assert "Primary" in TARGET_CATEGORIES
    assert "Promotions" in TARGET_CATEGORIES
    assert "Spam" in TARGET_CATEGORIES
    assert len(TARGET_CATEGORIES) == 3
