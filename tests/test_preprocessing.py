import pytest
import numpy as np
from mailsort.preprocessing import (
    clean_text,
    combine_subject_and_body,
    text_to_padded_sequence,
    extract_explanation_cues,
    URL_REGEX,
    EMAIL_REGEX,
    NUMBER_REGEX
)

def test_clean_text_normalization():
    # Lowercase, entity normalization, punctuation stripping
    raw = "URGENT: Visit https://secure-bank.example.com or email admin@bank.org! Call 12345 today."
    cleaned = clean_text(raw)
    assert "URL" in cleaned
    assert "EMAIL" in cleaned
    assert "NUMBER" in cleaned
    assert "urgent" in cleaned
    assert "!" not in cleaned
    assert ":" not in cleaned

def test_clean_text_edge_cases():
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_text("    ") == ""
    assert clean_text(12345) == "NUMBER"
    # Unicode handling
    unicode_text = "Café résumé — 100% genuine deal 🚀"
    cleaned = clean_text(unicode_text)
    assert "caf" in cleaned
    assert "genuine" in cleaned

def test_combine_subject_and_body():
    assert combine_subject_and_body("Meeting tomorrow", "Let's review the code.") == "Meeting tomorrow. Let's review the code."
    assert combine_subject_and_body("Only Subject", "") == "Only Subject"
    assert combine_subject_and_body(None, "Only Body") == "Only Body"
    assert combine_subject_and_body(None, None) == ""
    assert combine_subject_and_body("   ", "   ") == ""

def test_text_to_padded_sequence():
    vocab = {"hello": 2, "world": 3, "test": 4}
    # Text shorter than max_len
    seq = text_to_padded_sequence("Hello world", word_index=vocab, max_len=5, oov_token_index=1)
    assert isinstance(seq, np.ndarray)
    assert seq.shape == (1, 5)
    # 2, 3, padded with 0
    assert list(seq[0]) == [2, 3, 0, 0, 0]

    # Text longer than max_len (truncation)
    seq_trunc = text_to_padded_sequence("Hello world test unknown words", word_index=vocab, max_len=3, oov_token_index=1)
    assert seq_trunc.shape == (1, 3)
    assert list(seq_trunc[0]) == [2, 3, 4]

    # Out of vocabulary words
    seq_oov = text_to_padded_sequence("unseen token", word_index=vocab, max_len=3, oov_token_index=1)
    assert list(seq_oov[0]) == [1, 1, 0]

def test_extract_explanation_cues():
    # Promotions cues
    promo_text = "Get 50% off everything in our clearance sale! Limited time offer."
    cues_promo = extract_explanation_cues(promo_text, "Promotions")
    assert len(cues_promo) > 0
    assert any("Discount" in c or "Urgency" in c for c in cues_promo)

    # Spam cues
    spam_text = "Congratulations! You won $1,000,000 in the international lottery."
    cues_spam = extract_explanation_cues(spam_text, "Spam")
    assert len(cues_spam) > 0
    assert any("lottery" in c.lower() or "prize" in c.lower() for c in cues_spam)

    # Primary cues
    primary_text = "Hi Alex, attached is the agenda for our project sync meeting tomorrow."
    cues_primary = extract_explanation_cues(primary_text, "Primary")
    assert len(cues_primary) > 0
    assert any("greeting" in c.lower() or "work" in c.lower() or "collaboration" in c.lower() for c in cues_primary)

    # Fallback explanation
    empty_cues = extract_explanation_cues("The quick brown fox jumps over the lazy dog.", "Unknown")
    assert len(empty_cues) == 1
    assert "vocabulary distribution" in empty_cues[0]
