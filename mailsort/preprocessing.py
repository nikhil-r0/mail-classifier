import re
import numpy as np
from typing import List, Dict, Tuple, Optional

# Regular expressions for text standardization
URL_REGEX = re.compile(r"http[s]?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
NUMBER_REGEX = re.compile(r"\b\d+\b")
PUNCTUATION_REGEX = re.compile(r"[^\w\s]")
WHITESPACE_REGEX = re.compile(r"\s+")

# Keyword heuristics for pattern-based explainability (clearly documented as heuristic indicators)
EXPLANATION_PATTERNS = {
    "Promotions": [
        (r"\b(?:\d+%\s*off|discount|clearance|save\s+\$?\d+|sale|deal|voucher|coupon)\b", "Discount or sale terminology"),
        (r"\b(?:limited\s+time|ends\s+tonight|last\s+chance|act\s+now|flash\s+sale)\b", "Urgency/promotional call to action"),
        (r"\b(?:free\s+shipping|exclusive\s+offer|shop\s+now|buy\s+now|promo\s+code)\b", "E-commerce or shopping offer"),
        (r"\b(?:newsletter|unsubscribe|preferences|view\s+in\s+browser)\b", "Commercial newsletter or mailing list footer"),
    ],
    "Spam": [
        (r"\b(?:congratulations|you\s+won|selected\s+as\s+a\s+winner|claim\s+your\s+prize|lottery)\b", "Prize or lottery scam cues"),
        (r"\b(?:wire\s+transfer|inheritance|cryptocurrency\s+grant|western\s+union|million\s+dollars)\b", "Suspicious financial solicitation"),
        (r"\b(?:account\s+suspended|verify\s+immediately|security\s+breach|unauthorized\s+login)\b", "Urgent phishing or imposter pressure"),
        (r"\b(?:click\s+here\s+to\s+claim|confidential\s+proposal|risk-free)\b", "High-risk spam click prompt"),
    ],
    "Primary": [
        (r"\b(?:hi|hello|dear|hey)\s+[A-Z][a-z]+", "Direct personal greeting to individual"),
        (r"\b(?:meeting|agenda|schedule|reschedule|discuss|notes|sync|catch\s+up)\b", "Work or schedule coordination"),
        (r"\b(?:project|assignment|deadline|submission|review|draft|attached|feedback)\b", "Direct academic or professional collaboration"),
        (r"\b(?:let\s+me\s+know|what\s+do\s+you\s+think|thanks,|regards,|sincerely)\b", "Interpersonal correspondence sign-off"),
    ],
}

def clean_text(text: Optional[str]) -> str:
    """
    Standardize text:
    - Handle None / non-string gracefully
    - Lowercase
    - Replace URLs with 'URL'
    - Replace email addresses with 'EMAIL'
    - Replace numbers with 'NUMBER'
    - Remove punctuation
    - Normalize whitespace
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Unicode normalization / lowercase
    text = text.lower()
    
    # Entity replacements
    text = URL_REGEX.sub("URL", text)
    text = EMAIL_REGEX.sub("EMAIL", text)
    text = NUMBER_REGEX.sub("NUMBER", text)
    
    # Punctuation and whitespace
    text = PUNCTUATION_REGEX.sub(" ", text)
    text = WHITESPACE_REGEX.sub(" ", text).strip()
    return text

def combine_subject_and_body(subject: Optional[str], body: Optional[str]) -> str:
    """
    Combines subject and body into a single structured input string.
    Subject is weighted by appearing prominently at the start.
    """
    s = (subject or "").strip()
    b = (body or "").strip()
    
    if s and b:
        return f"{s}. {b}"
    elif s:
        return s
    elif b:
        return b
    return ""

def text_to_padded_sequence(
    text: str,
    word_index: Dict[str, int],
    max_len: int = 250,
    oov_token_index: int = 1
) -> np.ndarray:
    """
    Tokenizes text and pads or truncates to max_len (6-class neural model input shape).
    """
    cleaned = clean_text(text)
    words = cleaned.split()
    sequence = [word_index.get(w, oov_token_index) for w in words]
    
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    else:
        sequence = sequence + [0] * (max_len - len(sequence))
        
    return np.array([sequence], dtype=np.float32)

def extract_explanation_cues(text: str, predicted_category: str) -> List[str]:
    """
    Extracts transparent text evidence matching known domain patterns.
    Note: These are pattern-based heuristics provided for human readability,
    not internal gradient explanations of neural weights.
    """
    cues = []
    patterns = EXPLANATION_PATTERNS.get(predicted_category, [])
    for regex_pattern, explanation in patterns:
        matches = re.findall(regex_pattern, text, re.IGNORECASE)
        if matches:
            unique_matches = sorted(list(set([m.lower() for m in matches if isinstance(m, str)])))
            cues.append(f"{explanation}: matched '{', '.join(unique_matches[:3])}'")
            
    if not cues:
        cues.append("Classification driven by overall vocabulary distribution and sentence semantics.")
        
    return cues
