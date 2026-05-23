"""
OCR extraction using EasyOCR.

Design decisions:
- EasyOCR selected over Tesseract for better out-of-the-box accuracy on natural
  scene text and minimal setup (no system binary required).
- PaddleOCR is faster but heavier to install; EasyOCR is the practical default.
- Results are filtered to keep only price-like patterns (₹/$ amounts, "X for Y"
  promotions) and short uppercase labels (e.g. "SALE", "BUY 1 GET 1").
- The full image is passed to EasyOCR rather than individual crops because price
  tags occupy distinct zones and OCR context improves recognition accuracy.
"""

from __future__ import annotations

import re
from typing import List

import easyocr
import numpy as np

from config import MIN_OCR_CONFIDENCE, OCR_LANGUAGES

# Matches: ₹125  $1.99  2.50  3 for ₹99  2 for $5
_PRICE_RE = re.compile(
    r"[₹\$]\s*\d+\.?\d*"
    r"|\d+\.\d{2}"
    r"|\d+\s*for\s*[₹\$]?\s*\d+",
    re.IGNORECASE,
)

# Matches short uppercase/digit labels like "SALE", "MRP", "20% OFF"
_LABEL_RE = re.compile(r"^[A-Z0-9%\s\.\,/\-\+]{2,20}$")


def is_price_or_label(text: str) -> bool:
    """Return True if text looks like a shelf price tag or promotional label."""
    stripped = text.strip()
    if _PRICE_RE.search(stripped):
        return True
    if _LABEL_RE.match(stripped):
        return True
    return False


class OCRExtractor:
    def __init__(self, languages: List[str] = OCR_LANGUAGES) -> None:
        self._reader = easyocr.Reader(languages, verbose=False)

    def extract(self, image: np.ndarray) -> List[str]:
        """Return filtered shelf labels / price tag texts from the image."""
        raw_results = self._reader.readtext(image)
        labels: List[str] = []

        for _bbox, text, confidence in raw_results:
            if confidence < MIN_OCR_CONFIDENCE:
                continue
            text = text.strip()
            if not text:
                continue
            if is_price_or_label(text):
                labels.append(text)

        return labels
