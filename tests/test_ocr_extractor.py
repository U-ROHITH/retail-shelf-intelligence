"""Unit tests for pipeline/ocr_extractor.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pipeline.ocr_extractor import OCRExtractor, is_price_or_label


# ---------------------------------------------------------------------------
# is_price_or_label helper
# ---------------------------------------------------------------------------

class TestIsPriceOrLabel:
    @pytest.mark.parametrize("text", [
        "₹125",
        "$1.99",
        "2 for ₹99",
        "3 for $5",
        "1.99",
        "2.50",
    ])
    def test_price_patterns_accepted(self, text: str):
        assert is_price_or_label(text) is True

    @pytest.mark.parametrize("text", [
        "SALE",
        "MRP",
        "20% OFF",
        "BUY 2",
        "NEW",
    ])
    def test_short_uppercase_labels_accepted(self, text: str):
        assert is_price_or_label(text) is True

    @pytest.mark.parametrize("text", [
        "This is a long sentence that should definitely not match any pattern here",
        "hello world",
    ])
    def test_long_lowercase_text_rejected(self, text: str):
        assert is_price_or_label(text) is False

    def test_empty_string_rejected(self):
        assert is_price_or_label("") is False


# ---------------------------------------------------------------------------
# OCRExtractor
# ---------------------------------------------------------------------------

@pytest.fixture
def blank_image() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)


class TestOCRExtractor:
    @patch("pipeline.ocr_extractor.easyocr.Reader")
    def test_extract_returns_price_tags(self, mock_reader_cls, blank_image):
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "₹125", 0.92),
            ([[0, 20], [10, 20], [10, 30], [0, 30]], "$1.99", 0.88),
        ]
        mock_reader_cls.return_value = mock_reader

        extractor = OCRExtractor()
        labels = extractor.extract(blank_image)

        assert "₹125" in labels
        assert "$1.99" in labels

    @patch("pipeline.ocr_extractor.easyocr.Reader")
    def test_extract_filters_low_confidence(self, mock_reader_cls, blank_image):
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "₹50", 0.90),
            ([[0, 20], [10, 20], [10, 30], [0, 30]], "₹99", 0.15),  # below threshold
        ]
        mock_reader_cls.return_value = mock_reader

        extractor = OCRExtractor()
        labels = extractor.extract(blank_image)

        assert "₹50" in labels
        assert "₹99" not in labels

    @patch("pipeline.ocr_extractor.easyocr.Reader")
    def test_extract_filters_non_label_text(self, mock_reader_cls, blank_image):
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "random long sentence here", 0.95),
        ]
        mock_reader_cls.return_value = mock_reader

        extractor = OCRExtractor()
        labels = extractor.extract(blank_image)
        assert labels == []

    @patch("pipeline.ocr_extractor.easyocr.Reader")
    def test_extract_empty_results(self, mock_reader_cls, blank_image):
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []
        mock_reader_cls.return_value = mock_reader

        extractor = OCRExtractor()
        assert extractor.extract(blank_image) == []

    @patch("pipeline.ocr_extractor.easyocr.Reader")
    def test_extract_returns_list(self, mock_reader_cls, blank_image):
        mock_reader_cls.return_value.readtext.return_value = []
        extractor = OCRExtractor()
        result = extractor.extract(blank_image)
        assert isinstance(result, list)
