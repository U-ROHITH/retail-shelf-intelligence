"""Unit tests for pipeline/classifier.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from pipeline.classifier import BrandClassifier, MIN_CROP_PX
from pipeline.detector import Detection


@pytest.fixture
def rgb_image() -> np.ndarray:
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def valid_detection() -> Detection:
    return Detection(bbox=(50, 50, 200, 250), confidence=0.85, class_id=39, class_name="bottle")


def _make_mock_clip(winning_idx: int, num_brands: int = 7):
    """Return (mock_processor_cls, mock_model_cls) with the given brand winning."""
    mock_processor = MagicMock()
    mock_processor.return_value = {
        "input_ids": torch.zeros(num_brands, 10, dtype=torch.long),
        "pixel_values": torch.zeros(1, 3, 224, 224),
        "attention_mask": torch.ones(num_brands, 10, dtype=torch.long),
    }

    logits = torch.zeros(1, num_brands)
    logits[0, winning_idx] = 10.0
    mock_output = MagicMock()
    mock_output.logits_per_image = logits

    mock_model = MagicMock()
    mock_model.return_value = mock_output

    return mock_processor, mock_model


class TestBrandClassifier:
    @patch("pipeline.classifier.CLIPModel.from_pretrained")
    @patch("pipeline.classifier.CLIPProcessor.from_pretrained")
    def test_classify_returns_coca_cola(
        self, mock_proc_cls, mock_model_cls, rgb_image, valid_detection
    ):
        proc, model = _make_mock_clip(0)  # index 0 = Coca-Cola
        mock_proc_cls.return_value = proc
        mock_model_cls.return_value = model

        clf = BrandClassifier()
        assert clf.classify(rgb_image, valid_detection) == "Coca-Cola"

    @patch("pipeline.classifier.CLIPModel.from_pretrained")
    @patch("pipeline.classifier.CLIPProcessor.from_pretrained")
    def test_classify_returns_other_for_tiny_crop(
        self, mock_proc_cls, mock_model_cls, rgb_image
    ):
        mock_proc_cls.return_value = MagicMock()
        mock_model_cls.return_value = MagicMock()

        tiny = Detection(
            bbox=(0, 0, MIN_CROP_PX - 1, MIN_CROP_PX - 1),
            confidence=0.8,
            class_id=39,
            class_name="bottle",
        )
        clf = BrandClassifier()
        assert clf.classify(rgb_image, tiny) == "Other"

    @patch("pipeline.classifier.CLIPModel.from_pretrained")
    @patch("pipeline.classifier.CLIPProcessor.from_pretrained")
    def test_classify_batch_keys_match_detection_indices(
        self, mock_proc_cls, mock_model_cls, rgb_image
    ):
        proc, model = _make_mock_clip(1)  # index 1 = Pepsi
        mock_proc_cls.return_value = proc
        mock_model_cls.return_value = model

        detections = [
            Detection(bbox=(10, 10, 100, 100), confidence=0.8, class_id=39, class_name="bottle"),
            Detection(bbox=(200, 10, 300, 100), confidence=0.7, class_id=39, class_name="bottle"),
            Detection(bbox=(400, 10, 500, 100), confidence=0.75, class_id=39, class_name="bottle"),
        ]
        clf = BrandClassifier()
        result = clf.classify_batch(rgb_image, detections)

        assert set(result.keys()) == {0, 1, 2}
        for brand in result.values():
            assert brand in ["Coca-Cola", "Pepsi", "Lay's", "Doritos", "Amul", "Britannia", "Other"]

    @patch("pipeline.classifier.CLIPModel.from_pretrained")
    @patch("pipeline.classifier.CLIPProcessor.from_pretrained")
    def test_classify_batch_empty_detections(
        self, mock_proc_cls, mock_model_cls, rgb_image
    ):
        mock_proc_cls.return_value = MagicMock()
        mock_model_cls.return_value = MagicMock()

        clf = BrandClassifier()
        result = clf.classify_batch(rgb_image, [])
        assert result == {}
