"""Unit tests for pipeline/classifier.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from config import BRANDS
from pipeline.classifier import BrandClassifier, MIN_CROP_PX
from pipeline.detector import Detection


@pytest.fixture
def rgb_image() -> np.ndarray:
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def valid_detection() -> Detection:
    return Detection(bbox=(50, 50, 200, 250), confidence=0.85, class_id=39, class_name="bottle")


def _make_mock_clip(winning_idx: int):
    """
    Build (mock_processor_cls, mock_model_cls) so that CLIP always picks
    BRANDS[winning_idx].

    Uses a square identity embedding space (n × n) so every brand has a
    unique unit vector, guaranteeing no NaN after normalisation and an
    unambiguous argmax.

    Mocks the internal components used in transformers 5.x:
      text_model → pooler_output
      text_projection(pooler_output) → text features
      vision_model → pooler_output
      visual_projection(pooler_output) → image features
    """
    n = len(BRANDS)

    # Text features: identity matrix — brand i maps to standard basis vector i
    text_feats = torch.eye(n, dtype=torch.float16)

    # Image features: unit vector at the winning brand's position
    img_feats = torch.zeros(1, n, dtype=torch.float16)
    img_feats[0, winning_idx] = 1.0

    # text_model output
    text_model_out = MagicMock()
    text_model_out.pooler_output = torch.zeros(n, n, dtype=torch.float16)

    vision_model_out = MagicMock()
    vision_model_out.pooler_output = torch.zeros(1, n, dtype=torch.float16)

    mock_model = MagicMock()
    mock_model.text_model.return_value = text_model_out
    mock_model.text_projection.return_value = text_feats   # already normalised
    mock_model.vision_model.return_value = vision_model_out
    mock_model.visual_projection.return_value = img_feats  # already normalised

    mock_processor = MagicMock()
    # Return a dict with the right tensor types for any call
    mock_processor.return_value = {
        "input_ids": torch.zeros(n, 10, dtype=torch.long),
        "attention_mask": torch.ones(n, 10, dtype=torch.long),
        "pixel_values": torch.zeros(1, 3, 224, 224, dtype=torch.float16),
    }

    return mock_processor, mock_model


class TestBrandClassifier:
    @patch("pipeline.classifier.CLIPModel.from_pretrained")
    @patch("pipeline.classifier.CLIPProcessor.from_pretrained")
    def test_classify_returns_coca_cola(
        self, mock_proc_cls, mock_model_cls, rgb_image, valid_detection
    ):
        proc, model = _make_mock_clip(BRANDS.index("Coca-Cola"))
        mock_proc_cls.return_value = proc
        mock_model_cls.return_value = model

        clf = BrandClassifier()
        assert clf.classify(rgb_image, valid_detection) == "Coca-Cola"

    @patch("pipeline.classifier.CLIPModel.from_pretrained")
    @patch("pipeline.classifier.CLIPProcessor.from_pretrained")
    def test_classify_returns_pepsi(
        self, mock_proc_cls, mock_model_cls, rgb_image, valid_detection
    ):
        proc, model = _make_mock_clip(BRANDS.index("Pepsi"))
        mock_proc_cls.return_value = proc
        mock_model_cls.return_value = model

        clf = BrandClassifier()
        assert clf.classify(rgb_image, valid_detection) == "Pepsi"

    @patch("pipeline.classifier.CLIPModel.from_pretrained")
    @patch("pipeline.classifier.CLIPProcessor.from_pretrained")
    def test_classify_returns_amul(
        self, mock_proc_cls, mock_model_cls, rgb_image, valid_detection
    ):
        proc, model = _make_mock_clip(BRANDS.index("Amul"))
        mock_proc_cls.return_value = proc
        mock_model_cls.return_value = model

        clf = BrandClassifier()
        assert clf.classify(rgb_image, valid_detection) == "Amul"

    @patch("pipeline.classifier.CLIPModel.from_pretrained")
    @patch("pipeline.classifier.CLIPProcessor.from_pretrained")
    def test_classify_returns_other_for_tiny_crop(
        self, mock_proc_cls, mock_model_cls, rgb_image
    ):
        proc, model = _make_mock_clip(0)
        mock_proc_cls.return_value = proc
        mock_model_cls.return_value = model

        tiny = Detection(
            bbox=(0, 0, MIN_CROP_PX - 1, MIN_CROP_PX - 1),
            confidence=0.8,
            class_id=39,
            class_name="bottle",
        )
        clf = BrandClassifier()
        # Tiny crop skips CLIP entirely and returns "Other"
        assert clf.classify(rgb_image, tiny) == "Other"

    @patch("pipeline.classifier.CLIPModel.from_pretrained")
    @patch("pipeline.classifier.CLIPProcessor.from_pretrained")
    def test_classify_batch_keys_match_detection_indices(
        self, mock_proc_cls, mock_model_cls, rgb_image
    ):
        proc, model = _make_mock_clip(BRANDS.index("Lay's"))
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
            assert brand in BRANDS

    @patch("pipeline.classifier.CLIPModel.from_pretrained")
    @patch("pipeline.classifier.CLIPProcessor.from_pretrained")
    def test_classify_batch_empty_returns_empty_dict(
        self, mock_proc_cls, mock_model_cls, rgb_image
    ):
        proc, model = _make_mock_clip(0)
        mock_proc_cls.return_value = proc
        mock_model_cls.return_value = model

        clf = BrandClassifier()
        assert clf.classify_batch(rgb_image, []) == {}

    @patch("pipeline.classifier.CLIPModel.from_pretrained")
    @patch("pipeline.classifier.CLIPProcessor.from_pretrained")
    def test_text_features_precomputed_once(
        self, mock_proc_cls, mock_model_cls, rgb_image, valid_detection
    ):
        """text_model must be called only at init, not on each classify() call."""
        proc, model = _make_mock_clip(0)
        mock_proc_cls.return_value = proc
        mock_model_cls.return_value = model

        clf = BrandClassifier()
        init_call_count = model.text_model.call_count

        clf.classify(rgb_image, valid_detection)
        clf.classify(rgb_image, valid_detection)

        assert model.text_model.call_count == init_call_count
