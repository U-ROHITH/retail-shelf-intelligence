"""Integration-style unit tests for pipeline/analyzer.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pipeline.analyzer import ShelfAnalyzer
from pipeline.detector import Detection


@pytest.fixture
def sample_image() -> np.ndarray:
    return np.random.randint(0, 200, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_detection() -> Detection:
    return Detection(bbox=(50, 50, 200, 250), confidence=0.85, class_id=39, class_name="bottle")


class TestShelfAnalyzerAnalyze:
    @patch("pipeline.analyzer.annotate_image")
    @patch("pipeline.analyzer.ShelfSegmenter")
    @patch("pipeline.analyzer.OCRExtractor")
    @patch("pipeline.analyzer.BrandClassifier")
    @patch("pipeline.analyzer.ShelfDetector")
    @patch("pipeline.analyzer.cv2.cvtColor")
    @patch("pipeline.analyzer.cv2.imread")
    def test_returns_correct_result_structure(
        self,
        mock_imread,
        mock_cvtcolor,
        mock_det_cls,
        mock_clf_cls,
        mock_ocr_cls,
        mock_seg_cls,
        mock_annotate,
        sample_image,
        sample_detection,
        tmp_path,
    ):
        mock_imread.return_value = sample_image
        mock_cvtcolor.return_value = sample_image

        mock_det_cls.return_value.detect.return_value = [sample_detection]
        mock_clf_cls.return_value.classify_batch.return_value = {0: "Coca-Cola"}
        mock_ocr_cls.return_value.extract.return_value = ["₹125", "₹50"]
        mock_seg_cls.return_value.segment_rows.return_value = {0: 0}
        mock_seg_cls.return_value.shelf_share.return_value = {"Coca-Cola": 1.0}
        mock_annotate.return_value = sample_image

        img_file = tmp_path / "shelf.jpg"
        img_file.write_bytes(b"dummy")

        analyzer = ShelfAnalyzer()
        result, annotated = analyzer.analyze(str(img_file))

        assert result["image_name"] == "shelf.jpg"
        assert result["total_products"] == 1
        assert result["brands"] == {"Coca-Cola": 1}
        assert "₹125" in result["ocr_labels"]
        assert "shelf_share" in result
        assert isinstance(annotated, np.ndarray)

    @patch("pipeline.analyzer.cv2.imread")
    def test_raises_value_error_for_unreadable_image(self, mock_imread, tmp_path):
        mock_imread.return_value = None
        img_file = tmp_path / "bad.jpg"
        img_file.write_bytes(b"")

        analyzer = ShelfAnalyzer.__new__(ShelfAnalyzer)
        with pytest.raises(ValueError, match="Cannot read image"):
            analyzer.analyze(str(img_file))

    @patch("pipeline.analyzer.annotate_image")
    @patch("pipeline.analyzer.ShelfSegmenter")
    @patch("pipeline.analyzer.OCRExtractor")
    @patch("pipeline.analyzer.BrandClassifier")
    @patch("pipeline.analyzer.ShelfDetector")
    @patch("pipeline.analyzer.cv2.cvtColor")
    @patch("pipeline.analyzer.cv2.imread")
    def test_zero_detections_handled_gracefully(
        self,
        mock_imread,
        mock_cvtcolor,
        mock_det_cls,
        mock_clf_cls,
        mock_ocr_cls,
        mock_seg_cls,
        mock_annotate,
        sample_image,
        tmp_path,
    ):
        mock_imread.return_value = sample_image
        mock_cvtcolor.return_value = sample_image

        mock_det_cls.return_value.detect.return_value = []
        mock_clf_cls.return_value.classify_batch.return_value = {}
        mock_ocr_cls.return_value.extract.return_value = []
        mock_seg_cls.return_value.segment_rows.return_value = {}
        mock_seg_cls.return_value.shelf_share.return_value = {}
        mock_annotate.return_value = sample_image

        img_file = tmp_path / "empty_shelf.jpg"
        img_file.write_bytes(b"dummy")

        analyzer = ShelfAnalyzer()
        result, _ = analyzer.analyze(str(img_file))

        assert result["total_products"] == 0
        assert result["brands"] == {}
        assert result["ocr_labels"] == []

    @patch("pipeline.analyzer.annotate_image")
    @patch("pipeline.analyzer.ShelfSegmenter")
    @patch("pipeline.analyzer.OCRExtractor")
    @patch("pipeline.analyzer.BrandClassifier")
    @patch("pipeline.analyzer.ShelfDetector")
    @patch("pipeline.analyzer.cv2.cvtColor")
    @patch("pipeline.analyzer.cv2.imread")
    def test_brand_counts_aggregated_correctly(
        self,
        mock_imread,
        mock_cvtcolor,
        mock_det_cls,
        mock_clf_cls,
        mock_ocr_cls,
        mock_seg_cls,
        mock_annotate,
        sample_image,
        tmp_path,
    ):
        mock_imread.return_value = sample_image
        mock_cvtcolor.return_value = sample_image

        dets = [
            Detection(bbox=(0, 0, 80, 80), confidence=0.8, class_id=39, class_name="b"),
            Detection(bbox=(100, 0, 180, 80), confidence=0.8, class_id=39, class_name="b"),
            Detection(bbox=(200, 0, 280, 80), confidence=0.8, class_id=39, class_name="b"),
        ]
        mock_det_cls.return_value.detect.return_value = dets
        mock_clf_cls.return_value.classify_batch.return_value = {
            0: "Coca-Cola", 1: "Pepsi", 2: "Coca-Cola"
        }
        mock_ocr_cls.return_value.extract.return_value = []
        mock_seg_cls.return_value.segment_rows.return_value = {0: 0, 1: 0, 2: 0}
        mock_seg_cls.return_value.shelf_share.return_value = {
            "Coca-Cola": 0.67, "Pepsi": 0.33
        }
        mock_annotate.return_value = sample_image

        img_file = tmp_path / "multi.jpg"
        img_file.write_bytes(b"dummy")

        analyzer = ShelfAnalyzer()
        result, _ = analyzer.analyze(str(img_file))

        assert result["brands"]["Coca-Cola"] == 2
        assert result["brands"]["Pepsi"] == 1
        assert result["total_products"] == 3
