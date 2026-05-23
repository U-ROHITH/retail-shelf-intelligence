"""Unit tests for pipeline/detector.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pipeline.detector import Detection, ShelfDetector, _grid_proposals, _is_valid_bbox


# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------

class TestDetection:
    def test_area(self):
        d = Detection(bbox=(0, 0, 100, 50), confidence=0.9, class_id=39, class_name="bottle")
        assert d.area == 5000

    def test_center_y(self):
        d = Detection(bbox=(0, 0, 100, 200), confidence=0.9, class_id=39, class_name="bottle")
        assert d.center_y == 100

    def test_center_x(self):
        d = Detection(bbox=(0, 0, 100, 200), confidence=0.9, class_id=39, class_name="bottle")
        assert d.center_x == 50

    def test_width_height(self):
        d = Detection(bbox=(10, 20, 110, 70), confidence=0.8, class_id=39, class_name="bottle")
        assert d.width == 100
        assert d.height == 50


# ---------------------------------------------------------------------------
# _is_valid_bbox
# ---------------------------------------------------------------------------

class TestIsValidBbox:
    def test_valid_product_bbox(self):
        assert _is_valid_bbox(100, 100, 200, 300, 640, 640) is True

    def test_too_small(self):
        assert _is_valid_bbox(0, 0, 2, 2, 640, 640) is False

    def test_too_large_area(self):
        assert _is_valid_bbox(0, 0, 640, 640, 640, 640) is False

    def test_too_wide(self):
        # 70% of image width
        assert _is_valid_bbox(0, 100, 448, 300, 640, 640) is False

    def test_too_tall(self):
        # height 450 / img_h 640 = 0.703 > MAX_BBOX_HEIGHT_RATIO (0.65)
        assert _is_valid_bbox(100, 0, 300, 450, 640, 640) is False

    def test_zero_size(self):
        assert _is_valid_bbox(50, 50, 50, 50, 640, 640) is False


# ---------------------------------------------------------------------------
# _grid_proposals
# ---------------------------------------------------------------------------

class TestGridProposals:
    def test_returns_list(self):
        image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        proposals = _grid_proposals(image)
        assert isinstance(proposals, list)

    def test_empty_image_returns_no_proposals(self):
        # uniform image → variance == 0 → all cells filtered
        image = np.full((480, 640, 3), 128, dtype=np.uint8)
        proposals = _grid_proposals(image)
        assert proposals == []

    def test_noisy_image_returns_proposals(self):
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        proposals = _grid_proposals(image)
        assert len(proposals) > 0

    def test_proposals_are_grid_class(self):
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        proposals = _grid_proposals(image)
        for p in proposals:
            assert p.class_name == "grid_proposal"
            assert p.class_id == -1


# ---------------------------------------------------------------------------
# ShelfDetector
# ---------------------------------------------------------------------------

@pytest.fixture
def yolo_result_bottle():
    """Mock a YOLO result with a single valid bottle detection."""
    box = MagicMock()
    box.xyxy = [MagicMock()]
    box.xyxy[0].tolist.return_value = [100.0, 100.0, 200.0, 300.0]
    box.conf = [0.85]
    box.cls = [39]

    result = MagicMock()
    result.boxes = [box]
    result.names = {39: "bottle"}
    return [result]


class TestShelfDetector:
    @patch("pipeline.detector.YOLO")
    def test_returns_detections_from_yolo(self, mock_yolo_cls, yolo_result_bottle):
        mock_yolo_cls.return_value.return_value = yolo_result_bottle * 10  # ≥ threshold
        detector = ShelfDetector()
        # Build a mock that returns enough results
        mock_yolo_cls.return_value.return_value = yolo_result_bottle * 10
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        dets = detector.detect(image)
        # At least one detection with valid class name
        assert any(d.class_name == "bottle" for d in dets)

    @patch("pipeline.detector.YOLO")
    def test_falls_back_to_grid_when_few_yolo_results(self, mock_yolo_cls):
        """When YOLO returns 0 valid detections, grid proposals should be used."""
        mock_yolo_cls.return_value.return_value = []
        detector = ShelfDetector()
        # Noisy image → grid proposals should kick in
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dets = detector.detect(image)
        assert all(d.class_name == "grid_proposal" for d in dets)

    @patch("pipeline.detector.YOLO")
    def test_filters_bbox_that_is_too_large(self, mock_yolo_cls):
        box = MagicMock()
        box.xyxy = [MagicMock()]
        box.xyxy[0].tolist.return_value = [0.0, 0.0, 640.0, 640.0]  # entire image
        box.conf = [0.90]
        box.cls = [39]
        result = MagicMock()
        result.boxes = [box]
        result.names = {39: "bottle"}
        mock_yolo_cls.return_value.return_value = [result]

        detector = ShelfDetector()
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        dets = detector.detect(image)
        # The only YOLO detection was filtered; fallback should have fired
        assert all(d.class_name != "bottle" for d in dets)
