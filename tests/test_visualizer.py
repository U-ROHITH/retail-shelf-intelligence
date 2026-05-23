"""Unit tests for utils/visualizer.py."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.detector import Detection
from utils.visualizer import annotate_image, _draw_legend, _draw_detections, _draw_row_overlays
from config import BRAND_COLORS


@pytest.fixture
def blank_rgb() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_detections() -> list[Detection]:
    return [
        Detection(bbox=(50, 50, 150, 200), confidence=0.9, class_id=39, class_name="bottle"),
        Detection(bbox=(200, 250, 300, 400), confidence=0.8, class_id=39, class_name="bottle"),
        Detection(bbox=(400, 50, 500, 200), confidence=0.7, class_id=39, class_name="bottle"),
    ]


@pytest.fixture
def sample_brands() -> list[str]:
    return ["Coca-Cola", "Pepsi", "Other"]


@pytest.fixture
def sample_row_map() -> dict[int, int]:
    return {0: 0, 1: 1, 2: 0}


class TestAnnotateImage:
    def test_returns_numpy_array(self, blank_rgb, sample_detections, sample_brands, sample_row_map):
        result = annotate_image(blank_rgb.copy(), sample_detections, sample_brands, sample_row_map)
        assert isinstance(result, np.ndarray)

    def test_output_shape_unchanged(self, blank_rgb, sample_detections, sample_brands, sample_row_map):
        result = annotate_image(blank_rgb.copy(), sample_detections, sample_brands, sample_row_map)
        assert result.shape == blank_rgb.shape

    def test_empty_detections(self, blank_rgb):
        result = annotate_image(blank_rgb.copy(), [], [], {})
        assert result.shape == blank_rgb.shape

    def test_image_is_modified(self, sample_detections, sample_brands, sample_row_map):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        original = image.copy()
        result = annotate_image(image, sample_detections, sample_brands, sample_row_map)
        # With bboxes drawn, some pixels must differ from the blank original
        assert not np.array_equal(result, original)

    def test_unknown_brand_uses_default_color(self, blank_rgb):
        dets = [Detection(bbox=(10, 10, 100, 100), confidence=0.8, class_id=-1, class_name="unknown")]
        result = annotate_image(blank_rgb.copy(), dets, ["UnknownBrand"], {0: 0})
        assert isinstance(result, np.ndarray)


class TestDrawLegend:
    def test_legend_modifies_image(self, blank_rgb):
        original = blank_rgb.copy()
        _draw_legend(blank_rgb, {"Coca-Cola", "Pepsi"})
        assert not np.array_equal(blank_rgb, original)

    def test_empty_active_brands_no_crash(self, blank_rgb):
        _draw_legend(blank_rgb, set())  # should silently return

    def test_other_always_last(self, blank_rgb):
        # Should not raise with Other in the set
        _draw_legend(blank_rgb, {"Pepsi", "Other", "Amul"})


class TestDrawDetections:
    def test_no_crash_on_empty(self, blank_rgb):
        _draw_detections(blank_rgb, [], [], {})

    def test_draws_all_brands(self, blank_rgb, sample_detections, sample_brands, sample_row_map):
        original = blank_rgb.copy()
        _draw_detections(blank_rgb, sample_detections, sample_brands, sample_row_map)
        assert not np.array_equal(blank_rgb, original)


class TestDrawRowOverlays:
    def test_no_crash_on_empty_row_map(self, blank_rgb, sample_detections):
        _draw_row_overlays(blank_rgb, sample_detections, {})

    def test_applies_overlay(self, blank_rgb, sample_detections, sample_row_map):
        image_copy = blank_rgb.copy()
        # Add some base brightness so overlay change is detectable
        image_copy[:] = 100
        original = image_copy.copy()
        _draw_row_overlays(image_copy, sample_detections, sample_row_map)
        # Overlay modifies pixel values
        assert not np.array_equal(image_copy, original)
