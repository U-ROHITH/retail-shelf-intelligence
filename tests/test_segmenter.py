"""Unit tests for pipeline/segmenter.py."""

from __future__ import annotations

import pytest

from pipeline.detector import Detection
from pipeline.segmenter import ShelfSegmenter


def _det(y_center: int, height: int = 60, width: int = 80, x: int = 0) -> Detection:
    y1 = y_center - height // 2
    y2 = y_center + height // 2
    return Detection(
        bbox=(x, y1, x + width, y2),
        confidence=0.8,
        class_id=39,
        class_name="bottle",
    )


class TestSegmentRows:
    def test_empty_input(self):
        seg = ShelfSegmenter()
        assert seg.segment_rows([]) == {}

    def test_single_detection_is_row_zero(self):
        seg = ShelfSegmenter()
        rows = seg.segment_rows([_det(100)])
        assert rows == {0: 0}

    def test_same_row_cluster(self):
        seg = ShelfSegmenter(eps=60)
        dets = [_det(100), _det(110), _det(120)]
        rows = seg.segment_rows(dets)
        assert len(set(rows.values())) == 1

    def test_two_distinct_rows(self):
        seg = ShelfSegmenter(eps=60)
        dets = [_det(100), _det(110), _det(400), _det(410)]
        rows = seg.segment_rows(dets)
        assert len(set(rows.values())) == 2

    def test_top_row_has_id_zero(self):
        """The topmost shelf row must receive row ID 0."""
        seg = ShelfSegmenter(eps=60)
        # index 0 → y=500 (bottom), index 1 → y=100 (top)
        dets = [_det(500), _det(100)]
        rows = seg.segment_rows(dets)
        # index 1 (y=100) should be row 0; index 0 (y=500) should be row 1
        assert rows[1] == 0
        assert rows[0] == 1

    def test_four_row_shelf(self):
        seg = ShelfSegmenter(eps=60)
        dets = [
            _det(50), _det(60),    # row 0
            _det(200), _det(210),  # row 1
            _det(350), _det(360),  # row 2
            _det(500), _det(510),  # row 3
        ]
        rows = seg.segment_rows(dets)
        assert len(set(rows.values())) == 4


class TestShelfShare:
    def test_equal_share(self):
        seg = ShelfSegmenter()
        dets = [
            Detection(bbox=(0, 0, 100, 100), confidence=0.8, class_id=39, class_name="bottle"),
            Detection(bbox=(0, 0, 100, 100), confidence=0.8, class_id=39, class_name="bottle"),
        ]
        share = seg.shelf_share(dets, ["Coca-Cola", "Pepsi"])
        assert share["Coca-Cola"] == pytest.approx(0.5)
        assert share["Pepsi"] == pytest.approx(0.5)

    def test_unequal_share(self):
        seg = ShelfSegmenter()
        dets = [
            Detection(bbox=(0, 0, 200, 100), confidence=0.8, class_id=39, class_name="bottle"),  # area 20000
            Detection(bbox=(0, 0, 100, 100), confidence=0.8, class_id=39, class_name="bottle"),  # area 10000
        ]
        share = seg.shelf_share(dets, ["Coca-Cola", "Pepsi"])
        assert share["Coca-Cola"] == pytest.approx(20000 / 30000, rel=1e-3)
        assert share["Pepsi"] == pytest.approx(10000 / 30000, rel=1e-3)

    def test_empty_returns_empty(self):
        seg = ShelfSegmenter()
        assert seg.shelf_share([], []) == {}

    def test_shares_sum_to_one(self):
        seg = ShelfSegmenter()
        dets = [
            Detection(bbox=(0, 0, 80, 80), confidence=0.8, class_id=39, class_name="b"),
            Detection(bbox=(0, 0, 60, 60), confidence=0.8, class_id=39, class_name="b"),
            Detection(bbox=(0, 0, 40, 40), confidence=0.8, class_id=39, class_name="b"),
        ]
        brands = ["Coca-Cola", "Pepsi", "Other"]
        share = seg.shelf_share(dets, brands)
        assert sum(share.values()) == pytest.approx(1.0, rel=1e-3)
