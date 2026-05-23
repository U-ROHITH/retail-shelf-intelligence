"""
Visualisation utilities: annotated shelf images with bounding boxes, brand
labels, shelf-row overlays, and a colour-coded legend.
"""

from __future__ import annotations

from typing import Dict, List, Set

import cv2
import numpy as np

from config import BRAND_COLORS
from pipeline.detector import Detection

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.42
_LINE_THICKNESS = 2
_LEGEND_BOX = 13   # colour swatch side length in pixels
_LEGEND_PAD = 5    # padding between legend rows


def _draw_row_overlays(
    image: np.ndarray,
    detections: List[Detection],
    row_map: Dict[int, int],
) -> None:
    """Draw semi-transparent horizontal bands for each detected shelf row."""
    if not row_map:
        return

    row_y_ranges: Dict[int, list] = {}
    for idx, row_id in row_map.items():
        y1, y2 = detections[idx].bbox[1], detections[idx].bbox[3]
        row_y_ranges.setdefault(row_id, []).extend([y1, y2])

    overlay = image.copy()
    for row_id, ys in row_y_ranges.items():
        band_y1, band_y2 = min(ys), max(ys)
        shade = 40 if row_id % 2 == 0 else 0
        cv2.rectangle(
            overlay,
            (0, band_y1), (image.shape[1], band_y2),
            (shade, shade, shade),
            -1,
        )
    cv2.addWeighted(overlay, 0.12, image, 0.88, 0, image)


def _draw_detections(
    image: np.ndarray,
    detections: List[Detection],
    brands: List[str],
    row_map: Dict[int, int],
) -> None:
    for idx, (det, brand) in enumerate(zip(detections, brands)):
        color = BRAND_COLORS.get(brand, (140, 140, 140))
        x1, y1, x2, y2 = det.bbox
        row = row_map.get(idx, -1)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, _LINE_THICKNESS)

        label = f"{brand} [R{row}]"
        (tw, th), baseline = cv2.getTextSize(label, _FONT, _FONT_SCALE, 1)
        label_y1 = max(y1 - th - baseline - 4, 0)
        label_y2 = label_y1 + th + baseline + 4

        cv2.rectangle(image, (x1, label_y1), (x1 + tw + 6, label_y2), color, -1)
        cv2.putText(
            image, label,
            (x1 + 3, label_y2 - baseline - 2),
            _FONT, _FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA,
        )


def _draw_legend(image: np.ndarray, active_brands: Set[str]) -> None:
    """Draw a legend showing only the brands that appear in this image."""
    # Keep insertion order; always put Other last
    ordered = [b for b in BRAND_COLORS if b in active_brands and b != "Other"]
    if "Other" in active_brands:
        ordered.append("Other")

    if not ordered:
        return

    max_label_w = max(
        cv2.getTextSize(b, _FONT, _FONT_SCALE, 1)[0][0] for b in ordered
    )
    legend_w = _LEGEND_BOX + _LEGEND_PAD * 3 + max_label_w
    legend_h = len(ordered) * (_LEGEND_BOX + _LEGEND_PAD) + _LEGEND_PAD

    lx, ly = 8, 8
    # Dark background panel
    cv2.rectangle(
        image,
        (lx - 4, ly - 4),
        (lx + legend_w + 4, ly + legend_h + 4),
        (20, 20, 20), -1,
    )

    for i, brand in enumerate(ordered):
        color = BRAND_COLORS.get(brand, (140, 140, 140))
        y = ly + i * (_LEGEND_BOX + _LEGEND_PAD) + _LEGEND_PAD
        cv2.rectangle(image, (lx, y), (lx + _LEGEND_BOX, y + _LEGEND_BOX), color, -1)
        cv2.putText(
            image, brand,
            (lx + _LEGEND_BOX + _LEGEND_PAD, y + _LEGEND_BOX - 2),
            _FONT, _FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA,
        )


def annotate_image(
    image: np.ndarray,
    detections: List[Detection],
    brands: List[str],
    row_map: Dict[int, int],
) -> np.ndarray:
    """
    Return *image* (RGB) annotated with:
    - Semi-transparent shelf-row bands
    - Colour-coded bounding boxes labelled with brand and row index
    - A legend showing only brands detected in this image
    """
    _draw_row_overlays(image, detections, row_map)
    _draw_detections(image, detections, brands, row_map)
    _draw_legend(image, set(brands))
    return image
