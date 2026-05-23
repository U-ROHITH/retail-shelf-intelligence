"""
Shelf row segmentation and share-of-shelf estimation.

Design decisions:
- DBSCAN on bounding-box center-y values groups products into shelf rows without
  needing to know the number of rows in advance (unlike K-Means).
- eps (max within-row y-distance) is configurable in config.py; 55 px works well
  for ~600–1000 px tall images with 3–5 rows.
- Row IDs are re-indexed top-to-bottom so row 0 is always the topmost shelf.
- Share-of-shelf is computed as each brand's fraction of total detected bounding-box
  area; pixel area is a reasonable proxy for facing count on dense shelves.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.cluster import DBSCAN

from config import ROW_CLUSTER_EPS, ROW_CLUSTER_MIN_SAMPLES
from pipeline.detector import Detection


class ShelfSegmenter:
    def __init__(
        self,
        eps: float = ROW_CLUSTER_EPS,
        min_samples: int = ROW_CLUSTER_MIN_SAMPLES,
    ) -> None:
        self._eps = eps
        self._min_samples = min_samples

    def segment_rows(self, detections: List[Detection]) -> Dict[int, int]:
        """
        Cluster detections into shelf rows.

        Returns a dict mapping detection index → zero-based row ID,
        where row 0 is the topmost shelf row.
        """
        if not detections:
            return {}

        centers_y = np.array([[d.center_y] for d in detections], dtype=float)
        raw_labels = DBSCAN(
            eps=self._eps, min_samples=self._min_samples
        ).fit_predict(centers_y)

        # Sort clusters by mean y so row 0 is at the top of the image
        unique_labels = sorted(
            set(raw_labels),
            key=lambda lbl: float(np.mean(centers_y[raw_labels == lbl])),
        )
        remap = {old: new for new, old in enumerate(unique_labels)}
        return {i: remap[int(raw_labels[i])] for i in range(len(detections))}

    def shelf_share(
        self,
        detections: List[Detection],
        brands: List[str],
    ) -> Dict[str, float]:
        """
        Compute each brand's share of detected bounding-box area.

        Returns {brand: fraction} where fractions sum to 1.0.
        """
        area_by_brand: Dict[str, int] = {}
        for det, brand in zip(detections, brands):
            area_by_brand[brand] = area_by_brand.get(brand, 0) + det.area

        total = sum(area_by_brand.values()) or 1
        return {b: round(a / total, 4) for b, a in sorted(area_by_brand.items())}
