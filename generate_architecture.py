"""
Generate and save the pipeline architecture diagram as architecture.png.

Usage
-----
python generate_architecture.py
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def _box(ax, x, y, w, h, label, sublabel="", color="#2563EB", text_color="white"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor="white", linewidth=1.5,
        zorder=3,
    )
    ax.add_patch(patch)
    if sublabel:
        ax.text(
            x + w / 2, y + h * 0.62, label,
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=text_color, zorder=4,
        )
        ax.text(
            x + w / 2, y + h * 0.28, sublabel,
            ha="center", va="center", fontsize=7.5, color=text_color,
            alpha=0.88, zorder=4,
        )
    else:
        ax.text(
            x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=text_color, zorder=4,
        )


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#94A3B8",
            lw=2,
        ),
        zorder=2,
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # ---- Title -------------------------------------------------------
    ax.text(
        7, 8.6, "Retail Shelf Intelligence — Pipeline Architecture",
        ha="center", va="center", fontsize=14, fontweight="bold", color="white",
    )

    # ---- Input -------------------------------------------------------
    _box(ax, 5.5, 7.3, 3, 0.85, "Input Shelf Image", "JPEG / PNG", color="#1E293B")

    # ---- Stage 1: Detection -----------------------------------------
    _box(ax, 0.4, 5.5, 3.8, 1.2,
         "1  Object Detection",
         "YOLOv8n (COCO, 80 classes)\nconf=0.20 • size filter\nGrid fallback if <5 hits",
         color="#1D4ED8")

    # ---- Stage 2: Classification ------------------------------------
    _box(ax, 4.7, 5.5, 4.6, 1.2,
         "2  Brand Classification",
         "CLIP ViT-B/32 (zero-shot)\n7 brands + Other\nNo fine-tuning required",
         color="#6D28D9")

    # ---- Stage 3: OCR -----------------------------------------------
    _box(ax, 9.8, 5.5, 3.8, 1.2,
         "3  OCR Extraction",
         "EasyOCR (English)\nPrice pattern filter\n₹/$ regex + short caps labels",
         color="#065F46")

    # ---- Stage 4: Segmentation ---------------------------------------
    _box(ax, 2.5, 3.8, 4.0, 1.2,
         "4  Shelf Row Segmentation",
         "DBSCAN on bbox center-y\neps=55 px • sorted top→bottom\nRow IDs assigned per product",
         color="#92400E")

    # ---- Stage 5: Metrics -------------------------------------------
    _box(ax, 7.5, 3.8, 4.0, 1.2,
         "5  Metrics Aggregation",
         "total_products • brand_counts\nshelf_share (bbox area %)\nocr_labels list",
         color="#7F1D1D")

    # ---- Stage 6: Visualisation -------------------------------------
    _box(ax, 1.5, 1.8, 4.0, 1.2,
         "6  Visualisation",
         "OpenCV annotation layer\nColoured bboxes by brand\nRow overlays + legend",
         color="#164E63")

    # ---- Output: Annotated image ------------------------------------
    _box(ax, 6.0, 1.5, 2.5, 0.85, "Annotated Image", "JPEG output", color="#1E293B")

    # ---- Output: JSON -----------------------------------------------
    _box(ax, 9.5, 1.5, 3.0, 0.85, "JSON Metrics", ".json output", color="#1E293B")

    # ---- Arrows ------------------------------------------------------
    # Input → Detection
    _arrow(ax, 7.0, 7.3, 2.3, 6.7)
    # Input → Classification
    _arrow(ax, 7.0, 7.3, 7.0, 6.7)
    # Input → OCR
    _arrow(ax, 7.0, 7.3, 11.7, 6.7)

    # Detection → Segmentation
    _arrow(ax, 2.3, 5.5, 4.5, 5.0)
    # Classification → Segmentation
    _arrow(ax, 7.0, 5.5, 5.5, 5.0)
    # Classification → Metrics
    _arrow(ax, 7.0, 5.5, 8.5, 5.0)
    # OCR → Metrics
    _arrow(ax, 11.7, 5.5, 9.5, 5.0)

    # Segmentation → Visualisation
    _arrow(ax, 4.5, 3.8, 3.5, 3.0)
    # Segmentation → Metrics
    _arrow(ax, 6.5, 3.8, 8.0, 5.0)

    # Metrics → JSON output
    _arrow(ax, 9.5, 3.8, 10.5, 2.35)
    # Visualisation → Annotated image
    _arrow(ax, 3.5, 1.8, 7.25, 2.35)
    # Metrics → Annotated image (combined)
    _arrow(ax, 8.5, 3.8, 7.25, 2.35)

    # ---- Legend / model notes ----------------------------------------
    legend_items = [
        mpatches.Patch(color="#1D4ED8", label="YOLOv8n — detection backbone"),
        mpatches.Patch(color="#6D28D9", label="CLIP ViT-B/32 — zero-shot classifier"),
        mpatches.Patch(color="#065F46", label="EasyOCR — scene-text extraction"),
        mpatches.Patch(color="#92400E", label="DBSCAN — unsupervised row clustering"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower left", bbox_to_anchor=(0.01, 0.01),
        framealpha=0.15, labelcolor="white",
        fontsize=8, title="Model choices", title_fontsize=8,
    )

    plt.tight_layout()
    plt.savefig("architecture.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("Saved: architecture.png")


if __name__ == "__main__":
    main()
