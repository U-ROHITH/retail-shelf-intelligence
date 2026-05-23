"""
CLI entry point for the Retail Shelf Intelligence Pipeline.

Usage examples
--------------
# Analyse a single image
python main.py "images/img_1 (1).jpg"

# Analyse all three provided test images
python main.py "images/img_1 (1).jpg" "images/img_2 (1).jpg" "images/img_3 (1).jpg"

# Write outputs to a custom directory
python main.py images/*.jpg --output-dir results/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from config import OUTPUT_DIR
from pipeline.analyzer import ShelfAnalyzer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retail Shelf Intelligence Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="One or more shelf image paths to analyse",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory where annotated images and JSON results are saved "
             f"(default: {OUTPUT_DIR})",
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = ShelfAnalyzer()

    for image_path in args.images:
        print(f"\nProcessing: {image_path}")

        try:
            result, annotated = analyzer.analyze(image_path)
        except (ValueError, FileNotFoundError) as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            continue

        # Pretty-print JSON to console
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Save annotated image (BGR for OpenCV)
        stem = Path(image_path).stem
        img_out = output_dir / f"{stem}_annotated.jpg"
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(img_out), annotated_bgr)

        # Save JSON result
        json_out = output_dir / f"{stem}_result.json"
        json_out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print(f"  Annotated image -> {img_out}")
        print(f"  JSON result     -> {json_out}")


if __name__ == "__main__":
    run()
