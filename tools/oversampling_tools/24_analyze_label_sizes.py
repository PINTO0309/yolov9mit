#!/usr/bin/env python3
"""
Aggregate YOLO label boxes into small/medium/large size buckets.

python 24_analyze_label_sizes.py datasetA/labels/train
Analyzed datasetA/labels/train -> 908,135 boxes
small  592,085 (65.2%)
medium 212,507 (23.4%)
large  103,543 (11.4%)
total 908,135 (100.0%)

python 24_analyze_label_sizes.py datasetB/labels/train
Analyzed datasetB/labels/train -> 903,554 boxes
small  587,872 (65.1%)
medium 213,457 (23.6%)
large  102,225 (11.3%)
total 903,554 (100.0%)
"""

import argparse
from pathlib import Path
from typing import Iterable

SIZE_NAMES = ("small", "medium", "large")


def iter_label_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.txt"):
        if path.is_file():
            yield path


def bucket(area: float, thr_small: float, thr_medium: float) -> int:
    if area < thr_small:
        return 0
    if area < thr_medium:
        return 1
    return 2


def analyze(root: Path, img_size: int, small_px: int, medium_px: int):
    thr_small = (small_px / img_size) ** 2
    thr_medium = (medium_px / img_size) ** 2

    counts = [0, 0, 0]
    bad_lines = 0

    for label_path in iter_label_files(root):
        try:
            lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue

        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                bad_lines += 1
                continue
            try:
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                bad_lines += 1
                continue
            if w <= 0 or h <= 0:
                bad_lines += 1
                continue
            counts[bucket(w * h, thr_small, thr_medium)] += 1

    total = sum(counts)
    return counts, total, bad_lines


def fmt_count(name: str, count: int, total: int) -> str:
    pct = (count / total * 100) if total else 0.0
    return f"{name:<6} {count:,} ({pct:.1f}%)"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Path to YOLO label directory (e.g. datasetA/labels/train)")
    parser.add_argument("--img-size", type=int, default=640, help="Image size used when normalising boxes")
    parser.add_argument("--small-px", type=int, default=32, help="Pixel threshold between small and medium")
    parser.add_argument("--medium-px", type=int, default=96, help="Pixel threshold between medium and large")
    args = parser.parse_args()

    counts, total, bad_lines = analyze(args.root, args.img_size, args.small_px, args.medium_px)

    if total == 0:
        print("No boxes found.")
        return

    print(f"Analyzed {args.root} -> {total:,} boxes")
    for name, count in zip(SIZE_NAMES, counts):
        print(fmt_count(name, count, total))
    print(f"total {total:,} (100.0%)")
    if bad_lines:
        print(f"Skipped {bad_lines} invalid entries")


if __name__ == "__main__":
    main()
