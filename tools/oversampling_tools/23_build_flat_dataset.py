# 23_build_flat_dataset.py

"""
# If you want to copy anyway
python 23_build_flat_dataset.py \
--src dataset1 \
--dst datasetA \
--train-list dataset1/train_weighted.txt \
--mode copy

python 23_build_flat_dataset.py \
--src dataset2 \
--dst datasetB \
--train-list dataset2/train_weighted.txt \
--mode copy
"""

import argparse
import csv
import hashlib
import os
import shutil
from pathlib import Path
from typing import Tuple, Optional

IMG_EXTS = {".jpg", ".jpeg", ".JPG", ".png", ".bmp", ".PNG"}

def parse_args():
    ap = argparse.ArgumentParser(description="Flatten dataset using weighted train list.")
    ap.add_argument("--src", type=Path, required=True, help="source root (e.g., dataset1)")
    ap.add_argument("--dst", type=Path, required=True, help="destination root (e.g., dataset3)")
    ap.add_argument("--train-list", type=Path, required=True, help="weighted train list (e.g., dataset1/train_weighted.txt)")
    ap.add_argument("--mode", choices=["copy", "hardlink", "symlink"], default="hardlink", help="how to place files in dst (default: hardlink). Hardlink falls back to copy on failure.")
    ap.add_argument("--allow-missing-label", action="store_true", help="if set, images without labels are kept (label omitted). Default: skip those images.")
    ap.add_argument("--dry-run", action="store_true", help="print planned operations without creating files.")
    return ap.parse_args()

def ensure_dirs(dst: Path):
    (dst / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dst / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dst / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dst / "labels" / "val").mkdir(parents=True, exist_ok=True)

def find_label_from_image(img_path: Path) -> Optional[Path]:
    """
    Convert .../images/.../xxx.jpg  -> .../labels/.../xxx.txt
    Return None if mapping can't be derived.
    """
    parts = list(img_path.parts)
    if "images" in parts:
        i = parts.index("images")
        parts[i] = "labels"
        return Path(*parts).with_suffix(".txt")
    # fallback (string replace) for odd layouts
    s = str(img_path)
    if "/images/" in s:
        return Path(s.replace("/images/", "/labels/")).with_suffix(".txt")
    if "\\images\\" in s:
        return Path(s.replace("\\images\\", "\\labels\\")).with_suffix(".txt")
    return None

def rel_under_images(img_path: Path) -> Path:
    """
    Return path relative to the 'images' directory segment.
    ex) .../dataset1/images/train/dir/a.jpg -> train/dir/a.jpg
    """
    parts = list(img_path.parts)
    if "images" in parts:
        i = parts.index("images")
        return Path(*parts[i+1:])
    # If not found, fall back to filename only
    return Path(img_path.name)

def unique_stem_for(img_path: Path) -> str:
    """
    Create a stable unique stem per source image using hash of relative path under 'images'.
    """
    rel = rel_under_images(img_path).as_posix().lower()
    h = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:8]
    return f"{img_path.stem}_{h}"

def place_one(src: Path, dst: Path, mode: str, dry_run: bool):
    if dry_run:
        print(f"[DRY] {mode} -> {dst}  (from {src})")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)  # fallback
        return
    if mode == "symlink":
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src, dst)
        except Exception:
            shutil.copy2(src, dst)  # fallback

def iter_images(root_images_split: Path):
    for p in root_images_split.rglob("*"):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            yield p

def read_weighted_list(path: Path):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(Path(s))
    return lines

def build_train(src_root: Path, dst_root: Path, train_list: Path,
                mode: str, allow_missing_label: bool, dry_run: bool) -> Tuple[int, int]:
    """
    Use weighted train list; duplicates allowed.
    Returns: (num_images_written, num_labels_written)
    """
    imgs = read_weighted_list(train_list)
    count_map = {}  # per-src repetition counter
    n_img = n_lbl = 0

    idx_csv = []  # mapping for index_train.csv

    for img in imgs:
        # Allow absolute or relative paths in train_list
        img_abs = img if img.is_absolute() else (src_root / img).resolve()
        if not img_abs.exists():
            print(f"[WARN] image not found: {img}")
            continue
        if img_abs.suffix.lower() not in IMG_EXTS:
            print(f"[WARN] skip non-image: {img_abs}")
            continue

        lbl_abs = find_label_from_image(img_abs)
        has_label = lbl_abs is not None and lbl_abs.exists()

        if not has_label and not allow_missing_label:
            print(f"[WARN] label missing -> skip image: {img_abs}")
            continue

        # repetition index for this source
        k = count_map.get(img_abs, 0)
        count_map[img_abs] = k + 1

        stem = unique_stem_for(img_abs)
        if k > 0:
            stem = f"{stem}__rep{k}"
        ext = img_abs.suffix.lower()

        dst_img = dst_root / "images" / "train" / f"{stem}{ext}"
        place_one(img_abs, dst_img, mode, dry_run)
        n_img += 1

        if has_label:
            dst_lbl = dst_root / "labels" / "train" / f"{stem}.txt"
            place_one(lbl_abs, dst_lbl, mode, dry_run)
            n_lbl += 1
        else:
            dst_lbl = None

    return n_img, n_lbl

def build_val(src_root: Path, dst_root: Path, mode: str, allow_missing_label: bool, dry_run: bool) -> Tuple[int, int]:
    """
    Flatten val split by scanning src_root/images/val.
    """
    src_imgs_val = src_root / "images" / "val"
    if not src_imgs_val.exists():
        print(f"[WARN] {src_imgs_val} not found; skip val.")
        return 0, 0

    n_img = n_lbl = 0
    idx_csv = []

    for img_abs in iter_images(src_imgs_val):
        lbl_abs = find_label_from_image(img_abs)
        has_label = lbl_abs is not None and lbl_abs.exists()

        if not has_label and not allow_missing_label:
            print(f"[WARN] label missing -> skip val image: {img_abs}")
            continue

        stem = unique_stem_for(img_abs)  # There are basically no duplicates in val (just to be safe, uniqueness is the same)
        ext = img_abs.suffix.lower()

        dst_img = dst_root / "images" / "val" / f"{stem}{ext}"
        place_one(img_abs, dst_img, mode, dry_run)
        n_img += 1

        if has_label:
            dst_lbl = dst_root / "labels" / "val" / f"{stem}.txt"
            place_one(lbl_abs, dst_lbl, mode, dry_run)
            n_lbl += 1
        else:
            dst_lbl = None

    return n_img, n_lbl

def main():
    args = parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()
    ensure_dirs(dst)

    print(f"[INFO] SRC: {src}")
    print(f"[INFO] DST: {dst}")
    print(f"[INFO] MODE: {args.mode}  (hardlink falls back to copy)")
    print(f"[INFO] TRAIN LIST: {args.train_list}")
    if args.dry_run:
        print("[INFO] DRY RUN (no files will be created)")

    ti, tl = build_train(src, dst, args.train_list, args.mode, args.allow_missing_label, args.dry_run)
    vi, vl = build_val(src, dst, args.mode, args.allow_missing_label, args.dry_run)

    print(f"[DONE] train: images={ti}, labels={tl}")
    print(f"[DONE]   val: images={vi}, labels={vl}")

if __name__ == "__main__":
    main()
