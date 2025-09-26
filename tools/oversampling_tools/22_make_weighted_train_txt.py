# 22_make_weighted_train_txt.py

"""
original 9780 -> weighted 12910  (dataset1/train_weighted.txt)
original 9780 -> weighted 12278  (dataset2/train_weighted.txt)
"""

import numpy as np
from pathlib import Path

def main():
    root = Path("../data/wholebody34")
    split = "train"
    train_txt = root / f"{split}.txt"
    weights_npy = root / f"{split}_weights.npy"
    out_txt = root / f"{split}_weighted.txt"

    imgs = [l.strip() for l in open(train_txt) if l.strip()]
    w = np.load(weights_npy)
    assert len(imgs) == len(w), "Weights and number of images do not match"

    # Normalized to mean 1. Set w to integer number of copies (minimum 1).
    scale = len(imgs) / w.sum()
    reps = np.maximum(1, np.round(w * scale).astype(int))

    lines = []
    for p, k in zip(imgs, reps):
        lines.extend([p] * int(k))
    np.random.shuffle(lines)
    out_txt.write_text("\n".join(lines))
    print(f"original {len(imgs)} -> weighted {len(lines)}  ({out_txt})")

if __name__ == "__main__":
    main()
