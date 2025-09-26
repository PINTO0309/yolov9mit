# 21_make_weights_sizeaware.py

"""
MED-skew: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 27, 28, 29, 30]
LG -skew: [0, 1, 2, 3, 4, 5, 6]
Saved ../data/wholebody34/train_weights.npy  mean=1.000 min=0.421 max=2.525
"""

import os
from pathlib import Path
import numpy as np

# ========= Setting =========
ROOT = Path("../data/wholebody34")  # ★ root
SPLIT = "train"
TRAIN_TXT = ROOT / f"{SPLIT}.txt"
OUT_NPY   = ROOT / f"{SPLIT}_weights.npy"

BODY_IDS = {0, 5, 6}    # body type
IMG_SIZE_REF = 640      # Match with learning imgsz

SMALL_PX  = 32          # COCO compliant
MEDIUM_PX = 96
MED_SKEW_THR  = 0.30    # Medium bias automatic threshold
LARGE_SKEW_THR = 0.30   # Automatic threshold for large bias

# ---- Manual set (A: Example of balanced setting) ----
MANUAL_MEDIUM_SKEW_INCLUDE = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16,   26, 27, 28, 29, 30}    # head, front, right-front, right-side, right-back, back, left-back, left-side, left-front, face, hand, hand_left, hand_right, abdomen, hip_joint
MANUAL_LARGE_SKEW_INCLUDE  = {0, 1, 2, 3, 4, 5, 6} # body, adult, child, male, female, wheelchair, crutches
MANUAL_MEDIUM_SKEW_EXCLUDE = {0, 1, 2, 3, 4} # Exclude "People/Body" from Medium (to prevent duplicate boosts)
MANUAL_LARGE_SKEW_EXCLUDE  = set()

# ---- Weight Parameter ----
EPS = 1e-3
BETA_BODY  = 1.5
BETA_MACRO = 1.0
BOOST_MED  = 0.20
BOOST_LG   = 0.20
W_MIN, W_MAX = 0.5, 3.0
# ========================

def img_to_label(img_path: Path) -> Path:
    parts = list(img_path.parts)
    try:
        i = parts.index("images")
        parts[i] = "labels"
    except ValueError:
        raise RuntimeError(f"A path that does not contain 'images': {img_path}")
    return Path(*parts).with_suffix(".txt")

def size_bucket(area_norm: float):
    thr_s = (SMALL_PX / IMG_SIZE_REF) ** 2
    thr_m = (MEDIUM_PX / IMG_SIZE_REF) ** 2
    if area_norm < thr_s:
        return 0  # small
    elif area_norm < thr_m:
        return 1  # medium
    else:
        return 2  # large

def parse_label_file(label_path: Path):
    items = []
    if not label_path.exists():
        return items
    with open(label_path, "r") as f:
        for line in f:
            s = line.strip().split()
            if len(s) < 5:
                continue
            try:
                c = int(float(s[0])); w = float(s[3]); h = float(s[4])
            except:
                continue
            if w <= 0 or h <= 0:
                continue
            a = w * h
            b = size_bucket(a)
            items.append((c, b))
    return items

def scan(train_txt: Path):
    imgs = [Path(l.strip()) for l in open(train_txt) if l.strip()]
    img_infos = []
    stats = {}  # class -> [S,M,L]
    for img in imgs:
        lab = img_to_label(img)
        items = parse_label_file(lab)
        total = len(items)
        body_cnt  = sum(1 for (c, _) in items if c in BODY_IDS)
        macro_cnt = sum(1 for (_, b) in items if b in (1, 2))
        present   = {c for (c, _) in items}

        img_infos.append({
            "total": total,
            "body_cnt": body_cnt,
            "macro_cnt": macro_cnt,
            "present": present,
        })

        for c, b in items:
            stats.setdefault(c, [0,0,0])[b] += 1
    return imgs, img_infos, stats

def decide_skew(stats):
    med_skew, lg_skew = set(), set()
    for c, (s, m, l) in stats.items():
        tot = s + m + l
        if tot == 0:
            continue
        if m / tot >= MED_SKEW_THR:
            med_skew.add(c)
        if l / tot >= LARGE_SKEW_THR:
            lg_skew.add(c)
    # manual override
    med_skew |= MANUAL_MEDIUM_SKEW_INCLUDE
    lg_skew  |= MANUAL_LARGE_SKEW_INCLUDE
    med_skew -= MANUAL_MEDIUM_SKEW_EXCLUDE
    lg_skew  -= MANUAL_LARGE_SKEW_EXCLUDE
    return med_skew, lg_skew

def compute_weights(imgs, infos, med_skew, lg_skew, out_npy: Path):
    ratios_body  = []
    ratios_macro = []
    has_med = []
    has_lg  = []
    for info in infos:
        t = info["total"]
        rb = (info["body_cnt"]  / t) if t > 0 else 0.0
        rm = (info["macro_cnt"] / t) if t > 0 else 0.0
        ratios_body.append(rb)
        ratios_macro.append(rm)
        present = info["present"]
        has_med.append(1 if any(c in med_skew for c in present) else 0)
        has_lg.append(1 if any(c in lg_skew  for c in present) else 0)

    ratios_body  = np.asarray(ratios_body,  dtype=np.float64)
    ratios_macro = np.asarray(ratios_macro, dtype=np.float64)
    has_med = np.asarray(has_med, dtype=np.float64)
    has_lg  = np.asarray(has_lg,  dtype=np.float64)

    mean_body  = max(ratios_body [ratios_body  > 0].mean() if (ratios_body  > 0).any() else 0.0, EPS)
    mean_macro = max(ratios_macro[ratios_macro > 0].mean() if (ratios_macro > 0).any() else 0.0, EPS)

    w = ((ratios_body  + EPS) / (mean_body  + EPS)) ** BETA_BODY
    w*= ((ratios_macro + EPS) / (mean_macro + EPS)) ** BETA_MACRO
    w*= (1.0 + BOOST_MED * has_med)
    w*= (1.0 + BOOST_LG  * has_lg)

    w = np.clip(w, W_MIN, W_MAX)
    w = w / w.mean()
    np.save(out_npy, w)
    print(f"Saved {out_npy}  mean={w.mean():.3f} min={w.min():.3f} max={w.max():.3f}")
    return w

def main():
    imgs, infos, stats = scan(TRAIN_TXT)
    med_skew, lg_skew = decide_skew(stats)

    # Simple dump
    print(f"MED-skew: {sorted(med_skew)}")
    print(f"LG -skew: {sorted(lg_skew)}")

    compute_weights(imgs, infos, med_skew, lg_skew, OUT_NPY)

if __name__ == "__main__":
    main()
