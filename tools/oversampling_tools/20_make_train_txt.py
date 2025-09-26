# 20_make_train_txt.py

import os
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".JPG", ".png", ".PNG"}

def img_to_label(img_path: Path) -> Path:
    # .../images/train/xxx.jpg -> .../labels/train/xxx.txt
    parts = list(img_path.parts)
    try:
        i = parts.index("images")
        parts[i] = "labels"
    except ValueError:
        raise RuntimeError(f"A path that does not contain 'images': {img_path}")
    return Path(*parts).with_suffix(".txt")

def main():
    root = Path("../data/wholebody34")  # ★ Change this to suit your route
    split = "train"                     # You can create val/test in the same way.
    img_dir = root / "images" / split
    out_txt = root / f"{split}.txt"

    lines = []
    for p in img_dir.rglob("*"):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        lbl = img_to_label(p)
        if lbl.exists():
            # If you want to skip empty labels, uncomment the following:
            # if lbl.stat().st_size == 0:
            #     continue
            lines.append(str(p.resolve()))
        else:
            print(f"[warn] label not found for {p}")

    out_txt.write_text("\n".join(lines))
    print(f"wrote {len(lines)} lines -> {out_txt}")

if __name__ == "__main__":
    main()
