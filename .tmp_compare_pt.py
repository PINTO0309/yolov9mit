import sys
from pathlib import Path
import torch
from collections import OrderedDict

best_path = Path('runs/train/v9-t/lightning_logs/version_0/checkpoints/best_0001_0.0674.pt')
off_path = Path('weights/v9-t.pt')

r = {}
for name, p in [('best', best_path), ('official', off_path)]:
    obj = torch.load(p, map_location='cpu', weights_only=False)
    r[name] = obj
    print(f'[{name}] type={type(obj)} len={len(obj) if hasattr(obj, "__len__") else "-"}')
    if isinstance(obj, dict):
        ks = list(obj.keys())
        print(f'  first keys: {ks[:6]}')
        shapes = [tuple(v.shape) for v in obj.values() if hasattr(v, "shape")]
        print(f'  tensors: {len(shapes)}')

# Basic checks: both OrderedDict with flat keys
fmt_match = isinstance(r['best'], OrderedDict) and isinstance(r['official'], OrderedDict)

# Compare key sets
kb = set(r['best'].keys())
ko = set(r['official'].keys())
common = sorted(kb & ko)
only_b = sorted(kb - ko)
only_o = sorted(ko - kb)

print(f'common keys: {len(common)}  only_best: {len(only_b)}  only_official: {len(only_o)}')

# Shape compare (ignore class-dependent by heuristic: allow mismatches if any dim differs and key contains "cls" or "head" or "detect")
shape_equal = []
shape_diff = []
for k in common:
    vb = r['best'][k]
    vo = r['official'][k]
    sb = tuple(vb.shape) if hasattr(vb, 'shape') else None
    so = tuple(vo.shape) if hasattr(vo, 'shape') else None
    if sb == so:
        shape_equal.append(k)
    else:
        shape_diff.append((k, sb, so))

print(f'same-shape common keys: {len(shape_equal)}  diff-shape common keys: {len(shape_diff)}')

# Naive class-dependent filter: keep diffs whose key contains likely classification words
class_keywords = ('cls', 'class', 'detect', 'head')
class_like = [(k,sb,so) for (k,sb,so) in shape_diff if any(kw in k.lower() for kw in class_keywords)]
other_diffs = [(k,sb,so) for (k,sb,so) in shape_diff if not any(kw in k.lower() for kw in class_keywords)]
print(f'class-like diffs: {len(class_like)}  other diffs: {len(other_diffs)}')
if other_diffs[:10]:
    print('sample other diffs:', other_diffs[:5])

# Dtype/device checks
import itertools
for name in ['best','official']:
    dtypes = set(str(v.dtype) for v in r[name].values() if hasattr(v, 'dtype'))
    devices = set(str(v.device) for v in r[name].values() if hasattr(v, 'device'))
    print(f'[{name}] dtypes={sorted(dtypes)} devices={sorted(devices)}')

