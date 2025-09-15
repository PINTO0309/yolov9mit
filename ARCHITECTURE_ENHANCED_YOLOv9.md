# YOLOv9 (MIT) — Enhanced Architecture & Training Pipeline

This document explains the complete data flow and architecture of this YOLOv9 (MIT) codebase: model graph syntax, heads/decoders, losses, training/validation/inference loops, EMA and checkpointing, online knowledge distillation (teacher E → student {C,S,T,N}), augmentation, and fast-inference export.

---

## 1) Big Picture

- Declarative model graphs (YAML) are parsed into a directed acyclic graph with tagged nodes and explicit sources (skip/fuse) to instantiate the module list.
- Two detection branches:
  - Main: the deploy-time head
  - AUX: an auxiliary supervision branch (self-distillation during train only)
- One-stage anchor-free head with Distribution Focal Loss (DFL) for bounding box regression.
- Post-process converts network outputs to image-space boxes and applies per-class NMS.
- Training uses Lightning with mixed-precision, EMA support, and checkpoint writers. Validation is COCO-compliant via TorchMetrics.
- Optional online KD combines an external teacher (E) with the self-distilled dual-branch pipeline.

---

## 2) Model Graph & Modules

Model graphs live under `yolo/config/model/*.yaml`. The parser in `yolo/model/yolo.py` reads the `model:` section and builds a `nn.ModuleList` in order. Each layer:

- has a `layer_type` (e.g., `Conv`, `RepNCSPELAN`, `SPPELAN`, `ADown`, `CBLinear`, `CBFuse`, `Concat`, `UpSample`, `MultiheadDetection`),
- may have `args` (hyperparameters), `source` (single index, relative index, tag name, or list), and an optional `tags` string (to reference later).

Key building blocks (see `yolo/model/module.py`):

- Conv, Pool, Concat, UpSample: standard ops
- ELAN, RepNCSP, RepNCSPELAN: backbone/neck building blocks (ELAN/CSP-style)
- AConv / ADown: down-sampling variants used in v9
- SPPELAN: SPP-style block
- CBLinear: splits convolutional output into multiple channel slices for routing
- CBFuse: resamples + additive fusion from multiple routing taps to the current stage
- MultiheadDetection: creates per-scale detection heads (see below)

YAML tags (e.g., `B3`, `N4`, `P5`, `A3`) mark layer outputs, allowing later nodes to reference them in `source`. Example (simplified):

```yaml
- SPPELAN: {args: {out_channels: 256}}
  tags: N3
- UpSample: {args: {scale_factor: 2, mode: nearest}}
- Concat: {source: [-1, B4]}
- RepNCSPELAN: {args: {out_channels: 192, part_channels: 192}}
  tags: N4
```

Variants provided: `v9-n`, `v9-t`, `v9-s`, `v9-m`, `v9-c`, `v9-e` (MIT). The E variant includes routing taps (via `CBLinear`) and fused stages consistent with the GPL architecture intent while respecting this codebase’s modules.

---

## 3) Detection Head & Predictions

`MultiheadDetection` builds one head per feature map (typically P3, P4, P5). Each head outputs:

- Class logits: shape `[B, C, H, W]`
- DFL anchor logits: shape `[B, 4*reg_max, H, W]` (packed as 4 directions × reg_max bins)
- Vector expectations: shape `[B, 4, H, W]` from `Anchor2Vec` (softmax over reg_max, then expectation)

Notes:
- `reg_max = 16` by default across the repo
- Heads are anchor-free; distances are predicted w.r.t. grid centers and later converted to xyxy.

---

## 4) Vec2Box Decoder & NMS

- Decoder: `yolo/utils/bounding_box_utils.py::Vec2Box`
  - Builds an anchor grid with configured `strides` (defaults to `[8,16,32]` unless set in YAML).
  - Concatenates predictions from all scales along the grid dimension.
  - Converts distances to absolute xyxy by scaling with the stride and grid centers.
- NMS: `bbox_nms`
  - Class-wise per-image NMS using `torchvision.ops.batched_nms`.
  - Inputs: class logits (sigmoid), boxes, and optional confidence.
  - Output per image: `[N, 6]` of `[class, x1, y1, x2, y2, score]` (capped by `max_bbox`).

The PostProcess wrapper (`yolo/utils/model_utils.py::PostProcess`) handles decoder updates for the current image size and calls `bbox_nms`.

---

## 5) Losses & Self-Distillation (Dual Branch)

Training loss implementation in `yolo/tools/loss_functions.py`:

- YOLOLoss (per-branch):
  - Classification: BCEWithLogits over grid × classes
  - IoU: CIoU-based `BoxLoss`
  - DFL: distance focal loss over the reg_max distributions
  - Box matching: configurable matcher (e.g., CIoU + top-k), producing aligned targets and valid masks
- DualLoss combines AUX and Main: weighted sum controlled by `task.loss.aux` (e.g., 0.25)
  - Total = (AUX + Main) for each component, then weighted by objective coefficients

The AUX branch is only used for training (self-distillation) and not in deploy-time inference.

---

## 6) Training/Validation/Inference Loops

Powered by Lightning via `yolo/lazy.py` and `yolo/tools/solver.py`:

- Trainer config:
  - mixed precision (`precision="16-mixed"`), gradient clipping, deterministic
  - EMA callback (optional) and rich/TensorBoard logging
- Train loop (`TrainModel`):
  - Builds student model; optimizer/scheduler from YAML
  - On each step: forward → decode (AUX/Main) → compute loss → log LR and losses
- Validation loop (`ValidateModel`):
  - Uses student EMA if enabled; COCO-style mAP via TorchMetrics (`faster_coco_eval` backend)
  - Pretty-printed AP/AR tables; optional per-class mAP at every epoch
- Inference loop (`InferenceModel`):
  - Uses `StreamDataLoader` for images/videos/webcams
  - PostProcess → draw_bboxes (outline-only by default; optional labels)
  - Optional Fast Inference (ONNX/TRT), see below

Resume training: set `task.resume_ckpt: <path_to.ckpt>` to continue; `task.epoch` is the absolute max epoch (not “+N”).

---

## 7) EMA & Checkpointing

- EMA: `yolo/utils/model_utils.py::EMA`
  - Maintains an EMA copy of the student model. Validation uses EMA; best/last weights export use EMA when present.
- SaveBestWeights: on each validation epoch end
  - Saves `last.pt` and `best_<variant>_<epoch>_<mAP>.pt` next to Lightning ckpts.
  - Export format is a flat state_dict (official-style), detached from the Lightning wrapper.

Teacher models (for online KD) are never included in EMA or saved student `.pt` files.

---

## 8) Online Knowledge Distillation (Teacher E → Student {C,S,T,N})

This section expands the teacher–student online KD design and how it layers on top of the built‑in self‑distillation (AUX branch). The mental model is “GT supervision + self‑distillation (AUX) + teacher mimic loss (KD)”, where KD does not replace GT but softly guides the student toward the teacher’s distributions.

### 8.1 High‑level pipeline (self‑distill + KD)

```
Images ──► Student (AUX, Main) ──► Decoders ──► GT Loss (AUX+Main)
    │             ▲                                    ▲
    │             └──── KD Loss (optional, apply_to: main|aux|both) ◄── Teacher (E, frozen)
    │
    └────────────────────────────────────────────────────────────────► Optimizer/EMA (student only)
```

Key points:
- Self‑distillation = AUX+Main heads supervised by the same GT (AUX is train‑time only, helps representation).
- Online KD = Add mimic loss between Student and Teacher outputs on selected branches.
- EMA/Checkpoints remain student‑only; the teacher is never saved or EMA’d.

### 8.2 Flow diagram (with branches)

```mermaid
flowchart LR
  I[Images] -->|forward| S[Student Net]
  S --> SA[AUX Heads]
  S --> SM[Main Heads]
  I -->|forward (no grad, FP16)| T[Teacher Net (E, frozen)]
  T --> TA[Teacher AUX Heads]
  T --> TM[Teacher Main Heads]

  subgraph GT_SelfDistill
    SA -->|Vec2Box| DA[Decode AUX]
    SM -->|Vec2Box| DM[Decode Main]
    DA -->|AUX GT Loss| L1[Loss]
    DM -->|Main GT Loss| L1
  end

  subgraph Online_KD
    SA -. optional .->|KD (apply_to=aux/both)| KDA[CLS/DFL/BOX KD]
    SM -->|KD (apply_to=main/both)| KDM[CLS/DFL/BOX KD]
    TA -. provides .- KDA
    TM -. provides .- KDM
  end

  KDA --> L1
  KDM --> L1
  L1 -->|backprop (student only)| OPT[Optimizer/EMA]
```

### 8.3 Configuration

Configurable via `task.kd` in `yolo/config/task/train.yaml` and CLI overrides:

```yaml
kd:
  enable: false            # turn on to use KD
  teacher_model: v9-e      # selects YAML from yolo/config/model/v9-e.yaml
  teacher_weight: weights/v9-e.pt
  apply_to: main           # main | aux | both
  temperature: {init: 4.0, final: 1.5, schedule: cosine}
  weights: {cls: 1.0, dfl: 0.5, box: 0.25}
  teacher_fp16: true
  freeze_teacher: true
```

Implementation (no changes required by users beyond config):

- Teacher is loaded in `TrainModel.setup` as a separate, frozen module (eval+no_grad; optional FP16).
- Temperature schedule: cosine annealing epoch-wise from `init` to `final`.
- KD losses are added on top of the standard GT loss:
  - Classification KD: BCEWithLogits(student/T vs teacher.sigmoid()/T) × T²
  - DFL KD: KL divergence over reg_max distributions (teacher/T || student/T) × T²
  - Box KD: L1 between vector expectations (pre-xyxy) or comparable continuous targets
- Branch selection: apply KD to `main`, `aux`, or `both` (recommend start with `main`).
- Logging: KD/cls, KD/dfl, KD/box metrics.
- Isolation: Teacher is not part of optimizer/EMA/checkpoint — deploy weights remain student-only.

Tuning tips:
- Start with `apply_to=main`, then try `both` with lower KD weights to avoid over-regularization.
- Keep `final` temperature ≥ ~1.5–2.0 for smoother targets.

### 8.4 Temperature schedule (cosine)

Let `E` be total epochs, `e` current epoch, `T0` init, `T1` final. We use:

```
T(e) = T1 + 0.5 * (T0 - T1) * (1 + cos(pi * e / (E - 1)))
```

This starts with softer targets (higher temperature) and gradually sharpens them, reducing KD strength toward the end as the model converges under GT.

### 8.5 Loss components and where they act

Per selected branch (Main and/or AUX), for each detection scale:

- Classification KD (multi‑label friendly):
  - Teacher: `p_t = sigmoid(z_t / T)`
  - Student: `z_s / T`
  - Loss: `BCEWithLogits(z_s/T, p_t) * T^2`
- DFL KD (distribution over reg_max bins per direction):
  - `KL(softmax(a_t/T) || log_softmax(a_s/T)) * T^2`
- Box KD (continuous target):
  - L1 between Student/Teacher vector expectations (`[B,4,H,W]` before xyxy transform)

All KD terms are added on top of standard GT losses. Weights `kd.weights.{cls,dfl,box}` control their relative strengths.

### 8.6 Self‑distillation vs Online KD — relationship

| Aspect               | Self‑distillation (AUX)                       | Online KD (Teacher→Student)                         |
|----------------------|-----------------------------------------------|----------------------------------------------------|
| Supervision source   | Ground truth                                   | Teacher distributions + vectors                     |
| Where applied        | AUX and Main (via DualLoss)                    | Main / AUX / both (configurable)                    |
| Training‑time cost   | Low (single forward)                           | +Teacher forward (no‑grad, optional FP16)           |
| Influence on deploy  | Main only used at inference                    | Still Main only; teacher never saved/EMA’d          |
| Risk                 | Overfitting in AUX if too strong               | Over‑regularization if KD too strong (tune weights) |

Guideline: keep self‑distillation (AUX) active as in defaults, add KD first to `main` only. If stable, consider `both` with reduced KD weights.

### 8.7 Pseudocode (how it integrates)

```python
# TrainModel.training_step
predicts = student(images)
aux_s = vec2box(predicts['AUX'])
main_s = vec2box(predicts['Main'])
loss, items = GT_loss(aux_s, main_s, targets)   # existing DualLoss

if kd.enabled:
    T = cosine_temperature(epoch, max_epochs, kd.T_init, kd.T_final)
    with torch.no_grad():
        teacher.eval(); t_out = teacher(images_half_if_cuda)
    kd_total = 0
    for branch in selected_branches:  # main and/or aux
        for (s_cls, s_anc, s_vec), (t_cls, t_anc, t_vec) in zip(predicts[branch], t_out[branch]):
            kd_total += kd_cls(s_cls, t_cls, T) + kd_dfl(s_anc, t_anc, T) + kd_box(s_vec, t_vec)
    loss = loss + w_cls*kd_total.cls + w_dfl*kd_total.dfl + w_box*kd_total.box
```

### 8.8 Practical recipes

- Start: `apply_to=main`, `weights={cls:1.0, dfl:0.5, box:0.25}`, `temperature: 4.0→1.5`.
- If training becomes unstable or mAP drops, try: increase `final` to 2.0, reduce `cls/dfl/box` by 20–40%.
- When switching to `both`, halve KD weights to compensate shared‑backbone pressure from AUX.


---

## 9) Data Loading & Augmentation

- Dataset: YOLO format (`images/{train,val}`, `labels/{train,val}`), or COCO JSON via helper utilities.
- Loader: `YoloDataset` caches metadata (`*.pache`), dynamic shapes per batch boundary (optional).
- Collate caps targets per image to 100 by default to constrain memory.
- Augmentations (`yolo/tools/data_augmentation.py`):
  - Geometric: RandomCrop, Horizontal/Vertical Flip (optional)
  - Photometric: RandomHSV (independent hue/saturation/value gains), RandomBrightness/Contrast/Saturation
  - Albumentations wrappers: blur/noise/compression/weather (optional)
  - Mosaic/MixUp supported (commented by default)

---

## 10) Fast Inference (ONNX / TensorRT)

- Toggle via `task.fast_inference: onnx|trt|deploy` (for inference task only)
- Loader: `yolo/utils/deploy_utils.py::FastModelLoader`
  - ONNX: exports to `<weight_path>.onnx`, then runs `onnxsim` 3 passes; loads via onnxruntime
  - TRT: converts via `torch2trt` and saves to `<weight_path>.trt`
  - Fallbacks to PyTorch on failure

Note: ONNX/TRT models return the same output structure as PyTorch (`{"Main": [...]}`) for downstream compatibility.

---

## 11) Visualization

- `draw_bboxes` supports outline-only boxes (sharp corners) and optional label rendering.
- In inference, outlines are drawn without fill by default; labels toggled by `task.render_labels`.

---

## 12) Configuration Structure (Hydra)

Top-level config: `yolo/config/config.yaml`

- `defaults` select `task` (train/validation/inference), `dataset`, `model`, and `general` settings.
- `general.yaml` contains hardware, logging, and image size defaults.
- Task configs:
  - `task/train.yaml`: data loader, optimizer, loss/matcher, scheduler, EMA, KD
  - `task/validation.yaml`: data loader, NMS, optional per-class mAP printing
  - `task/inference.yaml`: input source, NMS, save options, fast inference

Override examples (CLI):

```bash
# Train T with online KD from E (student-only EMA/best)
uv run python yolo/lazy.py \
  task=train model=v9-t dataset=wholebody25 device=cuda \
  task.kd.enable=true task.kd.teacher_model=v9-e task.kd.teacher_weight=weights/v9-e.pt

# Validation with exact batch size to match mAP reported during training
uv run python yolo/lazy.py task=validation task.data.batch_size=32

# Inference with ONNX (exports next to the weight file and loads it)
uv run python yolo/lazy.py \
  task=inference model=v9-t dataset=wholebody25 device=cuda \
  weight=runs/train/v9-t/.../best_t_xxxx_0.29xx.pt task.fast_inference=onnx \
  task.data.source=demo/images/inference/image.png
```

---

## 13) Practical Notes & Gotchas

- Match dataset (class count/order) with the weight. If mismatched, detection head weights won’t load and mAP collapses.
- mAP is sensitive to NMS settings and batch size at validation time. To reproduce training mAP, align `task.data.batch_size` and `task.nms.*`.
- Augmentations, Mosaic/MixUp, and dynamic shapes increase RAM usage; tune `batch_size` accordingly.
- `task.epoch` is the absolute max epoch. When resuming (`task.resume_ckpt`), training runs while `current_epoch < task.epoch`.

---

## 14) File Map (Key Components)

- Model definition: `yolo/model/yolo.py`, `yolo/model/module.py`
- Config schemas: `yolo/config/config.py`
- Task configs: `yolo/config/task/*.yaml`
- Model YAMLs: `yolo/config/model/*.yaml`
- Losses & matcher: `yolo/tools/loss_functions.py`, `yolo/utils/bounding_box_utils.py`
- Loops: `yolo/tools/solver.py`, `yolo/lazy.py`
- Fast inference: `yolo/utils/deploy_utils.py`
- Visualization: `yolo/tools/drawer.py`

---

## 15) Summary

This codebase provides a clean, configurable implementation of YOLOv9 under MIT, featuring a flexible model graph, dual-branch self-distillation, anchor-free DFL regression, robust training/validation/inference loops, EMA/best weight exports, optional ONNX/TRT deployment, and an extensible online KD mechanism that keeps teacher artifacts out of final student weights.
