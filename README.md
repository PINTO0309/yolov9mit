# This fork is an MIT version of YOLO, with some bug fixes and the addition of the V9-N (nano) and V9-E (extended) variants to the original. https://github.com/MultimediaTechLab/YOLO.

---

# YOLO: Official Implementation of YOLOv9, YOLOv7, YOLO-RD

[![Documentation Status](https://readthedocs.org/projects/yolo-docs/badge/?version=latest)](https://yolo-docs.readthedocs.io/en/latest/?badge=latest)
![GitHub License](https://img.shields.io/github/license/WongKinYiu/YOLO)
![WIP](https://img.shields.io/badge/status-WIP-orange)

[![Developer Mode Build & Test](https://github.com/WongKinYiu/YOLO/actions/workflows/develop.yaml/badge.svg)](https://github.com/WongKinYiu/YOLO/actions/workflows/develop.yaml)
[![Deploy Mode Validation & Inference](https://github.com/WongKinYiu/YOLO/actions/workflows/deploy.yaml/badge.svg)](https://github.com/WongKinYiu/YOLO/actions/workflows/deploy.yaml)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/yolov9-learning-what-you-want-to-learn-using/real-time-object-detection-on-coco)](https://paperswithcode.com/sota/real-time-object-detection-on-coco)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-green)](https://huggingface.co/spaces/henry000/YOLO)

<!-- > [!IMPORTANT]
> This project is currently a Work In Progress and may undergo significant changes. It is not recommended for use in production environments until further notice. Please check back regularly for updates.
>
> Use of this code is at your own risk and discretion. It is advisable to consult with the project owner before deploying or integrating into any critical systems. -->

Welcome to the official implementation of YOLOv7[^1] and YOLOv9[^2], YOLO-RD[^3]. This repository will contains the complete codebase, pre-trained models, and detailed instructions for training and deploying YOLOv9.

## TL;DR

- This is the official YOLO model implementation with an MIT License.
- For quick deployment: you can directly install by pip+git:

```shell
pip install git+https://github.com/WongKinYiu/YOLO.git
yolo task.data.source=0 # source could be a single file, video, image folder, webcam ID
```

## Introduction

- [**YOLOv9**: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616)
- [**YOLOv7**: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors](https://arxiv.org/abs/2207.02696)
- [**YOLO-RD**: Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary](https://arxiv.org/abs/2410.15346)

## Installation

To get started using YOLOv9's developer mode, we recommand you clone this repository and install the required dependencies:

```shell
git clone https://github.com/PINTO0309/yolov9mit.git
cd yolov9mit

curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

## Features

## Task

These are simple examples. For more customization details, please refer to [Notebooks](examples) and lower-level modifications **[HOWTO](docs/HOWTO.md)**.

## YOLO format dataset structure

```
data
└── wholebody34
    ├── train.pache # Cache file automatically generated when training starts
    ├── val.pache # Cache file automatically generated when training starts
    ├── images
    │   ├── train
    │   │   ├── 000000000036.jpg
    │   │   ├── 000000000077.jpg
    │   │   ├── 000000000110.jpg
    │   │   ├── 000000000113.jpg
    │   │   └── 000000000165.jpg
    │   └── val
    │       ├── 000000000241.jpg
    │       ├── 000000000294.jpg
    │       ├── 000000000308.jpg
    │       ├── 000000000322.jpg
    │       └── 000000000328.jpg
    └── labels
        ├── train
        │   ├── 000000000036.txt
        │   ├── 000000000077.txt
        │   ├── 000000000110.txt
        │   ├── 000000000113.txt
        │   └── 000000000165.txt
        └── val
            ├── 000000000241.txt
            ├── 000000000294.txt
            ├── 000000000308.txt
            ├── 000000000322.txt
            └── 000000000328.txt
```

- `000000000036.txt`
    |Item|Note|
    |:-|:-|
    |classId|classId|
    |cx, cy|0.0-1.0 normalized center coordinates|
    |w, h|0.0-1.0 normalized width and height|

    `classId cx cy w h`
    ```
    30 0.729688 0.959667 0.141042 0.080667
    25 0.919385 0.974417 0.052521 0.051167
    25 0.525000 0.680847 0.049167 0.071806
    23 0.663813 0.657361 0.100125 0.105889
    21 0.612667 0.519583 0.068542 0.068056
    29 0.628292 0.896000 0.292500 0.082889
    30 0.546063 0.957611 0.210792 0.084778
    19 0.547917 0.417986 0.073125 0.037361
    26 0.488281 0.653583 0.123104 0.151444
    24 0.840208 0.778889 0.080417 0.092222
    24 0.435312 0.790972 0.074375 0.089167
    22 0.411469 0.557500 0.103313 0.112222
    22 0.773646 0.546944 0.087708 0.110556
    9 0.560417 0.366667 0.233333 0.266667
    7 0.560417 0.366667 0.233333 0.266667
    27 0.956385 0.970417 0.087229 0.055833
    16 0.541667 0.370833 0.154167 0.197222
    26 0.956385 0.970417 0.087229 0.055833
    4 0.681458 0.621667 0.637083 0.756667
    0 0.681458 0.621667 0.637083 0.756667
    18 0.527188 0.373333 0.042917 0.047500
    20 0.644792 0.370028 0.023125 0.036667
    1 0.681458 0.621667 0.637083 0.756667
    28 0.488281 0.653583 0.123104 0.151444
    17 0.489687 0.370972 0.032917 0.020556
    17 0.561875 0.350694 0.044583 0.019722
    ```

## Dataset config

`yolo/config/dataset/wholebody34.yaml`

```yaml
path: data/wholebody34
train: train
validation: val

class_num: 34
class_list: ['body', 'adult', 'child', 'male', 'female', 'body_with_wheelchair', 'body_with_crutches', 'head', 'front', 'right-front', 'right-side', 'right-back', 'back', 'left-back', 'left-side', 'left-front', 'face', 'eye', 'nose', 'mouth', 'ear', 'collarbone', 'shoulder', 'solar_plexus', 'elbow', 'wrist', 'hand', 'hand_left', 'hand_right', 'abdomen', 'hip_joint', 'knee', 'ankle', 'foot']

auto_download:
```

## Training

To train YOLO on your machine/dataset:

1. Modify the configuration file `yolo/config/dataset/**.yaml` to point to your dataset.
2. Run the training script:

```shell
uv run python yolo/lazy.py task=train dataset=** use_wandb=True
uv run python yolo/lazy.py task=train task.data.batch_size=8 model=v9-c weight=False # or more args
```

### Transfer Learning

To perform transfer learning with YOLOv9:

- `configs`
  - https://github.com/PINTO0309/yolov9mit/blob/wholebody/yolo/config/config.yaml
  - https://github.com/PINTO0309/yolov9mit/blob/wholebody/yolo/config/general.yaml
  - https://github.com/PINTO0309/yolov9mit/blob/wholebody/yolo/config/task/train.yaml
  - https://github.com/PINTO0309/yolov9mit/blob/wholebody/yolo/config/task/validation.yaml
  - https://github.com/PINTO0309/yolov9mit/tree/wholebody/yolo/config/model

```shell
# n, t, s, c, e
VARIANT=n
EPOCH=100
BATCHSIZE=8

uv run python yolo/lazy.py \
task=train \
name=v9-${VARIANT} \
task.epoch=${EPOCH} \
task.data.batch_size=${BATCHSIZE} \
model=v9-${VARIANT} \
dataset=wholebody34 \
device=cuda \
use_wandb=False \
use_tensorboard=True

# When specifying trained weights as initial weights
uv run python yolo/lazy.py \
task=train \
name=v9-${VARIANT} \
task.epoch=${EPOCH} \
task.data.batch_size=${BATCHSIZE} \
model=v9-${VARIANT} \
weight="runs/train/v9-n/lightning_logs/version_1/checkpoints/best_n_0002_0.0065.pt" \
dataset=wholebody34 \
device=cuda \
use_wandb=False \
use_tensorboard=True

# Automatically downloading the initial weights published by the official repository
# Default: weight=True
# Weight download path: weights/*.pt
uv run python yolo/lazy.py \
task=train \
name=v9-${VARIANT} \
task.epoch=${EPOCH} \
task.data.batch_size=${BATCHSIZE} \
model=v9-${VARIANT} \
weight=True \
dataset=wholebody34 \
device=cuda \
use_wandb=False \
use_tensorboard=True

# When starting training without initial weights
# Default: weight=True
uv run python yolo/lazy.py \
task=train \
name=v9-${VARIANT} \
task.epoch=${EPOCH} \
task.data.batch_size=${BATCHSIZE} \
model=v9-${VARIANT} \
weight=False \
dataset=wholebody34 \
device=cuda \
use_wandb=False \
use_tensorboard=True

# Resume learning from where you left off
# Please note that you must specify the Lightning checkpoint file (.ckpt)
# and not the .pt file that contains only the EMA weights.
uv run python yolo/lazy.py \
task=train \
name=v9-${VARIANT} \
task.epoch=${EPOCH} \
task.data.batch_size=${BATCHSIZE} \
model=v9-${VARIANT} \
task.resume_ckpt="runs/train/v9-n/lightning_logs/version_3/checkpoints/epoch_5_step_3660.ckpt" \
dataset=wholebody34 \
device=cuda \
use_wandb=False \
use_tensorboard=True

# # DDP (Distributed data parallel training), Multi-GPU training
# # Below is a sample for 8 GPUs
# # n, t, s, c, e
# VARIANT=n
# EPOCH=100
# # Number of GPUs running on one node
# NPROC=8
# # When NPROC=8, the string [0,1,2,3,4,5,6,7] is set to DEVICES.
# DEVICES="[$(seq -s, 0 $((NPROC-1)))]"
# # When there are 8 GPUs and 8 batches are assigned to each GPU
# # {Batch size per GPU} x {Number of GPUs} = {Total batch size}
# # 8 x 8 = 64
# BATCHSIZE=8
# TOTALBATCHSIZE=$((BATCHSIZE * NPROC))

# uv run torchrun \
# --nproc_per_node=${NPROC} \
# yolo/lazy.py \
# task=train \
# device=${DEVICES} \
# name=v9-${VARIANT} \
# task.epoch=${EPOCH} \
# task.data.batch_size=${TOTALBATCHSIZE} \
# task.data.cpu_num=$((TOTALBATCHSIZE / NPROC)) \
# model=v9-${VARIANT} \
# weight=False \
# dataset=wholebody34 \
# use_wandb=False \
# use_tensorboard=True

↓↓↓ Experimental implementation. Not recommended as accuracy is significantly reduced. ↓↓↓
# Online Knowledge Distillation (Teacher E → Student {C,S,T,N})
# Default: task.kd.enable=False
# ./ARCHITECTURE_ENHANCED_YOLOv9.md#8-online-knowledge-distillation-teacher-e--student-cstn
# ./yolo/config/task/train.yaml
uv run python yolo/lazy.py \
task=train \
name=v9-${VARIANT} \
task.epoch=${EPOCH} \
task.data.batch_size=${BATCHSIZE} \
task.kd.enable=True \
model=v9-${VARIANT} \
weight=False \
dataset=wholebody34 \
device=cuda \
use_wandb=False \
use_tensorboard=True
↑↑↑ Experimental implementation. Not recommended as accuracy is significantly reduced. ↑↑↑
```

⚠️ **important points** ⚠️

Pay particular attention to the maximum number of CPU threads and the amount of RAM on the machine you are trying to train on. I'm talking RAM, not VRAM. The number of worker processes specified during training is `batch_size + 1`, but you must adjust `batch_size` so that it is less than the maximum number of CPU `threads - 1`. Also, the amount of RAM consumed increases in proportion to the number of enabled augmentations, so you need to pay attention to the amount of RAM installed on your PC. Checking only the amount of VRAM is not enough. If you need to run heavy augmentation that exceeds the RAM capacity, we recommend setting `batch_size` to a relatively small value.

The figure below shows the CPU and RAM status of my work PC. When I run 16 batches with the maximum number of augmentations enabled, 17 threads are started, which not only consumes a lot of RAM, causing the learning process to silently abort after a few epochs without outputting any errors.

<img width="640" alt="image" src="https://github.com/user-attachments/assets/74d2e28a-a351-4491-aa4f-605056656b34" />

### print_map_per_class

If you want to display the AP for each class for all epochs, change `yolo/config/task/validation.yaml`'s `print_map_per_class: True` and start training. If `print_map_per_class: False` is set, AP per class will be calculated and output only once at the end of the final epoch. Since `print_map_per_class` takes a very long time to process, we recommend setting it to `False` and automatically calculating `map_per_class` only in the final epoch.

```
┏━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━┓
┃Epoch┃Avg. Precision  ┃     %┃Avg. Recall     ┃     %┃
┡━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━┩
│    2│AP @ .5:.95     │000.77╎AR maxDets   1  │003.08│
│    2│AP @     .5     │002.02╎AR maxDets  10  │006.91│
│    2│AP @    .75     │000.45╎AR maxDets 100  │008.74│
│    2│AP  (small)     │000.33╎AR     (small)  │001.93│
│    2│AP (medium)     │000.69╎AR    (medium)  │007.74│
│    2│AP  (large)     │001.34╎AR     (large)  │008.55│
└─────┴────────────────┴──────┴────────────────┴──────┘
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ ID┃Name                     ┃     AP┃ ID┃Name                     ┃     AP┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│  0│body                     │ 0.0343│ 20│ear                      │ 0.0023│
│  1│adult                    │ 0.0320│ 21│collarbone               │ 0.0003│
│  2│child                    │ 0.0000│ 22│shoulder                 │ 0.0033│
│  3│male                     │ 0.0268│ 23│solar_plexus             │ 0.0003│
│  4│female                   │ 0.0103│ 24│elbow                    │ 0.0001│
│  5│body_with_wheelchair     │ 0.0029│ 25│wrist                    │ 0.0001│
│  6│body_with_crutches       │ 0.0455│ 26│hand                     │ 0.0029│
│  7│head                     │ 0.0340│ 27│hand_left                │ 0.0022│
│  8│front                    │ 0.0102│ 28│hand_right               │ 0.0027│
│  9│right-front              │ 0.0155│ 29│abdomen                  │ 0.0005│
│ 10│right-side               │ 0.0059│ 30│hip_joint                │ 0.0006│
│ 11│right-back               │ 0.0023│ 31│knee                     │ 0.0010│
│ 12│back                     │ 0.0001│ 32│ankle                    │ 0.0012│
│ 13│left-back                │ 0.0015│ 33│foot                     │ 0.0063│
│ 14│left-side                │ 0.0025│   │                         │       │
│ 15│left-front               │ 0.0105│   │                         │       │
│ 16│face                     │ 0.0047│   │                         │       │
│ 17│eye                      │ 0.0000│   │                         │       │
│ 18│nose                     │ 0.0000│   │                         │       │
│ 19│mouth                    │ 0.0000│   │                         │       │
└───┴─────────────────────────┴───────┴───┴─────────────────────────┴───────┘
```

### Weights after training
The weights after training are output to the following path.

|File|Note|
|:-|:-|
|`best_{variant}_{epoch:04}_{map:.4f}.pt`|Optimized weight file containing only EMA weights. The weights with the highest mAP are automatically saved.|
|`epoch_{epoch}_step_{step}.ckpt`|A checkpoint file containing all learning logs automatically saved by Lightning.|
|`last.pt`|Optimized weight file containing only EMA weights. The weights of the last epoch are automatically saved.|

e.g.
```
runs/train/v9-n/lightning_logs/version_0/checkpoints
├── best_n_0002_0.0065.pt
├── epoch_2_step_3462.ckpt
└── last.pt
```

### Inference

To use a model for object detection, use:

```shell
# n, t, s, c, e
VARIANT=n
RENDER_LABELS=False

# If you do not specify `dataset={dataset_name}` correctly,
# the classification head weights will not be loaded properly
# and you will not see any inference results.
# The number of classes in the head part of the weights used for inference
# must match `class_num`.
# https://github.com/PINTO0309/yolov9mit/blob/wholebody/yolo/config/dataset/wholebody34.yaml
---
path: data/wholebody34
train: train
validation: val

class_num: 34 # <--- Here
class_list: ['body', ..., 'foot']
---

uv run python yolo/lazy.py \
task=inference \
name=v9-${VARIANT} \
model=v9-${VARIANT} \
weight="runs/train/v9-n/lightning_logs/version_1/checkpoints/best_n_0002_0.0065.pt" \
dataset=wholebody34 \
task.nms.min_confidence=0.1 \
task.fast_inference=onnx \
task.data.source=data/wholebody34/images/val \
task.data.max_samples=100 \
task.render_labels=${RENDER_LABELS} \
+quite=True
```
<img width="480" height="360" alt="frame074" src="https://github.com/user-attachments/assets/8e39420e-7357-4d70-9d08-a70b913c4341" />

### Validation

To validate model performance, or generate a json file in COCO format:

```shell
# n, t, s, c, e
VARIANT=n
# Specify the same `batch_size` as the validation batch size used during training.
# Otherwise, the mAP value after validation will be significantly degraded.
# data:
#  batch_size: 32
# https://github.com/PINTO0309/yolov9mit/blob/wholebody/yolo/config/task/validation.yaml
BATCHSIZE=32
# The higher the model's performance, the more accurate the evaluation will be
# if the MAXDET value (the upper limit of the number of detections) is set to
# a larger value. The default value is 1,000. yolo/config/task/validation.yaml
# However, setting a value that exceeds the maximum number of labels contained
# in one image will have no effect. For example, in my dataset, an image contains
# a maximum of 3,875 labels, so setting it to 4,000 is appropriate.
MAXDET=20000

uv run python yolo/lazy.py \
task=validation \
name=v9-${VARIANT} \
task.data.batch_size=${BATCHSIZE} \
task.nms.pre_topk=${MAXDET} \
task.nms.max_bbox=${MAXDET} \
task.nms.multi_label=True \
task.nms.class_agnostic=False \
model=v9-${VARIANT} \
weight="runs/train/v9-n/lightning_logs/version_1/checkpoints/best_n_0002_0.0065.pt" \
dataset=wholebody34 \
device=cuda \
use_wandb=False
```

### Export

Use the Hydra-driven CLI to run the `export` task and produce a compact ONNX graph. The exporter emits a
single `[batch, 4 + num_classes, anchors]` tensor, keeps detection heads minimal, and derives an informative
filename (e.g. `best_n_0470_0.2904_1x3x480x640.onnx`). Example:

```bash
uv run python yolo/lazy.py \
task=export \
name=v9-demo \
model=v9-n \
dataset=wholebody34 \
weight="runs/train/v9-n/lightning_logs/version_3/checkpoints/best_n_0470_0.2904.pt" \
task.batch_size=1 \
task.dynamic_batch=False \
task.image_size=480x640 \
task.opset=13 \
task.simplify=True \
task.half=false \
task.apply_sigmoid=True \
task.include_metadata=True
```

- `output: [batch, [cx,cy,w,h,class_scores], anchors]`
    <img width="1307" height="756" alt="image" src="https://github.com/user-attachments/assets/0fe37232-d327-4516-9467-23ab411c1018" />


Key overrides (all optional):
- `task.batch_size`: dummy input batch size (default 1).
- `task.dynamic_batch`: `true` marks batch as symbolic `N` and names the file accordingly.
- `task.image_size`: input resolution. Accepts `'HxW'`.
- `task.opset`: ONNX opset version (default 13).
- `task.simplify`: run `onnxsim` for graph simplification.
- `task.half`: export weights/activations in FP16.
- `task.apply_sigmoid`: emit post-sigmoid class probabilities instead of raw logits.
- `task.include_metadata`: embed class names in ONNX metadata.
- `task.output_path`: explicit destination; omit to auto-name beside the weight file.
- `task.name`: experiment/run folder label (standard Hydra behaviour).

Other global defaults (`device`, `out_path`, etc.) still apply via `config/general.yaml` and can be
overridden the same way when invoking `yolo/lazy.py`.

## Generate and merge post-processing with NMS

[tools/post_process_gen_tools](./tools/post_process_gen_tools)

<img width="600" alt="image" src="https://github.com/user-attachments/assets/6b6233d7-e218-4ad0-bd6a-855cf8397312" />

## Convert ONNX with NMS to LiteRT/TensorFlow.js
If you want to use `webgpu`, you can use ONNX without NMS or TensorFlow.js models without NMS. If you don't want to go through ONNX, you can output the LiteRT model directly from PyTorch using [ai_edge_torch](https://github.com/google-ai-edge/ai-edge-torch).

- ONNX to TF/LiteRT
  ```bash
  # Transformation with `Grouped Convolution` disabled
  uv run onnx2tf -i yolov9_n_wholebody25_post_0100_1x3x480x640.onnx -dgc
  ```
- TF to TFJS
  ```bash
  uv run tensorflowjs_converter \
  --input_format tf_saved_model \
  --output_format tfjs_graph_model \
  saved_model \
  tfjs_model
  ```

## Simple performance benchmark using ONNX/TensorRT

```bash
# Install CUDA==12.9
#  https://developer.nvidia.com/cuda-toolkit-archive
# Install TensorRT==10.13.3.9-1+cuda12.9
#  https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html
```
```bash
uv run sit4onnx -if best_e_0205_0.4140_1x3x640x640.onnx -oep cpu

INFO: file: best_e_0205_0.4140_1x3x640x640.onnx
INFO: providers: ['CPUExecutionProvider']
INFO: input_name.1: images shape: [1, 3, 640, 640] dtype: float32
INFO: test_loop_count: 10
INFO: total elapsed time:  3673.502206802368 ms
INFO: avg elapsed time per pred:  367.3502206802368 ms
INFO: output_name.1: output shape: [1, 38, 8400] dtype: float32
```
```bash
uv run sit4onnx -if best_e_0205_0.4140_1x3x640x640.onnx -oep cuda

INFO: file: best_e_0205_0.4140_1x3x640x640.onnx
INFO: providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
INFO: input_name.1: images shape: [1, 3, 640, 640] dtype: float32
INFO: test_loop_count: 10
INFO: total elapsed time:  350.10218620300293 ms
INFO: avg elapsed time per pred:  35.01021862030029 ms
INFO: output_name.1: output shape: [1, 38, 8400] dtype: float32
```
```bash
# It will take a while to generate the TensorrtExecutionProvider_TRTKernel_*.engine cache.
uv run sit4onnx -if best_e_0205_0.4140_1x3x640x640.onnx -oep tensorrt

INFO: file: best_e_0205_0.4140_1x3x640x640.onnx
INFO: providers: ['TensorrtExecutionProvider', 'CPUExecutionProvider']
INFO: input_name.1: images shape: [1, 3, 640, 640] dtype: float32
INFO: test_loop_count: 10
INFO: total elapsed time:  104.28452491760254 ms
INFO: avg elapsed time per pred:  10.428452491760254 ms
INFO: output_name.1: output shape: [1, 38, 8400] dtype: float32
```
```bash
# With NMS + TensorRT
# For models with dynamic tensors as input, specify the size of the tensor
# to be tested using the --fixed_shapes / -fs option.
uv run sit4onnx -if yolov9_n_wholebody25_post_0100_1x3xHxW.onnx -oep tensorrt -fs 1 3 480 640

INFO: file: yolov9_n_wholebody25_post_0100_1x3x480x640.onnx
INFO: providers: ['TensorrtExecutionProvider', 'CPUExecutionProvider']
INFO: input_name.1: input_bgr shape: [1, 3, 480, 640] dtype: float32
INFO: test_loop_count: 10
INFO: total elapsed time:  20.3857421875 ms
INFO: avg elapsed time per pred:  2.03857421875 ms
INFO: output_name.1: batchno_classid_score_x1y1x2y2 shape: [0, 7] dtype: float32
```

## Contributing

Contributions to the YOLO project are welcome! See [CONTRIBUTING](docs/CONTRIBUTING.md) for guidelines on how to contribute.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=MultimediaTechLab/YOLO&type=Date)](https://star-history.com/#MultimediaTechLab/YOLO&Date)

## Citations

```
@inproceedings{wang2022yolov7,
      title={{YOLOv7}: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors},
      author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
      year={2023},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},

}
@inproceedings{wang2024yolov9,
      title={{YOLOv9}: Learning What You Want to Learn Using Programmable Gradient Information},
      author={Wang, Chien-Yao and Yeh, I-Hau and Liao, Hong-Yuan Mark},
      year={2024},
      booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
}
@inproceedings{tsui2024yolord,
      author={Tsui, Hao-Tang and Wang, Chien-Yao and Liao, Hong-Yuan Mark},
      title={{YOLO-RD}: Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary},
      booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
      year={2025},
}

```

[^1]: [**YOLOv7**: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors](https://arxiv.org/abs/2207.02696)

[^2]: [**YOLOv9**: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616)

[^3]: [**YOLO-RD**: Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary](https://arxiv.org/abs/2410.15346)
