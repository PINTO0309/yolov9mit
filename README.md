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

<table>
<tr><td>

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

```shell
uv run python yolo/lazy.py task=train task.data.batch_size=8 model=v9-c dataset={dataset_config} device={cpu, mps, cuda}

# n, t, s, c
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
```

If you want to display the AP for each class for all epochs, change `yolo/config/task/validation.yaml`'s `print_map_per_class: True` and start training.

```
┏━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━┓
┃Epoch┃Avg. Precision  ┃     %╇Avg. Recall     ┃     %┃
┡━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━┩
│    2│AP @ .5:.95     │000.77╎AR maxDets   1  │003.08│
│    2│AP @     .5     │002.02╎AR maxDets  10  │006.91│
│    2│AP @    .75     │000.45╎AR maxDets 100  │008.74│
│    2│AP  (small)     │000.33╎AR     (small)  │001.93│
│    2│AP (medium)     │000.69╎AR    (medium)  │007.74│
│    2│AP  (large)     │001.34╎AR     (large)  │008.55│
└─────┴────────────────┴──────┴────────────────┴──────┘
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ ID┃Name                     ┃     AP┃ ID┃Name                     ┃     AP┃ ID┃Name                     ┃     AP┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│  0│body                     │ 0.0343│ 20│ear                      │ 0.0023│   │                         │       │
│  1│adult                    │ 0.0320│ 21│collarbone               │ 0.0003│   │                         │       │
│  2│child                    │ 0.0000│ 22│shoulder                 │ 0.0033│   │                         │       │
│  3│male                     │ 0.0268│ 23│solar_plexus             │ 0.0003│   │                         │       │
│  4│female                   │ 0.0103│ 24│elbow                    │ 0.0001│   │                         │       │
│  5│body_with_wheelchair     │ 0.0029│ 25│wrist                    │ 0.0001│   │                         │       │
│  6│body_with_crutches       │ 0.0455│ 26│hand                     │ 0.0029│   │                         │       │
│  7│head                     │ 0.0340│ 27│hand_left                │ 0.0022│   │                         │       │
│  8│front                    │ 0.0102│ 28│hand_right               │ 0.0027│   │                         │       │
│  9│right-front              │ 0.0155│ 29│abdomen                  │ 0.0005│   │                         │       │
│ 10│right-side               │ 0.0059│ 30│hip_joint                │ 0.0006│   │                         │       │
│ 11│right-back               │ 0.0023│ 31│knee                     │ 0.0010│   │                         │       │
│ 12│back                     │ 0.0001│ 32│ankle                    │ 0.0012│   │                         │       │
│ 13│left-back                │ 0.0015│ 33│foot                     │ 0.0063│   │                         │       │
│ 14│left-side                │ 0.0025│   │                         │       │   │                         │       │
│ 15│left-front               │ 0.0105│   │                         │       │   │                         │       │
│ 16│face                     │ 0.0047│   │                         │       │   │                         │       │
│ 17│eye                      │ 0.0000│   │                         │       │   │                         │       │
│ 18│nose                     │ 0.0000│   │                         │       │   │                         │       │
│ 19│mouth                    │ 0.0000│   │                         │       │   │                         │       │
└───┴─────────────────────────┴───────┴───┴─────────────────────────┴───────┴───┴─────────────────────────┴───────┘
```

### Inference

To use a model for object detection, use:

```shell
python yolo/lazy.py # if cloned from GitHub
python yolo/lazy.py task=inference \ # default is inference
                    name=AnyNameYouWant \ # AnyNameYouWant
                    device=cpu \ # hardware cuda, cpu, mps
                    model=v9-s \ # model version: v9-c, m, s
                    task.nms.min_confidence=0.1 \ # nms config
                    task.fast_inference=onnx \ # onnx, trt, deploy
                    task.data.source=data/toy/images/train \ # file, dir, webcam
                    +quite=True \ # Quite Output
yolo task.data.source={Any Source} # if pip installed
yolo task=inference task.data.source={Any}
```

### Validation

To validate model performance, or generate a json file in COCO format:

```shell
python yolo/lazy.py task=validation
python yolo/lazy.py task=validation dataset=toy
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
