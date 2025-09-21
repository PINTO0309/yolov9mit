from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from torch import nn


@dataclass
class AnchorConfig:
    strides: List[int]
    reg_max: Optional[int]
    anchor_num: Optional[int]
    anchor: List[List[int]]


@dataclass
class LayerConfg:
    args: Dict
    source: Union[int, str, List[int]]
    tags: str


@dataclass
class BlockConfig:
    block: List[Dict[str, LayerConfg]]


@dataclass
class ModelConfig:
    name: Optional[str]
    anchor: AnchorConfig
    model: Dict[str, BlockConfig]


@dataclass
class DownloadDetail:
    url: str
    file_size: int


@dataclass
class DownloadOptions:
    details: Dict[str, DownloadDetail]


@dataclass
class DatasetConfig:
    path: str
    class_num: int
    class_list: List[str]
    auto_download: Optional[DownloadOptions]


@dataclass
class DataConfig:
    shuffle: bool
    batch_size: int
    pin_memory: bool
    cpu_num: int
    image_size: List[int]
    data_augment: Dict[str, int]
    source: Optional[Union[str, int]]
    dynamic_shape: Optional[bool] = None
    max_samples: Optional[int] = None
    class_biased_oversampling: Optional[bool] = False
    class_biased_batch_formation: Optional[bool] = False


@dataclass
class OptimizerArgs:
    lr: float
    weight_decay: float
    momentum: float


@dataclass
class OptimizerConfig:
    type: str
    args: OptimizerArgs


@dataclass
class MatcherConfig:
    iou: str
    topk: int
    factor: Dict[str, int]


@dataclass
class LossConfig:
    objective: Dict[str, int]
    aux: Union[bool, float]
    matcher: MatcherConfig


@dataclass
class SchedulerConfig:
    type: str
    warmup: Dict[str, Union[int, float]]
    args: Dict[str, Any]


@dataclass
class EMAConfig:
    enable: bool
    decay: float


@dataclass
class KDTempConfig:
    init: float
    final: float
    schedule: str


@dataclass
class KDWeightsConfig:
    cls: float
    dfl: float
    box: float


@dataclass
class KDConfig:
    enable: bool
    teacher_model: str
    teacher_weight: str
    apply_to: str  # 'main' | 'aux' | 'both'
    temperature: KDTempConfig
    weights: KDWeightsConfig
    teacher_fp16: bool
    freeze_teacher: bool


@dataclass
class NMSConfig:
    min_confidence: float
    min_iou: float
    max_bbox: int
    pre_topk: Optional[int] = 20000
    multi_label: bool = False
    class_agnostic: bool = False
    size_bias_alpha: float = 0.0


@dataclass
class InferenceConfig:
    task: str
    nms: NMSConfig
    data: DataConfig
    fast_inference: Optional[None]
    save_predict: bool
    render_labels: bool


@dataclass
class ValidationConfig:
    task: str
    nms: NMSConfig
    data: DataConfig

@dataclass
class ExportConfig:
    task: str
    batch_size: int = 1
    opset: int = 13
    simplify: bool = True
    half: bool = False
    dynamic_batch: bool = False
    apply_sigmoid: bool = True
    include_metadata: bool = True
    output_path: Optional[str] = None
    image_size: Optional[List[int]] = None


@dataclass
class TrainConfig:
    task: str
    epoch: int
    data: DataConfig
    optimizer: OptimizerConfig
    loss: LossConfig
    scheduler: SchedulerConfig
    ema: EMAConfig
    validation: ValidationConfig
    resume_ckpt: Optional[str] = None
    kd: Optional[KDConfig] = None


@dataclass
class Config:
    task: Union[TrainConfig, InferenceConfig, ValidationConfig, ExportConfig]
    dataset: DatasetConfig
    model: ModelConfig
    name: str

    device: Union[str, int, List[int]]
    cpu_num: int

    image_size: List[int]

    out_path: str
    exist_ok: bool

    lucky_number: 10
    use_wandb: bool
    use_tensorboard: bool

    weight: Optional[str]


@dataclass
class YOLOLayer(nn.Module):
    source: Union[int, str, List[int]]
    output: bool
    tags: str
    layer_type: str
    usable: bool
    external: Optional[dict]


IDX_TO_ID = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
]
