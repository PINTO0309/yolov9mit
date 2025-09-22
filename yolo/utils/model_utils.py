import os
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from math import exp
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.distributed as dist
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import ListConfig
from torch import Tensor, no_grad
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, _LRScheduler

from yolo.config.config import IDX_TO_ID, NMSConfig, OptimizerConfig, SchedulerConfig
from yolo.model.yolo import YOLO
from yolo.utils.bounding_box_utils import Anc2Box, Vec2Box, bbox_nms, transform_bbox
from yolo.utils.logger import logger


def lerp(start: float, end: float, step: Union[int, float], total: int = 1):
    """
    Linearly interpolates between start and end values.

    start * (1 - step) + end * step

    Parameters:
        start (float): The starting value.
        end (float): The ending value.
        step (int): The current step in the interpolation process.
        total (int): The total number of steps.

    Returns:
        float: The interpolated value.
    """
    return start + (end - start) * step / total


class EMA(Callback):
    def __init__(self, decay: float = 0.9999, tau: float = 2000):
        super().__init__()
        logger.info(":chart_with_upwards_trend: Enable Model EMA")
        self.decay = decay
        self.tau = tau
        self.step = 0
        self.ema_state_dict = None

    def setup(self, trainer, pl_module, stage):
        pl_module.ema = deepcopy(pl_module.model)
        self.tau /= trainer.world_size
        for param in pl_module.ema.parameters():
            param.requires_grad = False

    def on_validation_start(self, trainer: "Trainer", pl_module: "LightningModule"):
        if self.ema_state_dict is None:
            self.ema_state_dict = deepcopy(pl_module.model.state_dict())
        pl_module.ema.load_state_dict(self.ema_state_dict)

    @no_grad()
    def on_train_batch_end(self, trainer: "Trainer", pl_module: "LightningModule", *args, **kwargs) -> None:
        self.step += 1
        decay_factor = self.decay * (1 - exp(-self.step / self.tau))
        if self.ema_state_dict is None:
            # Initialize EMA weights the first time we see a batch. We clone to
            # avoid future in-place updates affecting the model parameters.
            self.ema_state_dict = {k: v.detach().clone() for k, v in pl_module.model.state_dict().items()}

        for key, param in pl_module.model.state_dict().items():
            ema_tensor = self.ema_state_dict[key]
            if isinstance(ema_tensor, torch.Tensor) and ema_tensor.device != param.device:
                ema_tensor = ema_tensor.to(param.device, non_blocking=True)
            self.ema_state_dict[key] = lerp(param.detach(), ema_tensor, decay_factor)

    def state_dict(self) -> dict:
        state = {"step": self.step, "tau": self.tau, "decay": self.decay}
        if self.ema_state_dict is not None:
            state["ema_state_dict"] = {k: v.detach().cpu() for k, v in self.ema_state_dict.items()}
        else:
            state["ema_state_dict"] = None
        return state

    def load_state_dict(self, state: dict) -> None:
        if not state:
            return
        self.step = int(state.get("step", 0))
        self.tau = float(state.get("tau", self.tau))
        self.decay = float(state.get("decay", self.decay))
        ema_state = state.get("ema_state_dict")
        if ema_state is not None:
            # Clone to avoid accidental reference sharing with Lightning internals
            self.ema_state_dict = {k: v.clone() for k, v in ema_state.items()}


class SaveBestWeights(Callback):
    """Save best and last model weights (.pt) during training.

    - At the end of each validation epoch, saves:
        - last.pt: current model/EMA weights
        - best_XXXX_0.0000.pt: when mAP improves; keeps only the latest best file
    - Saves into the same directory as Lightning checkpoints.
    """

    def __init__(self) -> None:
        super().__init__()
        self.best_map = float("-inf")
        self.best_path: Optional[Path] = None
        self.ckpt_dir: Optional[Path] = None

    def _resolve_ckpt_dir(self, trainer: "Trainer") -> Path:
        # Prefer the dirpath from any ModelCheckpoint-like callback if present
        ckpt_dir = None
        checkpoint_cb = getattr(trainer, "checkpoint_callback", None)
        if checkpoint_cb is not None and getattr(checkpoint_cb, "dirpath", None):
            ckpt_dir = Path(checkpoint_cb.dirpath)
        if ckpt_dir is None:
            for cb in getattr(trainer, "callbacks", []):
                dirpath = getattr(cb, "dirpath", None)
                if dirpath:
                    ckpt_dir = Path(dirpath)
                    break
        if ckpt_dir is None:
            log_dir = getattr(trainer, "log_dir", None)
            if log_dir:
                ckpt_dir = Path(log_dir) / "checkpoints"
        if ckpt_dir is None:
            logger_obj = getattr(trainer, "logger", None)
            save_dir = getattr(logger_obj, "save_dir", None)
            if save_dir:
                ckpt_dir = Path(save_dir) / "checkpoints"
        if ckpt_dir is None:
            ckpt_dir = Path(trainer.default_root_dir) / "checkpoints"

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir

    def _ensure_ckpt_dir(self, trainer: "Trainer") -> Path:
        if self.ckpt_dir is None:
            self.ckpt_dir = self._resolve_ckpt_dir(trainer)
        return self.ckpt_dir

    @staticmethod
    def _select_model(pl_module: "LightningModule"):
        model_to_save = getattr(pl_module, "ema", None)
        if model_to_save is None:
            model_to_save = getattr(pl_module, "model", pl_module)
        return model_to_save

    @staticmethod
    def _export_official_state_dict(module) -> "OrderedDict[str, torch.Tensor]":
        state_dict = module.state_dict()
        flat = OrderedDict()
        for key, value in state_dict.items():
            new_key = key[len("model.") :] if key.startswith("model.") else key
            flat[new_key] = value.detach().to("cpu")
        return flat

    @staticmethod
    def _extract_variant(pl_module: "LightningModule") -> str:
        cfg = getattr(pl_module, "cfg", None)
        model_cfg = getattr(cfg, "model", None) if cfg is not None else None
        name = getattr(model_cfg, "name", None)
        if not name:
            return "unknown"
        name_str = str(name).lower()
        return name_str.split("-")[-1] if "-" in name_str else name_str

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.item())
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _extract_map(self, metrics: Mapping[str, Any]) -> Optional[float]:
        # Primary key
        current = self._to_float(metrics.get("map"))
        if current is not None:
            return current

        # Fallback to human-readable alias if present
        alt = self._to_float(metrics.get("PyCOCO/AP @ .5:.95"))
        if alt is not None:
            return alt / 100.0 if alt > 1.0 else alt

        return None

    def update_from_metrics(
        self, trainer: "Trainer", pl_module: "LightningModule", metrics: Mapping[str, Any] | None
    ) -> None:
        metrics = metrics or {}
        # Ensure directory exists before attempting any writes
        ckpt_dir = self._ensure_ckpt_dir(trainer)

        model_to_save = self._select_model(pl_module)
        official_state = self._export_official_state_dict(model_to_save)

        # Always refresh last.pt so downstream export tools see the latest weights
        torch.save(official_state, ckpt_dir / "last.pt")

        current_map = self._extract_map(metrics)
        if current_map is None:
            return

        if current_map > self.best_map:
            epoch = int(getattr(trainer, "current_epoch", 0))
            variant = self._extract_variant(pl_module)
            best_name = f"best_{variant}_{epoch:04d}_{current_map:.4f}.pt"
            best_path = ckpt_dir / best_name

            torch.save(official_state, best_path)

            if self.best_path is not None and self.best_path.exists():
                try:
                    self.best_path.unlink()
                except Exception:
                    pass

            self.best_map = current_map
            self.best_path = best_path

    @rank_zero_only
    def on_fit_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        self.ckpt_dir = self._resolve_ckpt_dir(trainer)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        metrics = getattr(trainer, "callback_metrics", {}) or {}
        if not isinstance(metrics, Mapping):
            metrics = dict(metrics)
        self.update_from_metrics(trainer, pl_module, metrics)


def create_optimizer(model: YOLO, optim_cfg: OptimizerConfig) -> Optimizer:
    """Create an optimizer for the given model parameters based on the configuration.

    Returns:
        An instance of the optimizer configured according to the provided settings.
    """
    optimizer_class: Type[Optimizer] = getattr(torch.optim, optim_cfg.type)

    bias_params = [p for name, p in model.named_parameters() if "bias" in name]
    norm_params = [p for name, p in model.named_parameters() if "weight" in name and "bn" in name]
    conv_params = [p for name, p in model.named_parameters() if "weight" in name and "bn" not in name]

    model_parameters = [
        {"params": bias_params, "momentum": 0.937, "weight_decay": 0},
        {"params": conv_params, "momentum": 0.937},
        {"params": norm_params, "momentum": 0.937, "weight_decay": 0},
    ]

    def next_epoch(self, batch_num, epoch_idx):
        self.min_lr = self.max_lr
        self.max_lr = [param["lr"] for param in self.param_groups]
        # TODO: load momentum from config instead a fix number
        #       0.937: Start Momentum
        #       0.8  : Normal Momemtum
        #       3    : The warm up epoch num
        self.min_mom = lerp(0.8, 0.937, min(epoch_idx, 3), 3)
        self.max_mom = lerp(0.8, 0.937, min(epoch_idx + 1, 3), 3)
        self.batch_num = batch_num
        self.batch_idx = 0

    def next_batch(self):
        self.batch_idx += 1
        lr_dict = dict()
        for lr_idx, param_group in enumerate(self.param_groups):
            min_lr, max_lr = self.min_lr[lr_idx], self.max_lr[lr_idx]
            param_group["lr"] = lerp(min_lr, max_lr, self.batch_idx, self.batch_num)
            param_group["momentum"] = lerp(self.min_mom, self.max_mom, self.batch_idx, self.batch_num)
            lr_dict[f"LR/{lr_idx}"] = param_group["lr"]
            lr_dict[f"momentum/{lr_idx}"] = param_group["momentum"]
        return lr_dict

    optimizer_class.next_batch = next_batch
    optimizer_class.next_epoch = next_epoch

    optimizer = optimizer_class(model_parameters, **optim_cfg.args)
    optimizer.max_lr = [0.1, 0, 0]
    return optimizer


def create_scheduler(optimizer: Optimizer, schedule_cfg: SchedulerConfig) -> _LRScheduler:
    """Create a learning rate scheduler for the given optimizer based on the configuration.

    Returns:
        An instance of the scheduler configured according to the provided settings.
    """
    scheduler_class: Type[_LRScheduler] = getattr(torch.optim.lr_scheduler, schedule_cfg.type)
    schedule = scheduler_class(optimizer, **schedule_cfg.args)
    if hasattr(schedule_cfg, "warmup"):
        wepoch = schedule_cfg.warmup.epochs
        lambda1 = lambda epoch: (epoch + 1) / wepoch if epoch < wepoch else 1
        lambda2 = lambda epoch: 10 - 9 * ((epoch + 1) / wepoch) if epoch < wepoch else 1
        warmup_schedule = LambdaLR(optimizer, lr_lambda=[lambda2, lambda1, lambda1])
        schedule = SequentialLR(optimizer, schedulers=[warmup_schedule, schedule], milestones=[wepoch - 1])
    return schedule


def initialize_distributed() -> None:
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    logger.info(f"🔢 Initialized process group; rank: {rank}, size: {world_size}")
    return local_rank


def get_device(device_spec: Union[str, int, List[int]]) -> torch.device:
    ddp_flag = False
    if isinstance(device_spec, (list, ListConfig)):
        ddp_flag = True
        device_spec = initialize_distributed()
    if torch.cuda.is_available() and "cuda" in str(device_spec):
        return torch.device(device_spec), ddp_flag
    if not torch.cuda.is_available():
        if device_spec != "cpu":
            logger.warning(f"❎ Device spec: {device_spec} not support, Choosing CPU instead")
        return torch.device("cpu"), False

    device = torch.device(device_spec)
    return device, ddp_flag


class PostProcess:
    """
    TODO: function document
    scale back the prediction and do nms for pred_bbox
    """

    def __init__(self, converter: Union[Vec2Box, Anc2Box], nms_cfg: NMSConfig) -> None:
        self.converter = converter
        self.nms = nms_cfg

    def __call__(
        self, predict, rev_tensor: Optional[Union[Tensor, Dict[str, Tensor], Sequence[Dict[str, Any]]]] = None,
        image_size: Optional[List[int]] = None,
    ) -> List[Tensor]:
        if image_size is not None:
            self.converter.update(image_size)
        prediction = self.converter(predict["Main"])
        pred_class, _, pred_bbox = prediction[:3]
        pred_conf = prediction[3] if len(prediction) == 4 else None
        if rev_tensor is not None:
            gains, shifts = self._parse_letterbox_meta(rev_tensor, pred_bbox.device, pred_bbox.dtype)
            if gains is not None and shifts is not None and gains.numel() and shifts.numel():
                gains = gains.clamp_min(1e-6)
                pred_bbox = (pred_bbox - shifts[:, None, :]) / gains[:, None, :]
        pred_bbox = bbox_nms(pred_class, pred_bbox, self.nms, pred_conf)
        return pred_bbox

    @staticmethod
    def _to_tensor(value, device, dtype) -> Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=dtype)
        return torch.tensor(value, device=device, dtype=dtype)

    def _parse_letterbox_meta(
        self,
        meta: Union[Tensor, Dict[str, Any], Sequence[Dict[str, Any]]],
        device,
        dtype,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        if isinstance(meta, torch.Tensor):
            gains = meta[:, :1].repeat(1, 4).to(device=device, dtype=dtype)
            shifts = meta[:, 1:].to(device=device, dtype=dtype)
            return gains, shifts

        if isinstance(meta, dict):
            ratio = meta.get("ratio", (1.0, 1.0))
            pad = meta.get("pad", (0.0, 0.0))
            ratio_t = self._to_tensor(ratio, device, dtype)
            pad_t = self._to_tensor(pad, device, dtype)
            if ratio_t.ndim == 1:
                ratio_t = ratio_t.unsqueeze(0)
            if pad_t.ndim == 1:
                pad_t = pad_t.unsqueeze(0)
            gains = torch.stack([ratio_t[:, 0], ratio_t[:, 1], ratio_t[:, 0], ratio_t[:, 1]], dim=-1)
            shifts = torch.stack([pad_t[:, 0], pad_t[:, 1], pad_t[:, 0], pad_t[:, 1]], dim=-1)
            return gains, shifts

        if isinstance(meta, (list, tuple)):
            if not meta:
                empty = torch.zeros((0, 4), device=device, dtype=dtype)
                return empty, empty
            ratio = torch.tensor([item.get("ratio", (1.0, 1.0)) for item in meta], device=device, dtype=dtype)
            pad = torch.tensor([item.get("pad", (0.0, 0.0)) for item in meta], device=device, dtype=dtype)
            gains = torch.stack([ratio[:, 0], ratio[:, 1], ratio[:, 0], ratio[:, 1]], dim=-1)
            shifts = torch.stack([pad[:, 0], pad[:, 1], pad[:, 0], pad[:, 1]], dim=-1)
            return gains, shifts

        raise TypeError(f"Unsupported letterbox metadata type: {type(meta)!r}")


def collect_prediction(predict_json: List, local_rank: int) -> List:
    """
    Collects predictions from all distributed processes and gathers them on the main process (rank 0).

    Args:
        predict_json (List): The prediction data (can be of any type) generated by the current process.
        local_rank (int): The rank of the current process. Typically, rank 0 is the main process.

    Returns:
        List: The combined list of predictions from all processes if on rank 0, otherwise predict_json.
    """
    if dist.is_initialized() and local_rank == 0:
        all_predictions = [None for _ in range(dist.get_world_size())]
        dist.gather_object(predict_json, all_predictions, dst=0)
        predict_json = [item for sublist in all_predictions for item in sublist]
    elif dist.is_initialized():
        dist.gather_object(predict_json, None, dst=0)
    return predict_json


def predicts_to_json(img_paths, predicts, rev_tensor):
    """Convert predictions back to COCO json format using letterbox metadata."""

    def _expand_letterbox(meta, count):
        if meta is None:
            return [{"ratio": (1.0, 1.0), "pad": (0.0, 0.0)} for _ in range(count)]

        if isinstance(meta, dict):
            def _to_tensor(val, dtype=torch.float32):
                if val is None:
                    return None
                if isinstance(val, torch.Tensor):
                    tensor = val.detach().cpu().to(dtype=dtype)
                else:
                    tensor = torch.tensor(val, dtype=dtype)
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)
                return tensor

            ratio_t = _to_tensor(meta.get("ratio"))
            pad_t = _to_tensor(meta.get("pad"))
            size_t = _to_tensor(meta.get("size"))
            num = 0
            for tensor in (ratio_t, pad_t, size_t):
                if tensor is not None:
                    num = max(num, tensor.size(0))
            num = max(num, 1)
            entries = []
            for idx in range(num):
                entry = {
                    "ratio": tuple(ratio_t[idx].tolist()) if ratio_t is not None else (1.0, 1.0),
                    "pad": tuple(pad_t[idx].tolist()) if pad_t is not None else (0.0, 0.0),
                }
                if size_t is not None:
                    entry["size"] = tuple(size_t[idx].tolist())
                entries.append(entry)
            if len(entries) < count:
                entries.extend(entries[-1] for _ in range(count - len(entries)))
            return entries[:count]

        if isinstance(meta, torch.Tensor):
            scales = meta[:, 0].detach().cpu()
            pads = meta[:, 1:].detach().cpu().view(meta.size(0), 4)
            return [
                {
                    "ratio": (float(scale), float(scale)),
                    "pad": (float(pad[0]), float(pad[1])),
                }
                for scale, pad in zip(scales, pads)
            ]

        if isinstance(meta, (list, tuple)):
            return list(meta)

        raise TypeError(f"Unsupported letterbox metadata type: {type(meta)!r}")

    rev_list = _expand_letterbox(rev_tensor, len(predicts))
    batch_json = []
    for img_path, bboxes, info in zip(img_paths, predicts, rev_list):
        if isinstance(bboxes, torch.Tensor):
            boxes_tensor = bboxes.detach().cpu().clone()
        else:
            boxes_tensor = torch.as_tensor(bboxes, dtype=torch.float32)

        ratio = torch.tensor(info.get("ratio", (1.0, 1.0)), dtype=boxes_tensor.dtype)
        pad = torch.tensor(info.get("pad", (0.0, 0.0)), dtype=boxes_tensor.dtype)
        ratio = ratio.clamp_min(1e-6)

        boxes_tensor[:, [1, 3]] = (boxes_tensor[:, [1, 3]] - pad[0]) / ratio[0]
        boxes_tensor[:, [2, 4]] = (boxes_tensor[:, [2, 4]] - pad[1]) / ratio[1]
        boxes_tensor[:, 1:5] = transform_bbox(boxes_tensor[:, 1:5], "xyxy -> xywh")

        for cls, *pos, conf in boxes_tensor:
            bbox = {
                "image_id": int(Path(img_path).stem),
                "category_id": IDX_TO_ID[int(cls)],
                "bbox": [float(p) for p in pos],
                "score": float(conf),
            }
            batch_json.append(bbox)
    return batch_json
