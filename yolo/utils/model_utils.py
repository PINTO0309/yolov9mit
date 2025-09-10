import os
from copy import deepcopy
from math import exp
from pathlib import Path
from typing import List, Optional, Type, Union

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
        for key, param in pl_module.model.state_dict().items():
            self.ema_state_dict[key] = lerp(param.detach(), self.ema_state_dict[key], decay_factor)


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

    @rank_zero_only
    def on_fit_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        self.ckpt_dir = self._resolve_ckpt_dir(trainer)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        # Ensure checkpoint directory is ready
        if self.ckpt_dir is None:
            self.ckpt_dir = self._resolve_ckpt_dir(trainer)

        # Determine current weights to save (prefer EMA if available)
        model_to_save = getattr(pl_module, "ema", None)
        if model_to_save is None:
            model_to_save = getattr(pl_module, "model", pl_module)

        # Helper: export to official flat state_dict without leading 'model.'
        def export_official_state_dict(module) -> "OrderedDict[str, torch.Tensor]":
            sd = module.state_dict()
            flat = {}
            for k, v in sd.items():
                if k.startswith("model."):
                    nk = k[len("model."):]
                else:
                    nk = k
                flat[nk] = v.detach().to("cpu")
            # Preserve insertion order
            from collections import OrderedDict
            return OrderedDict(flat)

        # Save last.pt every validation epoch in official format
        last_path = self.ckpt_dir / "last.pt"
        torch.save(export_official_state_dict(model_to_save), last_path)

        # Get current mAP from logged metrics
        metrics = getattr(trainer, "callback_metrics", {}) or {}
        current_map = metrics.get("map")
        try:
            current_map = float(current_map) if current_map is not None else None
        except Exception:
            current_map = None

        if current_map is None:
            return

        # If new best, save best_{variant}_XXXX_0.0000.pt and remove previous best
        if current_map > self.best_map:
            epoch = int(getattr(trainer, "current_epoch", 0))
            # Extract variant from cfg.model.name (e.g., v9-t -> t)
            def extract_variant(module):
                cfg = getattr(module, "cfg", None)
                model_obj = getattr(cfg, "model", None) if cfg is not None else None
                name = getattr(model_obj, "name", None)
                if not name:
                    return "unknown"
                s = str(name).lower()
                if "-" in s:
                    return s.split("-")[-1]
                return s

            variant = extract_variant(pl_module)
            best_name = f"best_{variant}_{epoch:04d}_{current_map:.4f}.pt"
            best_path = self.ckpt_dir / best_name

            torch.save(export_official_state_dict(model_to_save), best_path)

            # Remove previous best if exists
            if self.best_path is not None and self.best_path.exists():
                try:
                    self.best_path.unlink()
                except Exception:
                    pass

            self.best_map = current_map
            self.best_path = best_path


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
        self, predict, rev_tensor: Optional[Tensor] = None, image_size: Optional[List[int]] = None
    ) -> List[Tensor]:
        if image_size is not None:
            self.converter.update(image_size)
        prediction = self.converter(predict["Main"])
        pred_class, _, pred_bbox = prediction[:3]
        pred_conf = prediction[3] if len(prediction) == 4 else None
        if rev_tensor is not None:
            pred_bbox = (pred_bbox - rev_tensor[:, None, 1:]) / rev_tensor[:, 0:1, None]
        pred_bbox = bbox_nms(pred_class, pred_bbox, self.nms, pred_conf)
        return pred_bbox


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
    """
    TODO: function document
    turn a batch of imagepath and predicts(n x 6 for each image) to a List of diction(Detection output)
    """
    batch_json = []
    for img_path, bboxes, box_reverse in zip(img_paths, predicts, rev_tensor):
        scale, shift = box_reverse.split([1, 4])
        bboxes = bboxes.clone()
        bboxes[:, 1:5] = (bboxes[:, 1:5] - shift[None]) / scale[None]
        bboxes[:, 1:5] = transform_bbox(bboxes[:, 1:5], "xyxy -> xywh")
        for cls, *pos, conf in bboxes:
            bbox = {
                "image_id": int(Path(img_path).stem),
                "category_id": IDX_TO_ID[int(cls)],
                "bbox": [float(p) for p in pos],
                "score": float(conf),
            }
            batch_json.append(bbox)
    return batch_json
