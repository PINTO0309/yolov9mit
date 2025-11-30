"""
Module for initializing logging tools used in machine learning and data processing.
Supports integration with Weights & Biases (wandb), Loguru, TensorBoard, and other
logging frameworks as needed.

This setup ensures consistent logging across various platforms, facilitating
effective monitoring and debugging.

Example:
    from tools.logger import custom_logger
    custom_logger()
"""

import logging
import os
import shutil
from collections import deque
from contextlib import suppress
from logging import FileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import wandb
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, RichModelSummary, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import CustomProgress
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import ListConfig, OmegaConf
from rich import get_console, reconfigure
from rich.console import Console, Group
from rich.logging import RichHandler
from rich.table import Table
from rich.text import Text
from torch import Tensor
from torch.nn import ModuleList
from typing_extensions import override

from yolo.config.config import Config, YOLOLayer
from yolo.model.yolo import YOLO
from yolo.utils.logger import logger
from yolo.utils.model_utils import EMA, SaveBestWeights
from yolo.utils.solver_utils import make_ap_table
from yolo.tools.drawer import draw_bboxes


# TODO: should be moved to correct position
def set_seed(seed):
    seed_everything(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class YOLOCustomProgress(CustomProgress):
    def get_renderable(self):
        renderable = Group(*self.get_renderables())
        if hasattr(self, "table"):
            renderable = Group(*self.get_renderables(), self.table)
        return renderable


class YOLORichProgressBar(RichProgressBar):
    @override
    @rank_zero_only
    def _init_progress(self, trainer: "Trainer") -> None:
        if self.is_enabled and (self.progress is None or self._progress_stopped):
            self._reset_progress_bar_ids()
            reconfigure(**self._console_kwargs)
            self._console = Console()
            # Guard against empty live stack in some rich versions
            try:
                self._console.clear_live()
            except IndexError:
                pass
            self.progress = YOLOCustomProgress(
                *self.configure_columns(trainer),
                auto_refresh=False,
                disable=self.is_disabled,
                console=self._console,
            )
            self.progress.start()

            self._progress_stopped = False

            self.max_result = 0
            self.past_results = deque(maxlen=5)
            self.progress.table = Table()

    @override
    def _get_train_description(self, current_epoch: int) -> str:
        return Text("[cyan]Train [white]|")

    @override
    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        self._init_progress(trainer)
        num_epochs = trainer.max_epochs - 1
        self.task_epoch = self._add_task(
            total_batches=num_epochs,
            description=f"[cyan]Start Training {num_epochs} epochs",
        )
        # When resuming from a checkpoint, keep epoch progress in sync with the trainer
        current_epoch = int(getattr(trainer, "current_epoch", 0))
        if current_epoch > 0:
            # Cap to the configured total to avoid overshooting in edge cases
            completed = min(current_epoch, num_epochs)
            self.progress.update(self.task_epoch, completed=completed)
        self.max_result = 0
        self.past_results.clear()

    @override
    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch: Any, batch_idx: int):
        self._update(self.train_progress_bar_id, batch_idx + 1)
        self._update_metrics(trainer, pl_module)
        epoch_descript = "[cyan]Train [white]|"
        batch_descript = "[green]Train [white]|"
        metrics = self.get_metrics(trainer, pl_module)
        # Some Lightning versions or logger configs may omit 'v_num'
        metrics.pop("v_num", None)
        for metrics_name, metrics_val in metrics.items():
            if "Loss_step" in metrics_name:
                epoch_descript += f"{metrics_name.removesuffix('_step').split('/')[1]: ^9}|"
                batch_descript += f"   {metrics_val:2.2f}  |"

        self.progress.update(self.task_epoch, advance=1 / self.total_train_batches, description=epoch_descript)
        self.progress.update(self.train_progress_bar_id, description=batch_descript)
        self.refresh()

    @override
    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self.is_disabled:
            return
        if trainer.sanity_checking:
            self._update(self.val_sanity_progress_bar_id, batch_idx + 1)
        elif self.val_progress_bar_id is not None:
            self._update(self.val_progress_bar_id, batch_idx + 1)
            _, mAP = outputs
            mAP_desc = f" mAP :{mAP['map']*100:6.2f} | mAP50 :{mAP['map_50']*100:6.2f} |"
            self.progress.update(self.val_progress_bar_id, description=f"[green]Valid [white]|{mAP_desc}")
        self.refresh()

    @override
    @rank_zero_only
    def on_train_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        self._update_metrics(trainer, pl_module)
        self.progress.remove_task(self.train_progress_bar_id)
        self.train_progress_bar_id = None

    @override
    @rank_zero_only
    def on_validation_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        if trainer.state.fn == "fit":
            self._update_metrics(trainer, pl_module)
        self.reset_dataloader_idx_tracker()
        all_metrics = self.get_metrics(trainer, pl_module)

        ap_ar_list = [
            key
            for key in all_metrics.keys()
            if key.startswith(("map", "mar")) and not key.endswith(("_step", "_epoch"))
        ]
        score = np.array([all_metrics[key] for key in ap_ar_list]) * 100

        self.progress.table, ap_main = make_ap_table(score, self.past_results, self.max_result, trainer.current_epoch)
        self.max_result = np.maximum(score, self.max_result)
        self.past_results.append((trainer.current_epoch, ap_main))

    @override
    def refresh(self) -> None:
        if self.progress:
            self.progress.refresh()

    @property
    def validation_description(self) -> str:
        return "[green]Validation"

class YOLORichModelSummary(RichModelSummary):
    @staticmethod
    @override
    def summarize(
        summary_data: List[Tuple[str, List[str]]],
        total_parameters: int,
        trainable_parameters: int,
        model_size: float,
        **summarize_kwargs: Any,
    ) -> None:
        from lightning.pytorch.utilities.model_summary import get_human_readable_count

        console = get_console()

        header_style: str = summarize_kwargs.get("header_style", "bold magenta")
        table = Table(header_style=header_style)
        table.add_column(" ", style="dim")
        table.add_column("Name", justify="left", no_wrap=True)
        table.add_column("Type")
        table.add_column("Params", justify="right")
        table.add_column("Mode")

        column_names = list(zip(*summary_data))[0]

        for column_name in ["In sizes", "Out sizes"]:
            if column_name in column_names:
                table.add_column(column_name, justify="right", style="white")

        rows = list(zip(*(arr[1] for arr in summary_data)))
        for row in rows:
            table.add_row(*row)

        console.print(table)

        parameters = []
        for param in [trainable_parameters, total_parameters - trainable_parameters, total_parameters, model_size]:
            parameters.append("{:<{}}".format(get_human_readable_count(int(param)), 10))

        grid = Table(header_style=header_style)
        table.add_column(" ", style="dim")
        grid.add_column("[bold]Attributes[/]")
        grid.add_column("Value")

        grid.add_row("[bold]Trainable params[/]", f"{parameters[0]}")
        grid.add_row("[bold]Non-trainable params[/]", f"{parameters[1]}")
        grid.add_row("[bold]Total params[/]", f"{parameters[2]}")
        grid.add_row("[bold]Total estimated model params size (MB)[/]", f"{parameters[3]}")
        # Lightning versions may or may not provide total_training_modes; default safely
        total_training_modes = summarize_kwargs.get("total_training_modes")
        if not isinstance(total_training_modes, dict):
            total_training_modes = {"train": 0, "eval": 0}

        grid.add_row("[bold]Modules in train mode[/]", f"{total_training_modes.get('train', 0)}")
        grid.add_row("[bold]Modules in eval mode[/]", f"{total_training_modes.get('eval', 0)}")

        console.print(grid)


class ImageLogger(Callback):
    def on_validation_batch_end(self, trainer: Trainer, pl_module, outputs, batch, batch_idx) -> None:
        if batch_idx != 0:
            return
        batch_size, images, targets, rev_tensor, img_paths = batch
        predicts, _ = outputs
        gt_boxes = targets[0] if targets.ndim == 3 else targets
        pred_boxes = predicts[0] if isinstance(predicts, list) else predicts
        images = [images[0]]
        step = trainer.current_epoch
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                logger.log_image("Input Image", images, step=step)
                logger.log_image("Ground Truth", images, step=step, boxes=[log_bbox(gt_boxes)])
                logger.log_image("Prediction", images, step=step, boxes=[log_bbox(pred_boxes)])


class ValidationImageSaver(Callback):
    """
    Save a fixed number of validation images with predicted boxes each epoch and
    keep only the most recent epochs to limit disk usage.
    """

    def __init__(self, *, max_images: int = 10, keep_epochs: int = 10):
        super().__init__()
        self.max_images = max_images
        self.keep_epochs = keep_epochs
        self._buffer: list[tuple[Union[str, Path], torch.Tensor]] = []
        self._version_dir: Optional[Path] = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self._version_dir is None:
            self._version_dir = self._resolve_version_dir(trainer)

    @rank_zero_only
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._buffer.clear()

    @rank_zero_only
    def on_validation_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx, dataloader_idx: int = 0
    ) -> None:
        if trainer.sanity_checking or len(self._buffer) >= self.max_images:
            return
        try:
            predicts, _ = outputs
        except Exception:
            return
        if predicts is None:
            return
        batch_size, images, targets, rev_tensor, img_paths = batch
        predict_list = list(predicts) if isinstance(predicts, (list, tuple)) else [predicts]

        for idx, pred in enumerate(predict_list):
            if len(self._buffer) >= self.max_images:
                break
            if not isinstance(pred, torch.Tensor):
                continue
            adj = self._unletterbox(pred, rev_tensor, idx)
            self._buffer.append((img_paths[idx], adj.detach().cpu()))

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self._buffer:
            return
        version_dir = self._version_dir or self._resolve_version_dir(trainer)
        epoch_dir = version_dir / f"{int(trainer.current_epoch):04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        idx2label = getattr(getattr(pl_module.cfg, "dataset", None), "class_list", None)

        for idx, (img_path, pred) in enumerate(self._buffer[: self.max_images]):
            try:
                with Image.open(img_path).convert("RGB") as img:
                    drawn = draw_bboxes(img, pred, idx2label=idx2label, fill=False, draw_labels=True)
                drawn.save(epoch_dir / f"val_{idx:02d}.png")
            except Exception as exc:
                logger.warning(f":warning: Failed to save validation image {idx}: {exc}")

        self._prune_old_epochs(version_dir)

    def _resolve_version_dir(self, trainer: Trainer) -> Path:
        def _from_logger(lg) -> Optional[Path]:
            log_dir = getattr(lg, "log_dir", None)
            if log_dir:
                return Path(log_dir)
            save_dir = getattr(lg, "save_dir", None)
            version = getattr(lg, "version", None)
            name = getattr(lg, "name", None) or "lightning_logs"
            if save_dir is not None and version is not None:
                return Path(save_dir) / name / f"version_{version}"
            return None

        loggers = []
        if hasattr(trainer, "loggers") and trainer.loggers:
            loggers.extend(trainer.loggers if isinstance(trainer.loggers, (list, tuple)) else [trainer.loggers])
        if trainer.logger and trainer.logger not in loggers:
            loggers.append(trainer.logger)

        for lg in loggers:
            path = _from_logger(lg)
            if path is not None:
                return path

        version = getattr(getattr(trainer, "logger", None), "version", 0)
        fallback = Path(trainer.default_root_dir) / "lightning_logs"
        if version is not None:
            fallback = fallback / f"version_{version}"
        return fallback

    def _unletterbox(self, pred: torch.Tensor, rev_tensor, idx: int) -> torch.Tensor:
        """
        Map boxes back to the original image coordinate space using letterbox metadata.
        """
        if not isinstance(pred, torch.Tensor):
            return pred
        if rev_tensor is None:
            return pred
        try:
            ratio = rev_tensor["ratio"][idx] if isinstance(rev_tensor, dict) else rev_tensor[idx, :2]
            pad = rev_tensor["pad"][idx] if isinstance(rev_tensor, dict) else rev_tensor[idx, 2:4]
        except Exception:
            return pred

        ratio = torch.as_tensor(ratio, dtype=pred.dtype, device=pred.device)
        pad = torch.as_tensor(pad, dtype=pred.dtype, device=pred.device)
        if ratio.numel() != 2 or pad.numel() != 2:
            return pred

        gains = torch.stack([ratio[0], ratio[1], ratio[0], ratio[1]])
        shifts = torch.stack([pad[0], pad[1], pad[0], pad[1]])

        adj = pred.clone()
        adj[:, 1:5] = (adj[:, 1:5] - shifts) / gains
        return adj

    def _prune_old_epochs(self, version_dir: Path) -> None:
        try:
            epoch_dirs = [p for p in version_dir.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 4]
        except FileNotFoundError:
            return
        keep = set(sorted(epoch_dirs, key=lambda p: int(p.name))[-self.keep_epochs :])
        for p in epoch_dirs:
            if p not in keep:
                with suppress(Exception):
                    shutil.rmtree(p)


def setup_logger(logger_name, quite=False):
    class EmojiFormatter(logging.Formatter):
        def format(self, record, emoji=":high_voltage:"):
            return f"{emoji} {super().format(record)}"

    rich_handler = RichHandler(markup=True)
    rich_handler.setFormatter(EmojiFormatter("%(message)s"))
    rich_logger = logging.getLogger(logger_name)
    if rich_logger:
        rich_logger.handlers.clear()
        rich_logger.addHandler(rich_handler)
        if quite:
            rich_logger.setLevel(logging.ERROR)

    coco_logger = logging.getLogger("faster_coco_eval.core.cocoeval")
    coco_logger.setLevel(logging.ERROR)


def setup(cfg: Config):
    quite = hasattr(cfg, "quite")
    setup_logger("lightning.fabric", quite=quite)
    setup_logger("lightning.pytorch", quite=quite)

    def custom_wandb_log(string="", level=int, newline=True, repeat=True, prefix=True, silent=False):
        if silent:
            return
        for line in string.split("\n"):
            logger.info(Text.from_ansi(":globe_with_meridians: " + line))

    wandb.errors.term._log = custom_wandb_log

    save_path = validate_log_directory(cfg, cfg.name)
    if save_path is None:
        # Non-zero ranks skip the rank_zero_only body; ensure they still know the log directory
        save_path = Path(cfg.out_path, cfg.task.task) / cfg.name
        save_path.mkdir(parents=True, exist_ok=True)

    progress, loggers = [], []

    if hasattr(cfg.task, "ema") and cfg.task.ema.enable:
        progress.append(EMA(cfg.task.ema.decay))
    ckpt_callback = ModelCheckpoint(filename="{epoch:04d}_{step:07d}")
    ckpt_callback.CHECKPOINT_JOIN_CHAR = "_"
    ckpt_callback.CHECKPOINT_EQUALS_CHAR = "_"
    progress.append(ckpt_callback)
    # Save best and last .pt files alongside .ckpt directory
    progress.append(SaveBestWeights())
    if quite:
        logger.setLevel(logging.ERROR)
        return progress, loggers, save_path

    progress.append(YOLORichProgressBar())
    progress.append(YOLORichModelSummary())
    progress.append(ImageLogger())
    progress.append(ValidationImageSaver())

    is_rank_zero = os.getenv("RANK", "0") == "0"
    if cfg.use_tensorboard and is_rank_zero:
        loggers.append(TensorBoardLogger(log_graph="all", save_dir=str(save_path)))
    if cfg.use_wandb and is_rank_zero:
        wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        loggers.append(WandbLogger(project="YOLO", name=cfg.name, save_dir=str(save_path), id=None, config=wandb_cfg))

    return progress, loggers, save_path


def log_model_structure(model: Union[ModuleList, YOLOLayer, YOLO]):
    if isinstance(model, YOLO):
        model = model.model
    console = Console()
    table = Table(title="Model Layers")

    table.add_column("Index", justify="center")
    table.add_column("Layer Type", justify="center")
    table.add_column("Tags", justify="center")
    table.add_column("Params", justify="right")
    table.add_column("Channels (IN->OUT)", justify="center")

    for idx, layer in enumerate(model, start=1):
        layer_param = sum(x.numel() for x in layer.parameters())  # number parameters
        in_channels, out_channels = getattr(layer, "in_c", None), getattr(layer, "out_c", None)
        if in_channels and out_channels:
            if isinstance(in_channels, (list, ListConfig)):
                in_channels = "M"
            if isinstance(out_channels, (list, ListConfig)):
                out_channels = "M"
            channels = f"{str(in_channels): >4} -> {str(out_channels): >4}"
        else:
            channels = "-"
        table.add_row(str(idx), layer.layer_type, layer.tags, f"{layer_param:,}", channels)
    console.print(table)


@rank_zero_only
def validate_log_directory(cfg: Config, exp_name: str) -> Path:
    base_path = Path(cfg.out_path, cfg.task.task)
    save_path = base_path / exp_name

    if not cfg.exist_ok:
        index = 1
        old_exp_name = exp_name
        while save_path.is_dir():
            exp_name = f"{old_exp_name}{index}"
            save_path = base_path / exp_name
            index += 1
        if index > 1:
            logger.opt(colors=True).warning(
                f"🔀 Experiment directory exists! Changed <red>{old_exp_name}</> to <green>{exp_name}</>"
            )

    save_path.mkdir(parents=True, exist_ok=True)
    if not getattr(cfg, "quite", False):
        logger.info(f"📄 Created log folder: [blue b u]{save_path}[/]")
    logger.addHandler(FileHandler(save_path / "output.log"))
    return save_path


def log_bbox(
    bboxes: Tensor, class_list: Optional[List[str]] = None, image_size: Tuple[int, int] = (640, 640)
) -> List[dict]:
    """
    Convert bounding boxes tensor to a list of dictionaries for logging, normalized by the image size.

    Args:
        bboxes (Tensor): Bounding boxes with shape (N, 5) or (N, 6), where each box is [class_id, x_min, y_min, x_max, y_max, (confidence)].
        class_list (Optional[List[str]]): List of class names. Defaults to None.
        image_size (Tuple[int, int]): The size of the image, used for normalization. Defaults to (640, 640).

    Returns:
        List[dict]: List of dictionaries containing normalized bounding box information.
    """
    bbox_list = []
    scale_tensor = torch.Tensor([1, *image_size, *image_size]).to(bboxes.device)
    normalized_bboxes = bboxes[:, :5] / scale_tensor
    for bbox in normalized_bboxes:
        class_id, x_min, y_min, x_max, y_max, *conf = [float(val) for val in bbox]
        if class_id == -1:
            break
        bbox_entry = {
            "position": {"minX": x_min, "maxX": x_max, "minY": y_min, "maxY": y_max},
            "class_id": int(class_id),
        }
        if class_list:
            bbox_entry["box_caption"] = class_list[int(class_id)]
        if conf:
            bbox_entry["scores"] = {"confidence": conf[0]}
        bbox_list.append(bbox_entry)

    return {"predictions": {"box_data": bbox_list}}
