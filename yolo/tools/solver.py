from collections.abc import Mapping, Sequence
from contextlib import suppress, nullcontext
from dataclasses import asdict, is_dataclass
from math import ceil
from pathlib import Path
import traceback

from lightning import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.detection import MeanAveragePrecision
import torch

from yolo.config.config import Config
from yolo.model.yolo import create_model
from yolo.tools.data_loader import create_dataloader
from yolo.tools.drawer import draw_bboxes
from yolo.tools.loss_functions import create_loss_function
from yolo.utils.bounding_box_utils import create_converter, to_metrics_format
from yolo.utils.logger import logger
from yolo.utils.model_utils import PostProcess, SaveBestWeights, create_optimizer, create_scheduler
from yolo.utils.deploy_utils import FastModelLoader


class BaseModel(LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.model = create_model(cfg.model, class_num=cfg.dataset.class_num, weight_path=cfg.weight)

    def forward(self, x):
        return self.model(x)


class ValidateModel(BaseModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        if self.cfg.task.task == "validation":
            self.validation_cfg = self.cfg.task
        else:
            self.validation_cfg = self.cfg.task.validation
        # Compute per-class metrics only if requested to avoid overhead
        class_metrics = getattr(self.validation_cfg, "print_map_per_class", False)
        self.metric = MeanAveragePrecision(
            iou_type="bbox",
            box_format="xyxy",
            backend="faster_coco_eval",
            class_metrics=class_metrics,
        )
        self.metric.warn_on_many_detections = False
        self.val_loader = create_dataloader(self.validation_cfg.data, self.cfg.dataset, self.validation_cfg.task)
        self.ema = self.model
        self._warn_missing_val_loss = False

    def setup(self, stage):
        self.vec2box = create_converter(
            self.cfg.model.name,
            model=self.model,
            anchor_cfg=self.cfg.model.anchor,
            image_size=self.cfg.image_size,
            device=self.device,
            class_num=self.cfg.dataset.class_num,
        )
        self.post_process = PostProcess(self.vec2box, self.validation_cfg.nms)
        self.loss_fn = None
        if hasattr(self.cfg.task, "loss"):
            try:
                self.loss_fn = create_loss_function(self.cfg, self.vec2box)
            except Exception as exc:
                logger.warning(f":warning: Validation loss logging disabled (failed to load loss fn): {exc}")

    def val_dataloader(self):
        return self.val_loader

    def validation_step(self, batch, batch_idx):
        batch_size, images, targets, rev_tensor, img_paths = batch
        H, W = images.shape[2:]
        ema_outputs = self.ema(images)
        calc_val_loss = bool(getattr(self.validation_cfg, "calc_epoch_total_val_loss", False))
        if calc_val_loss and self.loss_fn is not None:
            self.vec2box.update([W, H])
            aux_predicts = self.vec2box(ema_outputs["AUX"])
            main_predicts = self.vec2box(ema_outputs["Main"])
            val_loss, _ = self.loss_fn(aux_predicts, main_predicts, targets)
            self.log(
                "Loss/val_total",
                val_loss.detach(),
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                logger=False,
                batch_size=batch_size,
            )
        elif calc_val_loss and self.loss_fn is None and not self._warn_missing_val_loss:
            self._warn_missing_val_loss = True
            logger.warning(":warning: Validation loss logging requested but loss function is unavailable.")
        predicts = self.post_process(ema_outputs, image_size=[W, H])
        pred_list = [to_metrics_format(predict) for predict in predicts]
        tgt_list = [to_metrics_format(target) for target in targets]

        if getattr(self.validation_cfg, "skip_metric_when_empty", False):
            filtered_pairs = []
            for pred_dict, tgt_dict in zip(pred_list, tgt_list):
                pred_boxes = pred_dict.get("boxes")
                tgt_boxes = tgt_dict.get("boxes")
                if pred_boxes is None or tgt_boxes is None:
                    continue
                if pred_boxes.numel() == 0 or tgt_boxes.numel() == 0:
                    continue
                filtered_pairs.append((pred_dict, tgt_dict))

            if filtered_pairs:
                filtered_preds, filtered_tgts = zip(*filtered_pairs)
                mAP = self.metric(list(filtered_preds), list(filtered_tgts))
                if getattr(self.trainer, "is_global_zero", True):
                    logger.info(
                        ":straight_ruler: Updated detection metric with %s non-empty sample(s)",
                        len(filtered_pairs),
                    )
            else:
                # Keep metric state consistent across DDP ranks by performing a no-op update.
                self.metric([], [])
                mAP = None
                if getattr(self.trainer, "is_global_zero", True):
                    logger.info(
                        ":straight_ruler: Skipped detection metric update for empty predictions/targets"
                    )
        else:
            mAP = self.metric(pred_list, tgt_list)
        return predicts, mAP

    def on_validation_epoch_end(self):
        if getattr(self.trainer, "is_global_zero", True):
            logger.info(":stopwatch: Starting distributed mAP compute for epoch %s", int(self.current_epoch))
        epoch_metrics = self.metric.compute()
        if getattr(self.trainer, "is_global_zero", True):
            logger.info(":checkered_flag: Completed distributed mAP compute for epoch %s", int(self.current_epoch))
        # Pretty summary printing (skip during sanity check)
        if not getattr(self.trainer, "sanity_checking", False):
            try:
                self._print_ap_ar_combined_table(epoch_metrics, int(self.current_epoch))
            except Exception:
                pass
        # Optionally print per-class mAP to console
        try:
            if getattr(self.validation_cfg, "print_map_per_class", False) and not getattr(self.trainer, "sanity_checking", False):
                classes = epoch_metrics.get("classes", None)
                map_per_class = epoch_metrics.get("map_per_class", None)
                if classes is not None and map_per_class is not None:
                    self._print_map_per_class_table(classes, map_per_class, title="Per-class mAP:")
        except Exception:
            pass
        # If per-class printing is disabled, still compute and print at final epoch only
        # Final-epoch one-time per-class output when disabled in config
        if not getattr(self.validation_cfg, "print_map_per_class", False) and not getattr(self.trainer, "sanity_checking", False):
            max_epochs = getattr(self.trainer, "max_epochs", None)
            # Handle None safely; treat current epoch as final if max not set
            is_final = max_epochs is None or (int(self.trainer.current_epoch) + 1) >= int(max_epochs)
            if is_final:
                self._compute_and_print_per_class_once()
        # Remove non-scalar fields before logging (e.g., classes, map_per_class, mar_100_per_class)
        scalar_epoch_metrics = {}
        for k, v in epoch_metrics.items():
            if k == "classes" or "per_class" in k:
                continue
            try:
                # accept python numbers directly
                if isinstance(v, (int, float)):
                    scalar_epoch_metrics[k] = float(v)
                    continue
                # accept 0-dim tensors
                import torch as _torch

                if isinstance(v, _torch.Tensor) and v.ndim == 0:
                    scalar_epoch_metrics[k] = v
            except Exception:
                continue

        # Log scalars only
        if scalar_epoch_metrics:
            self.log_dict(scalar_epoch_metrics, prog_bar=True, sync_dist=True, rank_zero_only=True, logger=False)
        if "map" in scalar_epoch_metrics and "map_50" in scalar_epoch_metrics:
            self.log_dict(
                {"PyCOCO/AP @ .5:.95": scalar_epoch_metrics["map"], "PyCOCO/AP @ .5": scalar_epoch_metrics["map_50"]},
                sync_dist=True,
                rank_zero_only=True,
                logger=False,
            )

        # Additionally, push map*/mar* to TensorBoard with epoch as x-axis
        # Exclude per-class arrays like map_per_class, mar_100_per_class
        tb_metrics = {
            k: v
            for k, v in scalar_epoch_metrics.items()
            if k.startswith(("map", "mar")) and "per_class" not in k
        }
        if tb_metrics:
            for lg in self.trainer.loggers:
                if isinstance(lg, TensorBoardLogger):
                    exp = lg.experiment
                    step = int(self.current_epoch)
                    for k, v in tb_metrics.items():
                        try:
                            scalar = float(v)
                        except Exception:
                            continue
                        exp.add_scalar(k, scalar, global_step=step)
        # Log validation loss to TensorBoard using aggregated epoch metric
        callback_metrics = getattr(self.trainer, "callback_metrics", {}) or {}
        val_total_metric = callback_metrics.get("Loss/val_total")
        if val_total_metric is not None:
            try:
                val_scalar = float(val_total_metric)
            except Exception:
                val_scalar = None
            if val_scalar is not None:
                step = int(self.current_epoch)
                for lg in self.trainer.loggers:
                    if isinstance(lg, TensorBoardLogger):
                        exp = lg.experiment
                        exp.add_scalar("Loss/val_total", val_scalar, global_step=step)
        self.metric.reset()

    @torch.no_grad()
    def _compute_and_print_per_class_once(self):
        metric_pc = MeanAveragePrecision(
            iou_type="bbox", box_format="xyxy", backend="faster_coco_eval", class_metrics=True
        )
        metric_pc.warn_on_many_detections = False
        names = getattr(self.cfg.dataset, "class_list", None)
        model_to_use = getattr(self, "ema", self.model)
        model_to_use.eval()
        device = self.device
        for batch in self.val_loader:
            batch_size, images, targets, rev_tensor, img_paths = batch
            H, W = images.shape[2:]
            # Move to the same device as the model for a valid forward
            images = images.to(device)
            predicts = self.post_process(model_to_use(images), image_size=[W, H])
            # Ensure both predictions and targets are on the same device (CPU) for TorchMetrics COCO backend
            pred_list = [to_metrics_format(p.detach().cpu()) for p in predicts]
            tgt_list = [to_metrics_format(t.detach().cpu()) for t in targets]
            metric_pc(pred_list, tgt_list)
        m = metric_pc.compute()
        classes = m.get("classes", None)
        map_per_class = m.get("map_per_class", None)
        if classes is None or map_per_class is None:
            return
        self._print_map_per_class_table(classes, map_per_class, title="Per-class mAP (final epoch):")

    def _print_map_per_class_table(self, classes, map_per_class, title: str = "Per-class mAP:"):
        """Render per-class AP as a 20x3 grid table with aligned columns using box-drawing chars.

        Columns per cell: ID | Name | AP
        - Sorted by class id ascending
        - Avoids logger prefixes by printing directly to stdout
        """
        try:
            names = getattr(self.cfg.dataset, "class_list", None)
            # Build entries and sort by class id ascending
            # Prefer to list ALL dataset classes; fill missing with 0.0 so every class is shown
            ap_by_id = {}
            for cid, ap in zip(classes, map_per_class):
                try:
                    ap_by_id[int(cid)] = float(ap)
                except Exception:
                    continue
            entries = []
            if names is not None:
                try:
                    total = len(names)
                except Exception:
                    total = None
                if isinstance(total, int) and total > 0:
                    for idx in range(total):
                        try:
                            name = names[idx]
                        except Exception:
                            name = ""
                        val = ap_by_id.get(idx, 0.0)
                        entries.append((idx, str(name), val))
                else:
                    for idx in sorted(ap_by_id.keys()):
                        entries.append((idx, "", ap_by_id[idx]))
            else:
                for idx in sorted(ap_by_id.keys()):
                    entries.append((idx, "", ap_by_id[idx]))
            if not entries:
                return

            # Fixed per-field widths as requested: ID=3, Name=25, AP=7
            max_id_digits = 3
            name_w = 25
            ap_w = 7

            # Fixed grid: 20 rows x 3 columns
            col_height = 20
            num_cols = 3
            # Column width including two inner separators between fields
            col_width = max_id_digits + 1 + name_w + 1 + ap_w

            # Build per-cell texts aligned as: ID│Name│AP (with inner separators)
            def cell_text(idx: int, name: str, val: float) -> str:
                id_str = str(idx).rjust(max_id_digits)
                name_str = name[:name_w].ljust(name_w)
                ap_str = f"{val:.4f}".rjust(ap_w)
                return f"{id_str}│{name_str}│{ap_str}"

            cell_texts = [cell_text(idx, name, val) for idx, name, val in entries]

            # Build box-drawing borders with inner joints aligned to ID/Name/AP splits
            def build_border(left: str, inner: str, between: str, right: str, fill: str) -> str:
                # one column segment: ID fill + inner + Name fill + inner + AP fill
                seg = (fill * max_id_digits) + inner + (fill * name_w) + inner + (fill * ap_w)
                return left + (seg + between) * (num_cols - 1) + seg + right

            top = build_border("┏", "┳", "┳", "┓", "━")
            header_sep = build_border("┡", "╇", "╇", "┩", "━")
            bottom = build_border("└", "┴", "┴", "┘", "─")

            # Compose header row (repeat per-column header) using heavy inner separators ┃ with bold labels
            def _bold(text: str) -> str:
                return f"\033[1m{text}\033[0m"
            def ljust_ansi(text: str, width: int) -> str:
                # Pad based on printable length (excluding ANSI escape sequences)
                import re
                stripped = re.sub(r"\x1b\[[0-9;]*m", "", text)
                pad = max(0, width - len(stripped))
                return text + (" " * pad)

            hdr = f"{_bold('ID'.rjust(max_id_digits))}┃{_bold('Name'.ljust(name_w))}┃{_bold('AP'.rjust(ap_w))}"
            hdr_cells = [ljust_ansi(hdr, col_width) for _ in range(num_cols)]
            header_row = "┃" + "┃".join(hdr_cells) + "┃"

            # Compose body rows (always 20 rows)
            body_lines = []
            # Precompute an empty cell that still shows inner separators
            empty_cell = (" " * max_id_digits) + "│" + (" " * name_w) + "│" + (" " * ap_w)

            for r in range(col_height):
                row_cells = []
                for c in range(num_cols):
                    i = c * col_height + r
                    if i < len(cell_texts):
                        txt = cell_texts[i].ljust(col_width)
                    else:
                        txt = empty_cell.ljust(col_width)
                    row_cells.append(txt)
                body_lines.append("│" + "│".join(row_cells) + "│")

            # Assemble table string and print without logger prefixes
            table_lines = [top, header_row, header_sep] + body_lines + [bottom]
            print("\n".join(table_lines))
        except Exception:
            # Fail-safe: skip printing if any formatting error occurs
            pass

    # _print_avg_ap_ar_table removed

    def _print_ap_ar_combined_table(self, metrics: dict, epoch: int):
        # Build a combined AP/AR table similar to the sample, printed via stdout
        def to_f(x):
            try:
                return float(x)
            except Exception:
                return None

        ap = [
            ("AP @ .5:.95", to_f(metrics.get("map"))),
            ("AP @     .5", to_f(metrics.get("map_50"))),
            ("AP @    .75", to_f(metrics.get("map_75"))),
            ("AP  (small)", to_f(metrics.get("map_small"))),
            ("AP (medium)", to_f(metrics.get("map_medium"))),
            ("AP  (large)", to_f(metrics.get("map_large"))),
        ]
        ar = [
            ("AR maxDets   1", to_f(metrics.get("mar_1"))),
            ("AR maxDets  10", to_f(metrics.get("mar_10"))),
            ("AR maxDets 100", to_f(metrics.get("mar_100"))),
            ("AR     (small)", to_f(metrics.get("mar_small"))),
            ("AR    (medium)", to_f(metrics.get("mar_medium"))),
            ("AR     (large)", to_f(metrics.get("mar_large"))),
        ]

        # widths
        epoch_w = 5
        lab_w = 16
        pct_w = 6

        # helpers
        def border(l, j1, j2, r, fill):
            seg = (fill * epoch_w) + j1 + (fill * lab_w) + j1 + (fill * pct_w) + j2 + (fill * lab_w) + j1 + (fill * pct_w)
            return l + seg + r

        def fmt_pct(v):
            return ("-" if v is None else f"{v*100:0{pct_w}.2f}")

        top = border("┏", "┳", "┳", "┓", "━")
        mid = border("┡", "╇", "╇", "┩", "━")
        bot = border("└", "┴", "┴", "┘", "─")

        # header
        h_epoch = "Epoch".rjust(epoch_w)
        h_ap = "Avg. Precision".ljust(lab_w)
        h_ap_pct = "%".rjust(pct_w)
        h_ar = "Avg. Recall".ljust(lab_w)
        h_ar_pct = "%".rjust(pct_w)
        header = f"┃{h_epoch}┃{h_ap}┃{h_ap_pct}╇{h_ar}┃{h_ar_pct}┃"

        lines = [top, header, mid]
        for i in range(len(ap)):
            ep = str(epoch).rjust(epoch_w)
            ap_lab, ap_val = ap[i]
            ar_lab, ar_val = ar[i]
            ap_lab = ap_lab.ljust(lab_w)[:lab_w]
            ar_lab = ar_lab.ljust(lab_w)[:lab_w]
            ap_p = fmt_pct(ap_val).rjust(pct_w)
            ar_p = fmt_pct(ar_val).rjust(pct_w)
            row = f"│{ep}│{ap_lab}│{ap_p}╎{ar_lab}│{ar_p}│"
            lines.append(row)
        lines.append(bot)
        print("\n".join(lines))


class TrainModel(ValidateModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.train_loader = create_dataloader(self.cfg.task.data, self.cfg.dataset, self.cfg.task.task)
        self._nms_logged = False
        self._highest_eval_enabled = bool(
            getattr(getattr(self.validation_cfg, "nms", None), "highest_eval_in_the_final_epoch", False)
        )
        self._final_eval_triggered = False

    def setup(self, stage):
        super().setup(stage)
        if self.loss_fn is None:
            self.loss_fn = create_loss_function(self.cfg, self.vec2box)
        if not self._nms_logged and (stage is None or stage == "fit"):
            nms_cfg = getattr(self.validation_cfg, "nms", None)
            if nms_cfg is not None:
                nms_dict = None
                if is_dataclass(nms_cfg):
                    nms_dict = asdict(nms_cfg)
                elif isinstance(nms_cfg, Mapping):
                    nms_dict = dict(nms_cfg)
                else:
                    known_fields = (
                        "min_confidence",
                        "min_iou",
                        "pre_topk",
                        "max_bbox",
                        "multi_label",
                        "class_agnostic",
                        "size_bias_alpha",
                    )
                    extracted = {key: getattr(nms_cfg, key) for key in known_fields if hasattr(nms_cfg, key)}
                    if extracted:
                        nms_dict = extracted

                if nms_dict is not None:
                    ordered_keys = [
                        "min_confidence",
                        "min_iou",
                        "pre_topk",
                        "max_bbox",
                        "multi_label",
                        "class_agnostic",
                        "size_bias_alpha",
                    ]
                    summary = []
                    for key in ordered_keys:
                        if key in nms_dict:
                            summary.append(f"{key}={nms_dict[key]}")
                    for key, value in nms_dict.items():
                        if key not in ordered_keys:
                            summary.append(f"{key}={value}")
                    nms_repr = ", ".join(summary) if summary else str(nms_dict)
                else:
                    nms_repr = str(nms_cfg)

                logger.info(f":information_source: Validation NMS config: {nms_repr}")
                self._nms_logged = True
        # Optional: load teacher for online KD
        self.kd_cfg = getattr(self.cfg.task, "kd", None)
        self.teacher = None
        if self.kd_cfg and getattr(self.kd_cfg, "enable", False):
            try:
                from omegaconf import OmegaConf
                from pathlib import Path as _P

                teacher_name = getattr(self.kd_cfg, "teacher_model", "v9-e")
                model_path = _P("yolo/config/model") / f"{teacher_name}.yaml"
                if not model_path.exists():
                    raise FileNotFoundError(f"Teacher model config not found: {model_path}")
                teacher_cfg = OmegaConf.load(str(model_path))
                weight_path = getattr(self.kd_cfg, "teacher_weight", None)
                self.teacher = create_model(teacher_cfg, weight_path=weight_path, class_num=self.cfg.dataset.class_num)
                self.teacher.eval()
                if getattr(self.kd_cfg, "freeze_teacher", True):
                    for p in self.teacher.parameters():
                        p.requires_grad = False
                self.teacher = self.teacher.to(self.device)
                if self.device.type == "cuda" and getattr(self.kd_cfg, "teacher_fp16", True):
                    self.teacher = self.teacher.half()
                logger.info(":school: Online KD enabled with teacher model loaded")
            except Exception as e:
                logger.warning(f":warning: Failed to load teacher for KD: {e}")
                self.teacher = None

    def train_dataloader(self):
        return self.train_loader

    def on_fit_end(self):
        super().on_fit_end()
        trainer = getattr(self, "trainer", None)
        if not trainer or getattr(trainer, "sanity_checking", False):
            return
        if not self._highest_eval_enabled or self._final_eval_triggered:
            return

        # Guard against scenarios where training terminated before reaching the nominal final epoch.
        # Still honour the request by running the highest-precision evaluation once at shutdown.
        self._final_eval_triggered = True
        strategy = getattr(trainer, "strategy", None)
        if strategy and hasattr(strategy, "barrier"):
            with suppress(Exception):
                strategy.barrier("highest_eval_sync_start")

        progress_bar = getattr(trainer, "progress_bar_callback", None)
        pb_context = nullcontext()
        pb_restore = None
        if progress_bar and hasattr(progress_bar, "disable"):
            try:
                maybe_ctx = progress_bar.disable()
            except Exception:
                maybe_ctx = None
            if hasattr(maybe_ctx, "__enter__") and hasattr(maybe_ctx, "__exit__"):
                pb_context = maybe_ctx
            else:
                pb_restore = getattr(progress_bar, "enable", None)

        overrides_backup = None
        success = False
        eval_metrics: dict[str, object] = {}
        with pb_context:
            if getattr(trainer, "is_global_zero", True):
                # Ensure progress updates are paused so the status message is not overwritten.
                logger.info("⏳ The most accurate validation is underway and the evaluation takes just a few minutes.")
            try:
                val_loaders = self._prepare_validation_dataloaders()
                overrides_backup = self._apply_highest_eval_overrides()
                results = trainer.validate(self, dataloaders=val_loaders, ckpt_path=None, verbose=False)
                if isinstance(results, Sequence):
                    for item in results:
                        if isinstance(item, Mapping):
                            eval_metrics.update(item)
                success = True
            except Exception as exc:
                logger.warning(f":warning: Final high-precision validation failed: {exc}")
                logger.debug(traceback.format_exc())
            finally:
                self._restore_nms(overrides_backup)
                self._flush_tensorboard_loggers()
                if strategy and hasattr(strategy, "barrier"):
                    with suppress(Exception):
                        strategy.barrier("highest_eval_sync_flush")
                self._flush_tensorboard_loggers()
                if callable(pb_restore):
                    with suppress(Exception):
                        pb_restore()
                if getattr(trainer, "is_global_zero", True):
                    if success:
                        logger.info("✅ The most accurate validation has been completed.")
                    else:
                        logger.info("⚠ Most accurate validation was not completed. Please check the logs.")
        if success and getattr(trainer, "is_global_zero", True):
            if not eval_metrics:
                callback_metrics = getattr(trainer, "callback_metrics", {}) or {}
                if isinstance(callback_metrics, Mapping):
                    eval_metrics.update(callback_metrics)
                else:
                    eval_metrics.update(dict(callback_metrics))
            if eval_metrics:
                self._update_best_weights_after_highest_eval(trainer, eval_metrics)
        if getattr(trainer, "is_global_zero", True):
            if progress_bar:
                printer = getattr(progress_bar, "print", None)
                if callable(printer):
                    with suppress(Exception):
                        printer("")
                else:
                    print()
            else:
                print()

    def _flush_tensorboard_loggers(self):
        trainer = getattr(self, "trainer", None)
        if not trainer:
            return
        for lg in getattr(trainer, "loggers", []) or []:
            experiment = getattr(lg, "experiment", None)
            flush_fn = getattr(experiment, "flush", None)
            if callable(flush_fn):
                with suppress(Exception):
                    flush_fn()

    def _prepare_validation_dataloaders(self):
        loaders = self.val_dataloader()
        if loaders is None:
            return None
        if isinstance(loaders, Sequence):
            return list(loaders)
        return [loaders]

    def _get_highest_eval_overrides(self):
        nms_cfg = getattr(self.validation_cfg, "nms", None)
        if nms_cfg is None:
            return None

        overrides = getattr(nms_cfg, "highest_eval_overrides", None)
        if overrides:
            return dict(overrides)

        # Fallback to the documented "highest precision" defaults
        return {
            "min_confidence": 1e-4,
            "min_iou": 0.7,
            "pre_topk": 20000,
            "max_bbox": 20000,
            "multi_label": True,
            "class_agnostic": False,
        }

    def _apply_highest_eval_overrides(self):
        nms_cfg = getattr(self.validation_cfg, "nms", None)
        overrides = self._get_highest_eval_overrides()
        if nms_cfg is None or not overrides:
            return None

        state = vars(nms_cfg)
        backup = {key: state.get(key) for key in overrides}
        # Update in place so PostProcess sees the new thresholds
        for key, value in overrides.items():
            state[key] = value
        return backup

    def _restore_nms(self, backup):
        if not backup:
            return
        nms_cfg = getattr(self.validation_cfg, "nms", None)
        if nms_cfg is None:
            return
        state = vars(nms_cfg)
        for key, value in backup.items():
            state[key] = value

    def _update_best_weights_after_highest_eval(self, trainer, metrics: Mapping[str, object]) -> None:
        if not getattr(trainer, "is_global_zero", True):
            return
        callbacks = getattr(trainer, "callbacks", None)
        if not callbacks:
            return
        for cb in callbacks:
            if isinstance(cb, SaveBestWeights):
                cb.update_from_metrics(trainer, self, metrics)
                break

    def on_train_epoch_start(self):
        self.trainer.optimizers[0].next_epoch(
            ceil(len(self.train_loader) / self.trainer.world_size), self.current_epoch
        )
        self.vec2box.update(self.cfg.image_size)

    def training_step(self, batch, batch_idx):
        lr_dict = self.trainer.optimizers[0].next_batch()
        batch_size, images, targets, *_ = batch
        predicts = self(images)
        aux_predicts = self.vec2box(predicts["AUX"])
        main_predicts = self.vec2box(predicts["Main"])
        loss, loss_item = self.loss_fn(aux_predicts, main_predicts, targets)

        # Online KD loss (added to GT loss)
        if self.teacher is not None:
            kd_loss, kd_items = self._compute_kd_loss(images, predicts)
            loss = loss + kd_loss
            # Merge KD logs into loss_item for reporting
            loss_item.update(kd_items)
        # Log losses with stable TensorBoard ordering using numeric prefixes
        # Desired visual order:
        #   step:  BCELoss -> BoxLoss -> DFLLoss
        #   epoch: BCELoss -> BoxLoss -> DFLLoss
        # Implemented tags:
        #   Loss/01_BCELoss_step, Loss/02_BoxLoss_step, Loss/03_DFLLoss_step,
        #   Loss/11_BCELoss_epoch, Loss/12_BoxLoss_epoch, Loss/13_DFLLoss_epoch
        # Source keys in loss_item: 'Loss/BCELoss', 'Loss/BoxLoss', 'Loss/DFLLoss'
        order_step = ["BCELoss", "BoxLoss", "DFLLoss"]
        order_epoch = ["BCELoss", "BoxLoss", "DFLLoss"]
        step_prefixes = ["01", "02", "03"]
        epoch_prefixes = ["11", "12", "13"]

        # Step logs (throttled globally by Trainer.log_every_n_steps)
        for idx, name in enumerate(order_step):
            src_key = f"Loss/{name}"
            if src_key in loss_item:
                tag = f"Loss/{step_prefixes[idx]}_{name}_step"
                self.log(tag, loss_item[src_key], prog_bar=True, on_step=True, on_epoch=False, rank_zero_only=True)

        # Epoch logs (aggregated per epoch). Do not send to loggers here; we'll push with epoch x-axis manually.
        for idx, name in enumerate(order_epoch):
            src_key = f"Loss/{name}"
            if src_key in loss_item:
                tag = f"Loss/{epoch_prefixes[idx]}_{name}_epoch"
                self.log(
                    tag,
                    loss_item[src_key],
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    rank_zero_only=True,
                    logger=False,
                )
        # Ensure LR logs participate in global step throttling too
        self.log_dict(lr_dict, prog_bar=False, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)
        return loss * batch_size

    def _cosine_temperature(self, epoch: int, max_epochs: int, t0: float, t1: float) -> float:
        if max_epochs <= 1:
            return float(t1)
        import math

        cos = 0.5 * (1 + math.cos(math.pi * min(epoch, max_epochs - 1) / (max_epochs - 1)))
        return float(t1 + (t0 - t1) * cos)

    def _compute_kd_loss(self, images, student_out):
        import torch.nn.functional as F
        kd = self.kd_cfg
        # Temperature schedule
        T = self._cosine_temperature(int(self.current_epoch), int(self.trainer.max_epochs or self.cfg.task.epoch), kd.temperature.init, kd.temperature.final)

        # Forward teacher with no grad
        with torch.no_grad():
            t_images = images.half() if (self.device.type == "cuda" and getattr(kd, "teacher_fp16", True)) else images
            teacher_out = self.teacher(t_images)

        # Select branches
        branches = []
        apply_to = getattr(kd, "apply_to", "main").lower()
        if apply_to in ("main", "both"):
            branches.append((student_out["Main"], teacher_out["Main"]))
        if apply_to in ("aux", "both") and "AUX" in student_out and "AUX" in teacher_out:
            branches.append((student_out["AUX"], teacher_out["AUX"]))

        kd_cls = kd_dfl = kd_box = images.new_tensor(0.0)
        for s_list, t_list in branches:
            for (s_cls, s_anc, s_vec), (t_cls, t_anc, t_vec) in zip(s_list, t_list):
                # Shapes:
                # s_cls/t_cls: [B, C, H, W]
                # s_anc/t_anc: [B, R, 4, H, W]  (logits over reg_max along dim=1)
                # s_vec/t_vec: [B, 4, H, W]     (expected distances)

                # Classification KD: soft-BCE with temperature, scaled by T^2
                t_prob = (t_cls.float() / T).sigmoid()
                s_logit = s_cls.float() / T
                cls_loss = F.binary_cross_entropy_with_logits(s_logit, t_prob, reduction="mean") * (T * T)
                kd_cls = kd_cls + cls_loss

                # DFL KD: KL divergence between teacher and student distributions along reg axis
                # reshape to [B, 4, H, W, R] for stable softmax over R
                s_reg = (s_anc.float() / T).permute(0, 2, 3, 4, 1)  # B,4,H,W,R
                t_reg = (t_anc.float() / T).permute(0, 2, 3, 4, 1)
                s_logp = F.log_softmax(s_reg, dim=-1)
                t_prob_reg = F.softmax(t_reg, dim=-1)
                # KL(t||s) averaged
                dfl_loss = F.kl_div(s_logp, t_prob_reg, reduction="batchmean") * (T * T)
                kd_dfl = kd_dfl + dfl_loss

                # Box KD: L1 on vector expectation
                box_loss = F.l1_loss(s_vec.float(), t_vec.float())
                kd_box = kd_box + box_loss

        # Weighting
        w = kd.weights
        kd_total = w.cls * kd_cls + w.dfl * kd_dfl + w.box * kd_box
        return kd_total, {"KD/cls": kd_cls.detach().item(), "KD/dfl": kd_dfl.detach().item(), "KD/box": kd_box.detach().item()}

    def on_train_epoch_end(self):
        # Push epoch-aggregated loss metrics to TensorBoard with epoch as x-axis
        if not hasattr(self.trainer, "loggers"):
            return
        if not self.trainer.is_global_zero:
            return
        tags = [
            "Loss/11_BCELoss_epoch",
            "Loss/12_BoxLoss_epoch",
            "Loss/13_DFLLoss_epoch",
        ]
        metrics = getattr(self.trainer, "callback_metrics", {}) or {}
        step = int(self.current_epoch)
        for lg in self.trainer.loggers:
            if isinstance(lg, TensorBoardLogger):
                exp = lg.experiment
                for tag in tags:
                    if tag in metrics:
                        try:
                            scalar = float(metrics[tag])
                        except Exception:
                            continue
                        exp.add_scalar(tag, scalar, global_step=step)

    def configure_optimizers(self):
        optimizer = create_optimizer(self.model, self.cfg.task.optimizer)
        scheduler = create_scheduler(optimizer, self.cfg.task.scheduler)
        return [optimizer], [scheduler]


class InferenceModel(BaseModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        # Swap to fast inference model if requested (ONNX/TRT/deploy)
        compiler = getattr(cfg.task, "fast_inference", None)
        if compiler:
            try:
                device = str(cfg.device)
                loader = FastModelLoader(cfg)
                self.model = loader.load_model(device)
            except Exception as e:
                logger.warning(f":warning: Fast inference load failed ({compiler}), fallback to PyTorch. Error: {e}")
        self.predict_loader = create_dataloader(cfg.task.data, cfg.dataset, cfg.task.task)

    def setup(self, stage):
        self.vec2box = create_converter(
            self.cfg.model.name,
            model=self.model,
            anchor_cfg=self.cfg.model.anchor,
            image_size=self.cfg.image_size,
            device=self.device,
            class_num=self.cfg.dataset.class_num,
        )
        self.post_process = PostProcess(self.vec2box, self.cfg.task.nms)

    def predict_dataloader(self):
        return self.predict_loader

    def predict_step(self, batch, batch_idx):
        if len(batch) == 4:
            images, rev_tensor, origin_frame, meta = batch
        else:
            images, rev_tensor, origin_frame = batch
            meta = {}
        predicts = self.post_process(self(images), rev_tensor=rev_tensor)
        # Draw only box outlines during inference (no fill)
        render_labels = getattr(self.cfg.task, "render_labels", True)
        img = draw_bboxes(
            origin_frame,
            predicts,
            idx2label=self.cfg.dataset.class_list,
            fill=False,
            draw_labels=render_labels,
        )
        if getattr(self.predict_loader, "is_stream", None):
            fps = self._display_stream(img)
        else:
            fps = None
        if getattr(self.cfg.task, "save_predict", None):
            self._save_image(img, batch_idx, meta)
        return img, fps

    def _save_image(self, img, batch_idx, meta=None):
        save_dir = Path(self.trainer.default_root_dir)
        filename = f"frame{batch_idx:03d}.png"
        source_path = None
        is_single_image = False
        if isinstance(meta, dict):
            source_path = meta.get("source_path")
            is_single_image = bool(meta.get("is_single_image"))
        if source_path and is_single_image:
            stem = Path(source_path).stem
            filename = f"{stem}.png"
            save_image_path = save_dir / filename
            counter = 1
            while save_image_path.exists():
                save_image_path = save_dir / f"{stem}_{counter}.png"
                counter += 1
        else:
            save_image_path = save_dir / filename
        img.save(save_image_path)
        print(f"💾 Saved visualize image at {save_image_path}")
