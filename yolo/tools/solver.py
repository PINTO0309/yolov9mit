from math import ceil
from pathlib import Path

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
from yolo.utils.model_utils import PostProcess, create_optimizer, create_scheduler


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

    def setup(self, stage):
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device
        )
        self.post_process = PostProcess(self.vec2box, self.validation_cfg.nms)

    def val_dataloader(self):
        return self.val_loader

    def validation_step(self, batch, batch_idx):
        batch_size, images, targets, rev_tensor, img_paths = batch
        H, W = images.shape[2:]
        predicts = self.post_process(self.ema(images), image_size=[W, H])
        mAP = self.metric(
            [to_metrics_format(predict) for predict in predicts], [to_metrics_format(target) for target in targets]
        )
        return predicts, mAP

    def on_validation_epoch_end(self):
        epoch_metrics = self.metric.compute()
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
            rev_tensor = rev_tensor.to(device)
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

    def setup(self, stage):
        super().setup(stage)
        self.loss_fn = create_loss_function(self.cfg, self.vec2box)

    def train_dataloader(self):
        return self.train_loader

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
        # TODO: Add FastModel
        self.predict_loader = create_dataloader(cfg.task.data, cfg.dataset, cfg.task.task)

    def setup(self, stage):
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device
        )
        self.post_process = PostProcess(self.vec2box, self.cfg.task.nms)

    def predict_dataloader(self):
        return self.predict_loader

    def predict_step(self, batch, batch_idx):
        images, rev_tensor, origin_frame = batch
        predicts = self.post_process(self(images), rev_tensor=rev_tensor)
        img = draw_bboxes(origin_frame, predicts, idx2label=self.cfg.dataset.class_list)
        if getattr(self.predict_loader, "is_stream", None):
            fps = self._display_stream(img)
        else:
            fps = None
        if getattr(self.cfg.task, "save_predict", None):
            self._save_image(img, batch_idx)
        return img, fps

    def _save_image(self, img, batch_idx):
        save_image_path = Path(self.trainer.default_root_dir) / f"frame{batch_idx:03d}.png"
        img.save(save_image_path)
        print(f"💾 Saved visualize image at {save_image_path}")
