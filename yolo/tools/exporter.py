from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Tuple

import torch
from omegaconf import DictConfig, OmegaConf

from yolo.config.config import Config, ExportConfig
from yolo.model.yolo import create_model
from yolo.utils.bounding_box_utils import generate_anchors
from yolo.utils.logger import logger


class EfficientONNXModule(torch.nn.Module):
    """Wraps the YOLO model so the ONNX graph emits a single [B, 4+C, N] tensor."""

    def __init__(
        self,
        model: torch.nn.Module,
        anchor_cfg,
        image_size: Sequence[int],
        *,
        apply_sigmoid: bool,
    ) -> None:
        super().__init__()
        self.model = model
        self.apply_sigmoid = apply_sigmoid

        width, height = int(image_size[0]), int(image_size[1])
        strides = self._resolve_strides(model, anchor_cfg, width, height)
        anchor_grid, scaler = generate_anchors([width, height], strides)

        self.register_buffer("anchor_grid", anchor_grid.to(dtype=torch.float32), persistent=False)
        self.register_buffer("scaler", scaler.to(dtype=torch.float32).unsqueeze(-1), persistent=False)
        self._slices = self._build_slices(width, height, strides)

    @staticmethod
    def _resolve_strides(model: torch.nn.Module, anchor_cfg, width: int, height: int) -> List[int]:
        strides = getattr(anchor_cfg, "strides", None)
        if strides:
            return list(strides)
        with torch.inference_mode():
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype
            dummy = torch.zeros(1, 3, height, width, device=device, dtype=dtype)
            outputs = model(dummy)["Main"]
        resolved: List[int] = []
        for _, _, vec_map in outputs:
            stride = width // int(vec_map.shape[-1])
            resolved.append(int(stride))
        return resolved

    @staticmethod
    def _build_slices(width: int, height: int, strides: Iterable[int]) -> List[Tuple[int, int]]:
        slices: List[Tuple[int, int]] = []
        offset = 0
        for stride in strides:
            anchors = (width // stride) * (height // stride)
            slices.append((offset, offset + anchors))
            offset += anchors
        return slices

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        predictions = self.model(x)["Main"]
        cls_parts: List[torch.Tensor] = []
        box_parts: List[torch.Tensor] = []

        for (start, end), head in zip(self._slices, predictions):
            cls_map, _, vec_map = head
            batch, _, h, w = cls_map.shape
            cls_flat = cls_map.reshape(batch, cls_map.size(1), -1).permute(0, 2, 1)
            if self.apply_sigmoid:
                cls_flat = cls_flat.sigmoid()

            vec_flat = vec_map.reshape(batch, 4, -1).permute(0, 2, 1)
            scale = self.scaler[start:end].unsqueeze(0)
            grid = self.anchor_grid[start:end].unsqueeze(0)
            dist = vec_flat * scale
            lt = dist[..., :2]
            rb = dist[..., 2:]
            boxes = torch.cat([grid - lt, grid + rb], dim=-1)

            cls_parts.append(cls_flat)
            box_parts.append(boxes)

        cls_tensor = torch.cat(cls_parts, dim=1)
        box_tensor = torch.cat(box_parts, dim=1)
        fused = torch.cat([box_tensor, cls_tensor], dim=-1)
        return fused.permute(0, 2, 1)


class ONNXExporter:
    def __init__(self, cfg: Config, save_dir: Path) -> None:
        task_cfg = cfg.task
        if isinstance(task_cfg, DictConfig):
            task_cfg = ExportConfig(**OmegaConf.to_container(task_cfg, resolve=True))
        if not isinstance(task_cfg, ExportConfig):
            raise TypeError("cfg.task must be ExportConfig to run the ONNX exporter")
        self.cfg = cfg
        self.task_cfg = task_cfg
        self.save_dir = Path(save_dir)

    def run(self) -> Path:
        export_path = self._resolve_output_path()
        logger.info(f"📦 Exporting ONNX model to {export_path}")

        model = create_model(
            self.cfg.model,
            class_num=self.cfg.dataset.class_num,
            weight_path=self.cfg.weight,
        ).eval().to("cpu")

        width, height = self._resolve_image_size()

        wrapper = EfficientONNXModule(
            model,
            self.cfg.model.anchor,
            [width, height],
            apply_sigmoid=self.task_cfg.apply_sigmoid,
        ).eval()

        dtype = torch.float16 if self.task_cfg.half else torch.float32
        wrapper = wrapper.to(dtype=dtype)

        dummy = torch.zeros(
            (self.task_cfg.batch_size, 3, height, width),
            dtype=dtype,
        )

        dynamic_axes = None
        if self.task_cfg.dynamic_batch:
            dynamic_axes = {"images": {0: "batch"}, "output": {0: "batch"}}

        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                dummy,
                str(export_path),
                export_params=True,
                opset_version=self.task_cfg.opset,
                do_constant_folding=True,
                input_names=["images"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
            )

        self._post_process(export_path)

        anchors = wrapper.anchor_grid.shape[0]
        features = self.cfg.dataset.class_num + 4
        logger.info(f"✅ ONNX export complete (output dims: batch x {features} x {anchors})")
        return export_path

    def _resolve_image_size(self) -> Tuple[int, int]:
        raw_size = getattr(self.task_cfg, "image_size", None)
        if raw_size is None:
            raw_size = getattr(self.cfg, "image_size", None)
        if raw_size is None:
            raise ValueError("image_size must be specified in the export task or root config")
        if isinstance(raw_size, DictConfig):
            raw_size = OmegaConf.to_container(raw_size, resolve=True)
        if isinstance(raw_size, Mapping):
            height = raw_size.get("height")
            width = raw_size.get("width")
            if width is None or height is None:
                raise ValueError("image_size mapping must contain 'width' and 'height'")
        else:
            if isinstance(raw_size, str):
                raw_clean = raw_size.strip()
                lower = raw_clean.lower()
                has_x = 'x' in lower
                normalized = lower.replace('x', ',')
                parts = [p for p in normalized.replace(' ', '').split(',') if p]
                if len(parts) != 2:
                    raise ValueError("image_size string must look like 'width,height' or 'heightxwidth'")
                if has_x:
                    height, width = parts
                else:
                    width, height = parts
            else:
                if not isinstance(raw_size, (list, tuple)):
                    try:
                        raw_size = list(raw_size)
                    except TypeError as exc:
                        raise ValueError("image_size must be length 2 (width, height)") from exc
                if len(raw_size) != 2:
                    raise ValueError("image_size must be length 2 (width, height)")
                width, height = raw_size
        width, height = int(width), int(height)
        if width <= 0 or height <= 0:
            raise ValueError("image_size entries must be positive")
        return width, height

    def _resolve_output_path(self) -> Path:
        alias = self.cfg.model.name or "model"
        weight_hint = None
        if isinstance(self.cfg.weight, str):
            weight_hint = Path(self.cfg.weight)
        elif self.cfg.weight is True:
            weight_hint = Path("weights") / f"{alias}.pt"

        if self.task_cfg.output_path:
            path = Path(self.task_cfg.output_path)
            if not path.is_absolute():
                path = self.save_dir / path
        else:
            width, height = self._resolve_image_size()
            batch_size = "N" if self.task_cfg.dynamic_batch else str(self.task_cfg.batch_size)
            channel = 3
            suffix = f"_{batch_size}x{channel}x{height}x{width}.onnx"
            if weight_hint:
                base = weight_hint.with_suffix(".onnx")
                path = base.with_name(base.stem + suffix)
            else:
                path = self.save_dir / f"{alias}{suffix}"

        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _post_process(self, onnx_path: Path) -> None:
        raw_names = getattr(self.cfg.dataset, "class_list", None)
        need_meta = bool(raw_names) and self.task_cfg.include_metadata
        need_simplify = self.task_cfg.simplify
        if not need_meta and not need_simplify:
            return

        try:
            import onnx
        except Exception as exc:
            logger.warning(f"⚠️ Unable to load ONNX utilities: {exc}")
            return

        try:
            model_onnx = onnx.load(str(onnx_path))
        except Exception as exc:
            logger.warning(f"⚠️ Failed to read exported ONNX file: {exc}")
            return

        updated = False

        if need_simplify:
            try:
                from onnxsim import simplify

                model_onnx, ok = simplify(model_onnx)
                if not ok:
                    logger.warning("⚠️ onnxsim reported check=False; keeping unsimplified graph")
                else:
                    updated = True
                    logger.info("🔁 Simplified ONNX graph using onnxsim")
            except Exception as exc:
                logger.warning(f"⚠️ Skipping onnxsim simplification: {exc}")

        if need_meta:
            try:
                from onnx.helper import set_model_props

                class_list = [str(name) for name in list(raw_names)]
                meta = {
                    "class_count": str(self.cfg.dataset.class_num),
                    "names": json.dumps(class_list),
                }
                set_model_props(model_onnx, meta)
                updated = True
            except Exception as exc:
                logger.warning(f"⚠️ Could not embed metadata: {exc}")

        if updated:
            try:
                onnx.save(model_onnx, str(onnx_path))
            except Exception as exc:
                logger.warning(f"⚠️ Failed to save post-processed ONNX: {exc}")

