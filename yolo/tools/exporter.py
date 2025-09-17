from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
from omegaconf import DictConfig, OmegaConf

from yolo.config.config import Config, ExportConfig
from yolo.model.yolo import create_model
from yolo.utils.bounding_box_utils import generate_anchors
from yolo.utils.logger import logger
from yolo.model.module import Anchor2Vec


EXPORT_SIGNATURE_KEY = "yolov9mit_export_version"
EXPORT_SIGNATURE_VALUE = "2"
EXPORT_ANCHOR_LAYOUT_KEY = "yolov9mit_anchor_layout"
EXPORT_ANCHOR_LAYOUT_VALUE = "bcn"


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

        anchor_grid = anchor_grid.to(dtype=torch.float32).transpose(0, 1).unsqueeze(0).contiguous()
        scaler = scaler.to(dtype=torch.float32).view(1, 1, -1)

        self.register_buffer("anchor_grid", anchor_grid, persistent=False)
        self.register_buffer("scaler", scaler, persistent=False)
        self.reg_max = getattr(anchor_cfg, "reg_max", None)

        coeff = torch.tensor(
            [
                [-0.5, 0.0, 0.5, 0.0],
                [0.0, -0.5, 0.0, 0.5],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("dist_to_box", coeff, persistent=False)
        box_bias = torch.cat([anchor_grid, torch.zeros_like(anchor_grid)], dim=1)
        self.register_buffer("box_bias", box_bias, persistent=False)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        predictions = self.model(x)["Main"]

        cls_chunks: List[torch.Tensor] = []
        dist_chunks: List[torch.Tensor] = []
        logit_chunks: List[torch.Tensor] = []

        reg_max = self.reg_max or 16

        for head in predictions:
            cls_map, _, vec_map = head
            batch, channels, h, w = cls_map.shape
            hw = h * w
            cls_chunks.append(cls_map.reshape(batch, channels, hw))

            if vec_map.ndim == 5:
                reg_bins = vec_map.shape[2]
                logit_chunks.append(vec_map.reshape(batch, 4, reg_bins, hw))
            elif vec_map.shape[1] == 4 * reg_max:
                logit_chunks.append(vec_map.reshape(batch, 4, reg_max, hw))
            else:
                dist_map = self._vector_to_distance(vec_map)
                dist_chunks.append(dist_map.reshape(batch, 4, hw))

        cls_tensor = torch.cat(cls_chunks, dim=2)

        if logit_chunks:
            logits = torch.cat(logit_chunks, dim=-1)
            dist_tensor = self._logits_to_distance(logits)
        else:
            dist_tensor = torch.cat(dist_chunks, dim=2)

        if self.apply_sigmoid:
            cls_tensor = cls_tensor.sigmoid()

        dist = dist_tensor * self.scaler.to(dist_tensor.dtype)
        coeff = self.dist_to_box.to(dist.dtype).unsqueeze(0)
        combo = torch.matmul(coeff, dist)
        box_tensor = combo + self.box_bias.to(dist.dtype)

        fused = torch.cat([box_tensor, cls_tensor], dim=1)
        return fused

    def _vector_to_distance(self, vec_map: torch.Tensor) -> torch.Tensor:
        """Convert either raw logits or projected distances to LTRB format."""
        if vec_map.ndim == 4:
            return vec_map

        raise ValueError(f"Unexpected vector map ndim={vec_map.ndim}")

    def _logits_to_distance(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert concatenated logits [B, 4, R, N] to distances [B, 4, N]."""
        if logits.ndim != 4:
            raise ValueError(f"Expected logits to be 4-D [B,4,R,N], got shape {tuple(logits.shape)}")
        probs = logits.softmax(dim=2)
        reg_max = logits.shape[2]
        bins = torch.arange(reg_max, device=logits.device, dtype=logits.dtype).view(1, 1, reg_max, 1)
        dist = torch.sum(probs * bins, dim=2)
        return dist


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

        self._enable_export_mode(model)

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

    @staticmethod
    def _enable_export_mode(model: torch.nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, Anchor2Vec):
                module.set_export_mode(True)

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
        want_names = bool(raw_names) and self.task_cfg.include_metadata
        need_simplify = self.task_cfg.simplify

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

        meta: Dict[str, str] = {
            EXPORT_SIGNATURE_KEY: EXPORT_SIGNATURE_VALUE,
            EXPORT_ANCHOR_LAYOUT_KEY: EXPORT_ANCHOR_LAYOUT_VALUE,
            "feature_layout": "channels_first",
        }

        if want_names:
            try:
                from onnx.helper import set_model_props

                class_list = [str(name) for name in list(raw_names)]
                meta.update(
                    {
                        "class_count": str(self.cfg.dataset.class_num),
                        "names": json.dumps(class_list),
                    }
                )
                set_model_props(model_onnx, meta)
                updated = True
            except Exception as exc:
                logger.warning(f"⚠️ Could not embed metadata: {exc}")
        else:
            try:
                from onnx.helper import set_model_props

                set_model_props(model_onnx, meta)
                updated = True
            except Exception as exc:
                logger.warning(f"⚠️ Could not embed export signature: {exc}")

        if updated:
            try:
                onnx.save(model_onnx, str(onnx_path))
            except Exception as exc:
                logger.warning(f"⚠️ Failed to save post-processed ONNX: {exc}")
