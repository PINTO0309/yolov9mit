from pathlib import Path
from copy import deepcopy

import torch
from torch import Tensor

from yolo.config.config import Config, ExportConfig
from yolo.model.yolo import create_model
from yolo.tools.exporter import ONNXExporter
from yolo.utils.logger import logger


class FastModelLoader:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.compiler = cfg.task.fast_inference
        self.class_num = cfg.dataset.class_num

        self._validate_compiler()
        if cfg.weight is True:
            cfg.weight = Path("weights") / f"{cfg.model.name}.pt"
        self.weight_path = Path(cfg.weight) if cfg.weight else None
        self.model_path = None
        if self.weight_path is not None and self.compiler == "trt":
            self.model_path = str(self.weight_path.with_suffix(f".{self.compiler}"))
        self._onnx_path = None

    def _validate_compiler(self):
        if self.compiler not in ["onnx", "trt", "deploy"]:
            logger.warning(f":warning: Compiler '{self.compiler}' is not supported. Using original model.")
            self.compiler = None
        if self.cfg.device == "mps" and self.compiler == "trt":
            logger.warning(":red_apple: TensorRT does not support MPS devices. Using original model.")
            self.compiler = None

    def load_model(self, device):
        if self.compiler == "onnx":
            return self._load_onnx_model(device)
        elif self.compiler == "trt":
            return self._load_trt_model().to(device)
        elif self.compiler == "deploy":
            self.cfg.model.model.auxiliary = {}
        return create_model(self.cfg.model, class_num=self.class_num, weight_path=self.cfg.weight).to(device)

    def _load_onnx_model(self, device):
        from onnxruntime import InferenceSession

        providers = ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider"]
        onnx_path = self._ensure_onnx_model()
        try:
            session = InferenceSession(str(onnx_path), providers=providers)
            logger.info(f":rocket: Using ONNX as model backend! ({onnx_path.name})")
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model at {onnx_path}: {e}") from e
        return FusedONNXRuntime(session, device)


    def _ensure_onnx_model(self) -> Path:
        if self._onnx_path and Path(self._onnx_path).exists():
            return Path(self._onnx_path)
        if self.weight_path is None:
            raise ValueError("ONNX export requires a weight path")
        if self.weight_path.suffix.lower() == ".onnx":
            self._onnx_path = self.weight_path
            return self.weight_path

        weight_path = self.weight_path
        batch_size = int(getattr(getattr(self.cfg.task, "data", None), "batch_size", 1))
        dynamic_batch = bool(getattr(self.cfg.task, "dynamic_batch", False))
        image_size_cfg = getattr(self.cfg, 'image_size', None)
        if image_size_cfg is None:
            image_size_cfg = getattr(getattr(self.cfg.task, 'data', None), 'image_size', None)
        if image_size_cfg is None:
            raise ValueError('image_size must be provided for ONNX export')
        image_size = [int(image_size_cfg[0]), int(image_size_cfg[1])]

        export_task = ExportConfig(
            task="export",
            batch_size=batch_size,
            opset=13,
            simplify=True,
            half=False,
            dynamic_batch=dynamic_batch,
            apply_sigmoid=True,
            include_metadata=True,
            output_path=None,
            image_size=image_size,
        )

        export_cfg = deepcopy(self.cfg)
        export_cfg.task = export_task
        exporter = ONNXExporter(export_cfg, weight_path.parent)
        target_path = exporter._resolve_output_path()
        if not target_path.exists():
            target_path = exporter.run()
        self._onnx_path = target_path
        return target_path


    def _load_trt_model(self):
        from torch2trt import TRTModule

        try:
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(self.model_path))
            logger.info(":rocket: Using TensorRT as MODEL frameworks!")
        except FileNotFoundError:
            logger.warning(f"🈳 No found model weight at {self.model_path}")
            model_trt = self._create_trt_model()
        return model_trt

    def _create_trt_model(self):
        from torch2trt import torch2trt

        model = create_model(self.cfg.model, class_num=self.class_num, weight_path=self.cfg.weight).eval()
        dummy_input = torch.ones((1, 3, *self.cfg.image_size)).cuda()
        logger.info(f"♻️ Creating TensorRT model")
        model_trt = torch2trt(model.cuda(), [dummy_input])
        torch.save(model_trt.state_dict(), self.model_path)
        logger.info(f":inbox_tray: TensorRT model saved to {self.model_path}")
        return model_trt



class FusedONNXRuntime:
    def __init__(self, session, device):
        self.session = session
        self.device = device
        self.input_name = session.get_inputs()[0].name
        self.fused_onnx_output = True

    def to(self, device):
        self.device = device
        return self

    def __call__(self, x: Tensor):
        input_cpu = x.detach().to('cpu')
        outputs = self.session.run(None, {self.input_name: input_cpu.numpy()})
        fused = torch.from_numpy(outputs[0]).to(x.device)
        return {"Main": fused}
