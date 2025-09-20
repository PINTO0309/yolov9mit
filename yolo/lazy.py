import os
import sys
from pathlib import Path

import hydra
from lightning import Trainer

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.tools.solver import InferenceModel, TrainModel, ValidateModel
from yolo.tools.exporter import ONNXExporter
from yolo.utils.logging_utils import set_seed, setup
from yolo.utils.logger import logger


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config):
    # Ensure reproducibility across DDP processes before any workload starts
    seed = getattr(cfg, "lucky_number", None)
    if seed is not None:
        set_seed(seed)
        if os.getenv("RANK", "0") == "0":
            logger.info(f":seedling: Global seed set to {seed}")

    callbacks, loggers, save_path = setup(cfg)

    if cfg.task.task == "export":
        exporter = ONNXExporter(cfg, save_path)
        exporter.run()
        return

    trainer = Trainer(
        accelerator="auto",
        max_epochs=getattr(cfg.task, "epoch", None),
        precision="16-mixed",
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=getattr(cfg, "log_every_n_steps", 500),
        gradient_clip_val=10,
        gradient_clip_algorithm="norm",
        deterministic=True,
        enable_progress_bar=not getattr(cfg, "quite", False),
        default_root_dir=save_path,
    )

    if cfg.task.task == "train":
        model = TrainModel(cfg)
        ckpt = getattr(cfg.task, "resume_ckpt", None)
        trainer.fit(model, ckpt_path=ckpt)
    elif cfg.task.task == "validation":
        model = ValidateModel(cfg)
        trainer.validate(model)
    elif cfg.task.task == "inference":
        model = InferenceModel(cfg)
        trainer.predict(model)


if __name__ == "__main__":
    main()
