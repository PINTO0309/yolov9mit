from collections import defaultdict
from pathlib import Path
from queue import Empty, Queue
from statistics import mean
from threading import Event, Thread
from typing import Generator, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
import random
from rich.progress import track
from torch import Tensor
from torch.utils.data import BatchSampler, DataLoader, Dataset

SIZE_THRESHOLDS = (32 * 32, 96 * 96)

from yolo.config.config import DataConfig, DatasetConfig
from yolo.tools.data_augmentation import *
from yolo.tools.data_augmentation import AugmentationComposer
from omegaconf import DictConfig
from yolo.tools.dataset_preparation import prepare_dataset
from yolo.utils.dataset_utils import (
    convert_bboxes,
    create_image_metadata,
    locate_label_paths,
    scale_segmentation,
    tensorlize,
)
from yolo.utils.logger import logger


class YoloDataset(Dataset):
    def __init__(self, data_cfg: DataConfig, dataset_cfg: DatasetConfig, phase: str = "train2017"):
        augment_cfg = data_cfg.data_augment
        self.image_size = data_cfg.image_size
        phase_name = dataset_cfg.get(phase, phase)
        self.batch_size = data_cfg.batch_size
        self.dynamic_shape = getattr(data_cfg, "dynamic_shape", False)
        self.base_size = mean(self.image_size)

        transforms = []
        for aug, params in augment_cfg.items():
            cls = eval(aug)
            # If params is a scalar (e.g., 0.5), pass as positional argument
            if isinstance(params, (int, float)):
                transforms.append(cls(params))
            # If params is a mapping (OmegaConf DictConfig or dict), expand as kwargs
            elif isinstance(params, (dict, DictConfig)):
                transforms.append(cls(**dict(params)))
            # Fallback: pass through directly
            else:
                transforms.append(cls(params))
        self.transform = AugmentationComposer(transforms, self.image_size, self.base_size)
        self.transform.get_more_data = self.get_more_data
        dataset_path = Path(dataset_cfg.path)
        raw_data = self.load_data(dataset_path, phase_name)

        self.size_stats = None
        self.global_bucket_share = {"small": 0.0, "medium": 0.0, "large": 0.0}
        self.bucket_to_indices = {"small": [], "medium": [], "large": []}
        self.rare_buckets: List[str] = []

        is_train_phase = phase.startswith("train")
        oversample_enabled = bool(getattr(data_cfg, "class_biased_oversampling", False) and is_train_phase)
        batch_bias_enabled = bool(getattr(data_cfg, "class_biased_batch_formation", False) and is_train_phase)

        pre_stats = None
        post_stats = None
        total_duplicates = 0
        bucket_deltas = {"small": 0, "medium": 0, "large": 0}

        if is_train_phase and (oversample_enabled or batch_bias_enabled):
            pre_stats = self._compute_size_statistics(raw_data, dataset_path, dataset_cfg)

        if oversample_enabled and pre_stats is not None:
            raw_data, post_stats, total_duplicates, bucket_deltas = self._apply_class_biased_oversampling(
                raw_data, pre_stats, dataset_cfg
            )
        else:
            post_stats = pre_stats

        if is_train_phase and (oversample_enabled or batch_bias_enabled) and pre_stats is not None:
            self._log_size_distribution(pre_stats, post_stats, dataset_cfg)

        if oversample_enabled and pre_stats is not None:
            pre_share = self._compute_global_share(pre_stats["global_bucket_counts"])
            post_share = self._compute_global_share(post_stats["global_bucket_counts"])
            summary = ", ".join(
                f"{bucket}:Δ{int(bucket_deltas.get(bucket, 0))}" for bucket in ("small", "medium", "large")
            )
            logger.info(
                f":repeat: Applied class_biased_oversampling (duplicates={total_duplicates}, dataset size {len(pre_stats['image_bucket_tags'])} -> {len(raw_data)})."
            )
            logger.info(
                ":bar_chart: Global size share "
                f"before (S={pre_share['small']:.3f}, M={pre_share['medium']:.3f}, L={pre_share['large']:.3f}) "
                f"after (S={post_share['small']:.3f}, M={post_share['medium']:.3f}, L={post_share['large']:.3f}); "
                f"increments {summary}"
            )

        if post_stats is not None:
            self._finalize_size_statistics(post_stats, dataset_cfg)

        self.img_paths, self.bboxes, self.ratios = tensorlize(raw_data)

    def load_data(self, dataset_path: Path, phase_name: str):
        """
        Loads data from a cache or generates a new cache for a specific dataset phase.

        Parameters:
            dataset_path (Path): The root path to the dataset directory.
            phase_name (str): The specific phase of the dataset (e.g., 'train', 'test') to load or generate data for.

        Returns:
            dict: The loaded data from the cache for the specified phase.
        """
        cache_path = dataset_path / f"{phase_name}.pache"

        if not cache_path.exists():
            logger.info(f":factory: Generating {phase_name} cache")
            data = self.filter_data(dataset_path, phase_name, self.dynamic_shape)
            torch.save(data, cache_path)
        else:
            try:
                data = torch.load(cache_path, weights_only=False)
            except Exception as e:
                logger.error(
                    f":rotating_light: Failed to load the cache at '{cache_path}'.\n"
                    ":rotating_light: This may be caused by using cache from different other YOLO.\n"
                    ":rotating_light: Please clean the cache and try running again."
                )
                raise e
            logger.info(f":package: Loaded {phase_name} cache, there are {len(data)} data in total.")
        return data

    def filter_data(self, dataset_path: Path, phase_name: str, sort_image: bool = False) -> list:
        """
        Filters and collects dataset information by pairing images with their corresponding labels.

        Parameters:
            images_path (Path): Path to the directory containing image files.
            labels_path (str): Path to the directory containing label files.
            sort_image (bool): If True, sorts the dataset by the width-to-height ratio of images in descending order.

        Returns:
            list: A list of tuples, each containing the path to an image file and its associated segmentation as a tensor.
        """
        images_path = dataset_path / "images" / phase_name
        labels_path, data_type = locate_label_paths(dataset_path, phase_name)
        file_list, adjust_path = dataset_path / f"{phase_name}.txt", False
        if file_list.exists():
            data_type, adjust_path = "txt", True
            # TODO: should i sort by name?
            with open(file_list, "r") as file:
                images_list = [dataset_path / line.rstrip() for line in file]
            labels_list = [
                Path(str(image_path).replace("images", "labels")).with_suffix(".txt") for image_path in images_list
            ]
        else:
            images_list = sorted([p.name for p in Path(images_path).iterdir() if p.is_file()])

        if data_type == "json":
            annotations_index, image_info_dict = create_image_metadata(labels_path)

        data = []
        valid_inputs = 0
        for idx, image_name in enumerate(track(images_list, description="Filtering data")):
            if not adjust_path and not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            image_id = Path(image_name).stem

            if data_type == "json":
                image_info = image_info_dict.get(image_id, None)
                if image_info is None:
                    continue
                annotations = annotations_index.get(image_info["id"], [])
                image_seg_annotations = scale_segmentation(annotations, image_info)
            elif data_type == "txt":
                label_path = labels_list[idx] if adjust_path else labels_path / f"{image_id}.txt"
                if not label_path.is_file():
                    image_seg_annotations = []
                else:
                    with open(label_path, "r") as file:
                        annotations = [list(map(float, line.strip().split())) for line in file]
                        image_seg_annotations = convert_bboxes(annotations)
            else:
                image_seg_annotations = []

            labels = self.load_valid_labels(image_id, image_seg_annotations)
            img_path = image_name if adjust_path else images_path / image_name
            if sort_image:
                with Image.open(img_path) as img:
                    width, height = img.size
            else:
                width, height = 0, 1
            data.append((img_path, labels, width / height))
            if len(image_seg_annotations) != 0:
                valid_inputs += 1

        data = sorted(data, key=lambda x: x[2], reverse=True)

        logger.info(f"Recorded {valid_inputs}/{len(images_list)} valid inputs")
        return data

    def load_valid_labels(self, label_path: str, seg_data_one_img: list) -> Union[Tensor, None]:
        """
        Loads valid COCO style segmentation data (values between [0, 1]) and converts it to bounding box coordinates
        by finding the minimum and maximum x and y values.

        Parameters:
            label_path (str): The filepath to the label file containing annotation data.
            seg_data_one_img (list): The actual list of annotations (in segmentation format)

        Returns:
            Tensor or None: A tensor of all valid bounding boxes if any are found; otherwise, None.
        """
        bboxes = []
        for seg_data in seg_data_one_img:
            cls = seg_data[0]
            points = np.array(seg_data[1:]).reshape(-1, 2).clip(0, 1)
            valid_points = points[(points >= 0) & (points <= 1)].reshape(-1, 2)
            if valid_points.size > 1:
                bbox = torch.tensor([cls, *valid_points.min(axis=0), *valid_points.max(axis=0)])
                bboxes.append(bbox)

        if bboxes:
            return torch.stack(bboxes)
        else:
            logger.warning(f"No valid BBox in {label_path}")
            return torch.zeros((0, 5))

    def get_data(self, idx):
        img_path, bboxes = self.img_paths[idx], self.bboxes[idx]
        valid_mask = bboxes[:, 0] != -1
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        return img, torch.from_numpy(bboxes[valid_mask]), img_path

    def get_more_data(self, num: int = 1):
        indices = torch.randint(0, len(self), (num,))
        return [self.get_data(idx)[:2] for idx in indices]

    def _resolve_image_path(self, img_path, dataset_path: Path) -> Path:
        p = Path(img_path)
        if p.exists():
            return p
        candidate = dataset_path / p
        return candidate if candidate.exists() else p

    @staticmethod
    def _bucket_from_area(area: float) -> str:
        if area < SIZE_THRESHOLDS[0]:
            return "small"
        if area < SIZE_THRESHOLDS[1]:
            return "medium"
        return "large"

    def _compute_size_statistics(self, data, dataset_path: Path, dataset_cfg: DatasetConfig):
        class_num = int(getattr(dataset_cfg, "class_num", 0) or 0)
        if class_num <= 0:
            return None

        class_bucket_counts = [defaultdict(int) for _ in range(class_num)]
        class_bucket_indices = [defaultdict(set) for _ in range(class_num)]
        image_class_bucket_counts: List[dict] = []
        image_bucket_tags: List[set] = []
        global_bucket_counts = defaultdict(int)
        image_size_cache = {}

        if data:
            iterator = track(
                enumerate(data),
                total=len(data),
                description="Analyzing dataset size buckets",
            )
        else:
            iterator = enumerate(data)

        for idx, (img_path, labels, _) in iterator:
            per_image_counts = defaultdict(int)
            tags: set = set()

            if isinstance(labels, torch.Tensor):
                boxes = labels.to(torch.float32)
            else:
                boxes = torch.tensor(labels, dtype=torch.float32)

            if boxes.numel() == 0:
                image_class_bucket_counts.append(dict(per_image_counts))
                image_bucket_tags.append(tags)
                continue

            real_path = self._resolve_image_path(img_path, dataset_path)
            if real_path not in image_size_cache:
                try:
                    with Image.open(real_path) as img:
                        image_size_cache[real_path] = img.size
                except Exception:
                    logger.warning(
                        f":warning: Failed to read image size for {real_path}, skipping statistics contribution"
                    )
                    image_class_bucket_counts.append(dict(per_image_counts))
                    image_bucket_tags.append(tags)
                    continue

            width, height = image_size_cache[real_path]
            if width <= 0 or height <= 0:
                image_class_bucket_counts.append(dict(per_image_counts))
                image_bucket_tags.append(tags)
                continue

            for box in boxes:
                cls = int(box[0].item())
                if cls < 0 or cls >= class_num:
                    continue
                x1, y1, x2, y2 = box[1:5].clamp(0.0, 1.0)
                bw = max((x2 - x1) * width, 0.0)
                bh = max((y2 - y1) * height, 0.0)
                area = bw * bh
                if area <= 0:
                    continue
                bucket = self._bucket_from_area(area)
                per_image_counts[(cls, bucket)] += 1
                tags.add(bucket)
                class_bucket_counts[cls][bucket] += 1
                class_bucket_indices[cls][bucket].add(idx)
                global_bucket_counts[bucket] += 1

            image_class_bucket_counts.append(dict(per_image_counts))
            image_bucket_tags.append(tags)

        return {
            "class_bucket_counts": class_bucket_counts,
            "class_bucket_indices": class_bucket_indices,
            "image_class_bucket_counts": image_class_bucket_counts,
            "image_bucket_tags": image_bucket_tags,
            "global_bucket_counts": global_bucket_counts,
        }

    def _log_size_distribution(self, pre_stats, post_stats, dataset_cfg: DatasetConfig):
        if pre_stats is None or post_stats is None:
            return

        class_num = int(getattr(dataset_cfg, "class_num", 0) or 0)
        if class_num <= 0:
            return

        class_names = getattr(dataset_cfg, "class_list", None)
        if not isinstance(class_names, (list, tuple)) or len(class_names) < class_num:
            class_names = [f"class_{i}" for i in range(class_num)]

        header = (
            f"{'Class':<18}{'Pre-S':>10}{'Pre-M':>10}{'Pre-L':>10}"
            f"{'Post-S':>10}{'Post-M':>10}{'Post-L':>10}"
        )
        lines = [header, "-" * len(header)]

        for cls_idx in range(class_num):
            name = str(class_names[cls_idx]) if cls_idx < len(class_names) else f"class_{cls_idx}"
            pre_counts = pre_stats["class_bucket_counts"][cls_idx] if pre_stats else {}
            post_counts = post_stats["class_bucket_counts"][cls_idx] if post_stats else {}
            pre_small = int(pre_counts.get("small", 0))
            pre_medium = int(pre_counts.get("medium", 0))
            pre_large = int(pre_counts.get("large", 0))
            post_small = int(post_counts.get("small", 0))
            post_medium = int(post_counts.get("medium", 0))
            post_large = int(post_counts.get("large", 0))
            lines.append(
                f"{name:<18}{pre_small:>10}{pre_medium:>10}{pre_large:>10}{post_small:>10}{post_medium:>10}{post_large:>10}"
            )

        pre_global = pre_stats["global_bucket_counts"] if pre_stats else defaultdict(int)
        post_global = post_stats["global_bucket_counts"] if post_stats else defaultdict(int)
        lines.append("-" * len(header))
        lines.append(
            f"{'TOTAL':<18}"
            f"{int(pre_global.get('small', 0)):>10}"
            f"{int(pre_global.get('medium', 0)):>10}"
            f"{int(pre_global.get('large', 0)):>10}"
            f"{int(post_global.get('small', 0)):>10}"
            f"{int(post_global.get('medium', 0)):>10}"
            f"{int(post_global.get('large', 0)):>10}"
        )
        table_text = "\n".join(lines)
        logger.info(":bar_chart: Size distribution per class (pre vs. post adjustment)\n" + table_text)

    @staticmethod
    def _compute_global_share(global_bucket_counts) -> dict:
        total = sum(int(v) for v in global_bucket_counts.values())
        if total <= 0:
            return {"small": 0.0, "medium": 0.0, "large": 0.0}
        return {
            "small": int(global_bucket_counts.get("small", 0)) / total,
            "medium": int(global_bucket_counts.get("medium", 0)) / total,
            "large": int(global_bucket_counts.get("large", 0)) / total,
        }

    def _finalize_size_statistics(self, stats, dataset_cfg: DatasetConfig):
        self.size_stats = stats
        self.global_bucket_share = self._compute_global_share(stats["global_bucket_counts"])

        bucket_to_indices = {"small": [], "medium": [], "large": []}
        for idx, tags in enumerate(stats["image_bucket_tags"]):
            for bucket in tags:
                bucket_to_indices.setdefault(bucket, []).append(idx)

        for bucket, indices in bucket_to_indices.items():
            random.shuffle(indices)

        self.bucket_to_indices = bucket_to_indices
        self.rare_buckets = self._select_rare_buckets(self.global_bucket_share, bucket_to_indices)

    @staticmethod
    def _select_rare_buckets(global_share: dict, bucket_to_indices: dict) -> List[str]:
        available = {bucket: global_share.get(bucket, 0.0) for bucket in ("small", "medium", "large") if bucket_to_indices.get(bucket)}
        if not available:
            return []
        threshold = 0.2
        rare = [bucket for bucket, share in available.items() if share <= threshold]
        if not rare:
            rare = [min(available, key=available.get)]
        return rare

    def _apply_class_biased_oversampling(self, data, stats, dataset_cfg: DatasetConfig):
        """Duplicate images for rarity-compensation after analysing per-class size distribution."""

        class_num = int(getattr(dataset_cfg, "class_num", 0) or 0)
        if class_num <= 0 or stats is None:
            logger.warning(":warning: class_biased_oversampling requested but dataset statistics are unavailable; skipping")
            return data, stats, 0, {"small": 0, "medium": 0, "large": 0}

        max_dup_factor = 5
        growth_tolerance = 0.1  # require >10% deficit before increasing

        class_bucket_counts = stats["class_bucket_counts"]
        class_bucket_indices = stats["class_bucket_indices"]
        image_class_bucket_counts = stats["image_class_bucket_counts"]
        image_bucket_tags = stats["image_bucket_tags"]
        global_bucket_counts = stats["global_bucket_counts"]

        total_global = sum(int(v) for v in global_bucket_counts.values())
        if total_global == 0:
            logger.warning(":warning: class_biased_oversampling found no valid bounding boxes; skipping")
            return data, stats, 0, {"small": 0, "medium": 0, "large": 0}

        global_share = self._compute_global_share(global_bucket_counts)
        eps = 1e-9
        for bucket, ratio in list(global_share.items()):
            if ratio <= 0:
                global_share[bucket] = eps

        image_multipliers = [1] * len(data)
        for cls, counts in enumerate(class_bucket_counts):
            if not counts:
                continue
            class_total = sum(int(v) for v in counts.values())
            if class_total <= 0:
                continue
            for bucket in ("small", "medium", "large"):
                count = int(counts.get(bucket, 0))
                if count <= 0:
                    continue
                expected = class_total * global_share.get(bucket, eps)
                ratio = expected / count if count > 0 else 0.0
                if ratio <= 1.0 + growth_tolerance:
                    continue
                dup_factor = min(int(np.ceil(ratio)), max_dup_factor)
                if dup_factor <= 1:
                    continue
                for img_idx in class_bucket_indices[cls][bucket]:
                    image_multipliers[img_idx] = max(image_multipliers[img_idx], dup_factor)

        if all(factor == 1 for factor in image_multipliers):
            return data, stats, 0, {"small": 0, "medium": 0, "large": 0}

        expanded_indices: List[int] = []
        for idx, factor in enumerate(image_multipliers):
            expanded_indices.extend([idx] * factor)
        random.shuffle(expanded_indices)

        new_data = []
        new_image_class_counts: List[dict] = []
        new_image_bucket_tags: List[set] = []
        new_class_bucket_counts = [defaultdict(int) for _ in range(class_num)]
        new_class_bucket_indices = [defaultdict(set) for _ in range(class_num)]
        new_global_counts = defaultdict(int)

        if expanded_indices:
            iterator_new = track(
                enumerate(expanded_indices),
                total=len(expanded_indices),
                description="Applying class-biased oversampling",
            )
        else:
            iterator_new = enumerate(expanded_indices)

        for new_idx, original_idx in iterator_new:
            item = data[original_idx]
            per_image_counts = image_class_bucket_counts[original_idx]
            if not isinstance(per_image_counts, dict):
                per_image_counts = dict(per_image_counts)
            tags = image_bucket_tags[original_idx]
            new_data.append(item)
            new_image_class_counts.append(dict(per_image_counts))
            new_image_bucket_tags.append(set(tags))
            for (cls, bucket), count in per_image_counts.items():
                new_class_bucket_counts[cls][bucket] += int(count)
                new_class_bucket_indices[cls][bucket].add(new_idx)
                new_global_counts[bucket] += int(count)

        total_duplicates = sum(max(f - 1, 0) for f in image_multipliers)
        bucket_deltas = {
            bucket: int(new_global_counts.get(bucket, 0)) - int(global_bucket_counts.get(bucket, 0))
            for bucket in ("small", "medium", "large")
        }

        post_stats = {
            "class_bucket_counts": new_class_bucket_counts,
            "class_bucket_indices": new_class_bucket_indices,
            "image_class_bucket_counts": new_image_class_counts,
            "image_bucket_tags": new_image_bucket_tags,
            "global_bucket_counts": new_global_counts,
        }

        return new_data, post_stats, total_duplicates, bucket_deltas

    def _update_image_size(self, idx: int) -> None:
        """Update image size based on dynamic shape and batch settings."""
        batch_start_idx = (idx // self.batch_size) * self.batch_size
        image_ratio = self.ratios[batch_start_idx].clip(1 / 3, 3)
        shift = ((self.base_size / 32 * (image_ratio - 1)) // (image_ratio + 1)) * 32

        self.image_size = [int(self.base_size + shift), int(self.base_size - shift)]
        self.transform.pad_resize.set_size(self.image_size)

    def __getitem__(self, idx) -> Tuple[Image.Image, Tensor, Tensor, List[str]]:
        img, bboxes, img_path = self.get_data(idx)

        if self.dynamic_shape:
            self._update_image_size(idx)

        img, bboxes, rev_tensor = self.transform(img, bboxes)
        bboxes[:, [1, 3]] *= self.image_size[0]
        bboxes[:, [2, 4]] *= self.image_size[1]
        return img, bboxes, rev_tensor, img_path

    def __len__(self) -> int:
        return len(self.bboxes)


class ClassBiasedBatchSampler(BatchSampler):
    def __init__(self, dataset: YoloDataset, batch_size: int, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        if self.batch_size <= 0:
            raise ValueError("batch_size should be a positive integer")

        generator_seed = torch.initial_seed()
        rng = random.Random(generator_seed)

        all_indices = list(range(len(self.dataset)))
        rng.shuffle(all_indices)

        rare_buckets = [
            bucket
            for bucket in getattr(self.dataset, "rare_buckets", [])
            if self.dataset.bucket_to_indices.get(bucket)
        ]

        bucket_orders = {}
        bucket_positions = {}
        for bucket in rare_buckets:
            pool = list(self.dataset.bucket_to_indices.get(bucket, []))
            rng.shuffle(pool)
            bucket_orders[bucket] = pool
            bucket_positions[bucket] = 0

        used = set()
        pointer = 0
        num_indices = len(all_indices)

        while pointer < num_indices:
            batch = []

            for bucket in rare_buckets:
                idx = self._pop_bucket_idx(bucket, bucket_orders, bucket_positions, used, rng)
                if idx is not None and idx not in batch:
                    batch.append(idx)
                    used.add(idx)

            while len(batch) < self.batch_size and pointer < num_indices:
                idx = all_indices[pointer]
                pointer += 1
                if idx in used:
                    continue
                batch.append(idx)
                used.add(idx)

            if len(batch) < self.batch_size and self.drop_last:
                break

            if batch:
                yield batch

        remaining = [idx for idx in all_indices if idx not in used]
        current = []
        for idx in remaining:
            current.append(idx)
            if len(current) == self.batch_size:
                yield current
                current = []
        if current and not self.drop_last:
            yield current

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    @staticmethod
    def _pop_bucket_idx(bucket, bucket_orders, bucket_positions, used, rng):
        pool = bucket_orders.get(bucket)
        if not pool:
            return None
        pos = bucket_positions.get(bucket, 0)
        total = len(pool)
        attempts = 0
        while attempts < total:
            if pos >= len(pool):
                rng.shuffle(pool)
                bucket_orders[bucket] = pool
                pos = 0
            idx = pool[pos]
            pos += 1
            attempts += 1
            if idx in used:
                continue
            bucket_positions[bucket] = pos
            return idx
        bucket_positions[bucket] = pos
        return None


def collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tensor]]:
    """
    A collate function to handle batching of images and their corresponding targets.

    Args:
        batch (list of tuples): Each tuple contains:
            - image (Tensor): The image tensor.
            - labels (Tensor): The tensor of labels for the image.

    Returns:
        Tuple[Tensor, List[Tensor]]: A tuple containing:
            - A tensor of batched images.
            - A list of tensors, each corresponding to bboxes for each image in the batch.
    """
    batch_size = len(batch)
    target_sizes = [item[1].size(0) for item in batch]
    # TODO: Improve readability of these process
    # TODO: remove maxBbox or reduce loss function memory usage
    batch_targets = torch.zeros(batch_size, min(max(target_sizes), 100), 5)
    batch_targets[:, :, 0] = -1
    for idx, target_size in enumerate(target_sizes):
        batch_targets[idx, : min(target_size, 100)] = batch[idx][1][:100]

    batch_images, _, batch_reverse, batch_path = zip(*batch)
    batch_images = torch.stack(batch_images)
    batch_reverse = torch.stack(batch_reverse)

    return batch_size, batch_images, batch_targets, batch_reverse, batch_path


def create_dataloader(data_cfg: DataConfig, dataset_cfg: DatasetConfig, task: str = "train"):
    if task == "inference":
        return StreamDataLoader(data_cfg)

    if getattr(dataset_cfg, "auto_download", False):
        prepare_dataset(dataset_cfg, task)
    dataset = YoloDataset(data_cfg, dataset_cfg, task)

    use_class_biased_batch = bool(
        task.startswith("train") and getattr(data_cfg, "class_biased_batch_formation", False)
    )

    is_training_task = isinstance(task, str) and task.lower().startswith("train")
    shuffle_data = bool(getattr(data_cfg, "shuffle", False)) if is_training_task else False

    if use_class_biased_batch:
        batch_sampler = ClassBiasedBatchSampler(dataset, data_cfg.batch_size)
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=data_cfg.cpu_num,
            pin_memory=data_cfg.pin_memory,
            collate_fn=collate_fn,
        )

    return DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.cpu_num,
        pin_memory=data_cfg.pin_memory,
        collate_fn=collate_fn,
        shuffle=shuffle_data,
    )


class StreamDataLoader:
    def __init__(self, data_cfg: DataConfig):
        self.source = data_cfg.source
        self.running = True
        self._stopped = False
        max_samples = getattr(data_cfg, "max_samples", None)
        self.max_samples: Optional[int] = None
        if max_samples is not None:
            try:
                value = int(max_samples)
                if value > 0:
                    self.max_samples = value
            except (TypeError, ValueError):
                logger.warning(f":warning: Ignoring invalid inference max_samples value: {max_samples}")
        self._loaded_samples = 0
        self._returned_samples = 0
        self.is_stream = isinstance(self.source, int) or str(self.source).lower().startswith("rtmp://")

        self.transform = AugmentationComposer([], data_cfg.image_size)
        self.stop_event = Event()
        self._frame_index = 0

        if self.is_stream:
            import cv2

            self.cap = cv2.VideoCapture(self.source)
        else:
            self.source = Path(self.source)
            self.queue = Queue()
            self.thread = Thread(target=self.load_source)
            self.thread.start()

    def load_source(self):
        if self.source.is_dir():  # image folder
            self.load_image_folder(self.source)
        elif any(self.source.suffix.lower().endswith(ext) for ext in [".mp4", ".avi", ".mkv"]):  # Video file
            self.load_video_file(self.source)
        else:  # Single image
            self.process_image(self.source)

    def load_image_folder(self, folder):
        folder_path = Path(folder)
        for file_path in folder_path.rglob("*"):
            if self.stop_event.is_set():
                break
            if file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                if not self.process_image(file_path):
                    break

    def process_image(self, image_path):
        if self.max_samples is not None and self._loaded_samples >= self.max_samples:
            return False
        image = Image.open(image_path).convert("RGB")
        if image is None:
            raise ValueError(f"Error loading image: {image_path}")
        return self.process_frame(image, source_path=image_path, is_single_image=True)

    def load_video_file(self, video_path):
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            if not self.process_frame(frame, source_path=self.source):
                break
        cap.release()

    def process_frame(self, frame, source_path=None, is_single_image: bool = False):
        if self.max_samples is not None and self._loaded_samples >= self.max_samples:
            self.stop_event.set()
            self.running = False
            return False
        if isinstance(frame, np.ndarray):
            # TODO: we don't need cv2
            import cv2

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
        origin_frame = frame
        frame, _, rev_tensor = self.transform(frame, torch.zeros(0, 5))
        frame = frame[None]
        rev_tensor = rev_tensor[None]
        meta = {
            "source_path": str(source_path) if source_path is not None else None,
            "frame_index": self._frame_index,
            "is_single_image": bool(is_single_image),
        }
        self._frame_index += 1
        self._loaded_samples += 1
        if not self.is_stream:
            self.queue.put((frame, rev_tensor, origin_frame, meta))
        else:
            self.current_frame = (frame, rev_tensor, origin_frame, meta)
        if self.max_samples is not None and self._loaded_samples >= self.max_samples:
            self.stop_event.set()
            if self.is_stream:
                self.running = False
        return True

    def __iter__(self) -> Generator[Tensor, None, None]:
        return self

    def __next__(self) -> Tensor:
        if self.max_samples is not None and self._returned_samples >= self.max_samples:
            self.stop()
            raise StopIteration
        if self.is_stream:
            if not self.running:
                self.stop()
                raise StopIteration
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                raise StopIteration
            if not self.process_frame(frame, source_path=self.source):
                self.stop()
                raise StopIteration
            self._returned_samples += 1
            if self.max_samples is not None and self._returned_samples >= self.max_samples:
                self.stop_event.set()
                self.stop()
            return self.current_frame
        else:
            try:
                frame = self.queue.get(timeout=1)
            except Empty:
                self.stop()
                raise StopIteration
            self._returned_samples += 1
            if self.max_samples is not None and self._returned_samples >= self.max_samples:
                self.stop_event.set()
                self.stop()
            return frame

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        self.running = False
        self.stop_event.set()
        if self.is_stream:
            cap = getattr(self, "cap", None)
            if cap is not None:
                cap.release()
        else:
            thread = getattr(self, "thread", None)
            if thread is not None:
                thread.join(timeout=1)

    def __len__(self):
        if self.max_samples is not None:
            return self.max_samples
        return self.queue.qsize() if not self.is_stream else 0
