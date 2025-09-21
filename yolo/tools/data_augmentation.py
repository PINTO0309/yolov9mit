from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
import inspect
import cv2

class AugmentationComposer:
    """Composes several transforms together."""

    def __init__(self, transforms, image_size: int = [640, 640], base_size: int = 640):
        self.transforms = transforms
        # TODO: handle List of image_size [640, 640]
        self.pad_resize = PadAndResize(image_size)
        self.base_size = base_size

        for transform in self.transforms:
            if hasattr(transform, "set_parent"):
                transform.set_parent(self)

    def __call__(self, image, boxes=torch.zeros(0, 5)):
        for transform in self.transforms:
            image, boxes = transform(image, boxes)
        image, boxes, rev_tensor = self.pad_resize(image, boxes)
        image = TF.to_tensor(image)
        return image, boxes, rev_tensor


class RemoveOutliers:
    """Removes outlier bounding boxes that are too small or have invalid dimensions."""

    def __init__(self, min_box_area=1e-8):
        """
        Args:
            min_box_area (float): Minimum area for a box to be kept, as a fraction of the image area.
        """
        self.min_box_area = min_box_area

    def __call__(self, image, boxes):
        """
        Args:
            image (PIL.Image): The cropped image.
            boxes (torch.Tensor): Bounding boxes in normalized coordinates (x_min, y_min, x_max, y_max).
        Returns:
            PIL.Image: The input image (unchanged).
            torch.Tensor: Filtered bounding boxes.
        """
        box_areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])

        valid_boxes = (box_areas > self.min_box_area) & (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 4] > boxes[:, 2])

        return image, boxes[valid_boxes]


class PadAndResize:
    def __init__(self, image_size, background_color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        """Initialize the object with the target image size."""
        self.target_width, self.target_height = image_size  # (w, h)
        self.pad_color = tuple(background_color)
        self.auto = bool(auto)
        self.scaleup = bool(scaleup)
        self.stride = int(stride)

    def set_size(self, image_size: List[int]):
        self.target_width, self.target_height = image_size

    def set_options(self, *, auto=None, scaleup=None, pad_color=None, stride=None):
        """Option to dynamically switch behavior from the caller (DataLoader, etc.)."""
        if auto is not None:
            self.auto = bool(auto)
        if scaleup is not None:
            self.scaleup = bool(scaleup)
        if pad_color is not None:
            self.pad_color = tuple(pad_color)
        if stride is not None:
            self.stride = int(stride)

    def __call__(self, image: Image.Image, boxes):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))

        img_w, img_h = image.size
        target_w, target_h = self.target_width, self.target_height
        if target_w <= 0 or target_h <= 0:
            raise ValueError("Target image size must be positive")

        gain = min(target_w / img_w, target_h / img_h) if img_w and img_h else 1.0
        if not self.scaleup:
            gain = min(gain, 1.0)

        new_w = max(int(round(img_w * gain)), 1)
        new_h = max(int(round(img_h * gain)), 1)
        dw = target_w - new_w
        dh = target_h - new_h

        if self.auto:
            dw %= self.stride
            dh %= self.stride

        dw *= 0.5
        dh *= 0.5

        pad_left = int(round(dw - 0.1))
        pad_top = int(round(dh - 0.1))

        resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        padded_image = Image.new("RGB", (target_w, target_h), self.pad_color)
        padded_image.paste(resized_image, (pad_left, pad_top))

        if boxes is None:
            boxes_tensor = torch.zeros(0, 5, dtype=torch.float32)
        elif isinstance(boxes, torch.Tensor):
            boxes_tensor = boxes.clone()
        else:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)

        if boxes_tensor.numel() > 0:
            boxes_tensor[:, [1, 3]] = (boxes_tensor[:, [1, 3]] * new_w + pad_left) / target_w
            boxes_tensor[:, [2, 4]] = (boxes_tensor[:, [2, 4]] * new_h + pad_top) / target_h

        ratio_x = new_w / img_w if img_w else 1.0
        ratio_y = new_h / img_h if img_h else 1.0
        transform_info = {
            "ratio": (ratio_x, ratio_y),  # scaling applied to width/height
            "pad": (pad_left, pad_top),   # padding offset applied after resize
            "size": (target_h, target_w),  # tensor size (h, w)
            "auto": self.auto,
            "scaleup": self.scaleup,
            "stride": self.stride,
        }

        return padded_image, boxes_tensor, transform_info


class HorizontalFlip:
    """Randomly horizontally flips the image along with the bounding boxes."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, boxes):
        if torch.rand(1) < self.prob:
            image = TF.hflip(image)
            boxes[:, [1, 3]] = 1 - boxes[:, [3, 1]]
        return image, boxes


class VerticalFlip:
    """Randomly vertically flips the image along with the bounding boxes."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, boxes):
        if torch.rand(1) < self.prob:
            image = TF.vflip(image)
            boxes[:, [2, 4]] = 1 - boxes[:, [4, 2]]
        return image, boxes


class Mosaic:
    """Applies the Mosaic augmentation to a batch of images and their corresponding boxes."""

    def __init__(self, prob=0.5):
        self.prob = prob
        self.parent = None

    def set_parent(self, parent):
        self.parent = parent

    def __call__(self, image, boxes):
        if torch.rand(1) >= self.prob:
            return image, boxes

        assert self.parent is not None, "Parent is not set. Mosaic cannot retrieve image size."

        img_sz = self.parent.base_size  # Assuming `image_size` is defined in parent
        more_data = self.parent.get_more_data(3)  # get 3 more images randomly

        data = [(image, boxes)] + more_data
        mosaic_image = Image.new("RGB", (2 * img_sz, 2 * img_sz), (114, 114, 114))
        vectors = np.array([(-1, -1), (0, -1), (-1, 0), (0, 0)])
        center = np.array([img_sz, img_sz])
        all_labels = []

        for (image, boxes), vector in zip(data, vectors):
            this_w, this_h = image.size
            coord = tuple(center + vector * np.array([this_w, this_h]))

            mosaic_image.paste(image, coord)
            xmin, ymin, xmax, ymax = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
            xmin = (xmin * this_w + coord[0]) / (2 * img_sz)
            xmax = (xmax * this_w + coord[0]) / (2 * img_sz)
            ymin = (ymin * this_h + coord[1]) / (2 * img_sz)
            ymax = (ymax * this_h + coord[1]) / (2 * img_sz)

            adjusted_boxes = torch.stack([boxes[:, 0], xmin, ymin, xmax, ymax], dim=1)
            all_labels.append(adjusted_boxes)

        all_labels = torch.cat(all_labels, dim=0)
        mosaic_image = mosaic_image.resize((img_sz, img_sz))
        return mosaic_image, all_labels


class MixUp:
    """Applies the MixUp augmentation to a pair of images and their corresponding boxes."""

    def __init__(self, prob=0.5, alpha=1.0):
        self.alpha = alpha
        self.prob = prob
        self.parent = None

    def set_parent(self, parent):
        """Set the parent dataset object for accessing dataset methods."""
        self.parent = parent

    def __call__(self, image, boxes):
        if torch.rand(1) >= self.prob:
            return image, boxes

        assert self.parent is not None, "Parent is not set. MixUp cannot retrieve additional data."

        # Retrieve another image and its boxes randomly from the dataset
        image2, boxes2 = self.parent.get_more_data()[0]

        # Calculate the mixup lambda parameter
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 0.5

        # Mix images
        image1, image2 = TF.to_tensor(image), TF.to_tensor(image2)
        mixed_image = lam * image1 + (1 - lam) * image2

        # Merge bounding boxes
        merged_boxes = torch.cat((boxes, boxes2))

        return TF.to_pil_image(mixed_image), merged_boxes


class CopyPaste:
    """Naive Copy-Paste augmentation that pastes random objects from other images."""

    def __init__(
        self,
        prob: float = 0.3,
        *,
        sample_num: int = 1,
        max_paste_objects: int = 5,
        scale_jitter: Optional[Sequence[float]] = (0.8, 1.2),
    ) -> None:
        self.prob = prob
        self.sample_num = max(1, int(sample_num))
        self.max_paste_objects = max(1, int(max_paste_objects))
        self.parent = None
        if scale_jitter is not None:
            if len(scale_jitter) != 2:
                raise ValueError("scale_jitter must have exactly two values (min, max)")
            self.scale_jitter = (float(scale_jitter[0]), float(scale_jitter[1]))
        else:
            self.scale_jitter = None

    def set_parent(self, parent):
        self.parent = parent

    def __call__(self, image: Image.Image, boxes: torch.Tensor):
        if torch.rand(1) >= self.prob:
            return image, boxes
        if self.parent is None:
            return image, boxes

        base_w, base_h = image.size
        if base_w <= 1 or base_h <= 1:
            return image, boxes

        dtype = boxes.dtype if boxes.numel() else torch.float32
        pasted_boxes = []

        helpers = self.parent.get_more_data(self.sample_num)
        for helper_image, helper_boxes in helpers:
            if helper_boxes.numel() == 0:
                continue

            helper_w, helper_h = helper_image.size
            if helper_w <= 1 or helper_h <= 1:
                continue

            helper_abs = helper_boxes.clone().to(dtype)
            helper_abs[:, [1, 3]] *= helper_w
            helper_abs[:, [2, 4]] *= helper_h

            num_objects = helper_abs.shape[0]
            if num_objects == 0:
                continue

            pick = torch.randperm(num_objects)[: self.max_paste_objects]
            for idx in pick:
                cls, x1, y1, x2, y2 = helper_abs[idx].tolist()
                x1_i, y1_i, x2_i, y2_i = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
                if x2_i - x1_i < 2 or y2_i - y1_i < 2:
                    continue

                crop = helper_image.crop((x1_i, y1_i, x2_i, y2_i))
                crop_w, crop_h = crop.size
                if crop_w <= 1 or crop_h <= 1:
                    continue

                if self.scale_jitter is not None:
                    factor = float(torch.empty(1).uniform_(self.scale_jitter[0], self.scale_jitter[1]).item())
                    factor = max(0.1, factor)
                    new_w = max(1, int(round(crop_w * factor)))
                    new_h = max(1, int(round(crop_h * factor)))
                    crop = crop.resize((new_w, new_h), Image.Resampling.BILINEAR)
                    crop_w, crop_h = crop.size

                if crop_w >= base_w or crop_h >= base_h:
                    continue

                max_x = base_w - crop_w
                max_y = base_h - crop_h
                if max_x <= 0 or max_y <= 0:
                    continue

                offset_x = int(torch.randint(0, max_x + 1, (1,)).item())
                offset_y = int(torch.randint(0, max_y + 1, (1,)).item())

                image.paste(crop, (offset_x, offset_y))

                x1_new = offset_x / base_w
                y1_new = offset_y / base_h
                x2_new = (offset_x + crop_w) / base_w
                y2_new = (offset_y + crop_h) / base_h

                new_box = torch.tensor([cls, x1_new, y1_new, x2_new, y2_new], dtype=dtype)
                new_box[1:] = new_box[1:].clamp(0.0, 1.0)
                pasted_boxes.append(new_box)

        if pasted_boxes:
            pasted_tensor = torch.stack(pasted_boxes)
            if boxes.numel():
                boxes = torch.cat([boxes, pasted_tensor.to(boxes.dtype)], dim=0)
            else:
                boxes = pasted_tensor

        return image, boxes


class RandomCrop:
    """Randomly crops the image to half its size along with adjusting the bounding boxes."""

    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): Probability of applying the crop.
        """
        self.prob = prob

    def __call__(self, image, boxes):
        if torch.rand(1) < self.prob:
            original_width, original_height = image.size
            crop_height, crop_width = original_height // 2, original_width // 2
            top = torch.randint(0, original_height - crop_height + 1, (1,)).item()
            left = torch.randint(0, original_width - crop_width + 1, (1,)).item()

            image = TF.crop(image, top, left, crop_height, crop_width)

            boxes[:, [1, 3]] = boxes[:, [1, 3]] * original_width - left
            boxes[:, [2, 4]] = boxes[:, [2, 4]] * original_height - top

            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, crop_width)
            boxes[:, [2, 4]] = boxes[:, [2, 4]].clamp(0, crop_height)

            boxes[:, [1, 3]] /= crop_width
            boxes[:, [2, 4]] /= crop_height

        return image, boxes


class Translation:
    """Translate image and boxes by random offsets with optional border padding."""

    def __init__(
        self,
        prob: float = 0.5,
        translate: float = 0.1,
        *,
        border: Sequence[int] = (0, 0),
        fill: Optional[Sequence[int]] = (114, 114, 114),
    ) -> None:
        self.prob = prob
        self.translate = max(0.0, float(translate))
        if len(border) != 2:
            raise ValueError("border must provide vertical and horizontal padding")
        self.border = (int(border[0]), int(border[1]))
        if fill is None:
            self.fill = None
        else:
            if len(fill) not in (1, 3):
                raise ValueError("fill must be length 1 or 3 when provided")
            self.fill = tuple(int(v) for v in fill)

    def __call__(self, image: Image.Image, boxes: torch.Tensor):
        if torch.rand(1) >= self.prob or self.translate <= 0.0:
            return image, boxes

        try:
            import cv2
        except Exception as e:
            raise RuntimeError("OpenCV is required for Translation. Install `opencv-python`.") from e

        np_image = np.array(image)
        if np_image.ndim != 3:
            return image, boxes

        h0, w0, _ = np_image.shape
        if h0 <= 1 or w0 <= 1:
            return image, boxes

        border_h, border_w = self.border
        height = h0 + border_h * 2
        width = w0 + border_w * 2

        tx = torch.empty(1).uniform_(0.5 - self.translate, 0.5 + self.translate).item() * width
        ty = torch.empty(1).uniform_(0.5 - self.translate, 0.5 + self.translate).item() * height

        C = np.eye(3, dtype=np.float32)
        C[0, 2] = -w0 / 2.0
        C[1, 2] = -h0 / 2.0

        T = np.eye(3, dtype=np.float32)
        T[0, 2] = tx
        T[1, 2] = ty

        M = T @ C
        affine = M[:2]

        border_value = self.fill if self.fill is not None else 0
        translated = cv2.warpAffine(
            np_image,
            affine,
            dsize=(width, height),
            flags=cv2.INTER_LINEAR,
            borderValue=border_value,
        )

        image = Image.fromarray(translated)

        if boxes.numel():
            dtype = boxes.dtype
            boxes_abs = boxes.clone().to(torch.float32)
            boxes_abs[:, [1, 3]] *= w0
            boxes_abs[:, [2, 4]] *= h0

            shift_x = tx - (w0 / 2.0)
            shift_y = ty - (h0 / 2.0)
            boxes_abs[:, [1, 3]] += shift_x
            boxes_abs[:, [2, 4]] += shift_y

            boxes_abs[:, [1, 3]] = boxes_abs[:, [1, 3]].clamp(0.0, float(width))
            boxes_abs[:, [2, 4]] = boxes_abs[:, [2, 4]].clamp(0.0, float(height))

            widths = boxes_abs[:, 3] - boxes_abs[:, 1]
            heights = boxes_abs[:, 4] - boxes_abs[:, 2]
            keep = (widths > 1e-6) & (heights > 1e-6)
            boxes_abs = boxes_abs[keep]

            boxes_abs[:, [1, 3]] /= float(width)
            boxes_abs[:, [2, 4]] /= float(height)

            boxes = boxes_abs.to(dtype)

        return image, boxes


class RandomScale:
    """Randomly scales the image while adjusting bounding boxes."""

    def __init__(
        self,
        prob: float = 0.5,
        scale_range=(0.8, 1.2),
        keep_aspect_ratio: bool = True,
        resample: int = Image.Resampling.BILINEAR,
    ) -> None:
        self.prob = prob
        if isinstance(scale_range, (int, float)):
            scale_range = (float(scale_range), float(scale_range))
        if len(scale_range) != 2:
            raise ValueError("scale_range must contain exactly two values (min, max).")
        self.scale_min = float(scale_range[0])
        self.scale_max = float(scale_range[1])
        if self.scale_min <= 0 or self.scale_max <= 0:
            raise ValueError("scale_range values must be positive.")
        self.keep_aspect_ratio = keep_aspect_ratio
        self.resample = resample

    def __call__(self, image, boxes):
        if torch.rand(1) >= self.prob:
            return image, boxes

        width, height = image.size
        if width <= 0 or height <= 0:
            return image, boxes

        scale_x = torch.empty(1).uniform_(self.scale_min, self.scale_max).item()
        if self.keep_aspect_ratio:
            scale_y = scale_x
        else:
            scale_y = torch.empty(1).uniform_(self.scale_min, self.scale_max).item()

        new_width = max(1, int(round(width * scale_x)))
        new_height = max(1, int(round(height * scale_y)))

        if new_width == width and new_height == height:
            return image, boxes

        image = image.resize((new_width, new_height), self.resample)

        if boxes.numel() == 0:
            return image, boxes

        boxes = boxes.clone()
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * width
        boxes[:, [2, 4]] = boxes[:, [2, 4]] * height

        ratio_x = new_width / width
        ratio_y = new_height / height
        boxes[:, [1, 3]] *= ratio_x
        boxes[:, [2, 4]] *= ratio_y

        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, new_width)
        boxes[:, [2, 4]] = boxes[:, [2, 4]].clamp(0, new_height)

        boxes[:, [1, 3]] /= new_width
        boxes[:, [2, 4]] /= new_height

        return image, boxes


class RandomBrightness:
    """Randomly adjust image brightness within a factor range."""

    def __init__(self, prob: float = 0.5, factor_range=(0.7, 1.3)):
        self.prob = prob
        self.factor_range = factor_range

    def __call__(self, image, boxes):
        if torch.rand(1) < self.prob:
            low, high = self.factor_range
            factor = torch.empty(1).uniform_(float(low), float(high)).item()
            image = TF.adjust_brightness(image, factor)
        return image, boxes


class RandomContrast:
    """Randomly adjust image contrast within a factor range."""

    def __init__(self, prob: float = 0.5, factor_range=(0.7, 1.3)):
        self.prob = prob
        self.factor_range = factor_range

    def __call__(self, image, boxes):
        if torch.rand(1) < self.prob:
            low, high = self.factor_range
            factor = torch.empty(1).uniform_(float(low), float(high)).item()
            image = TF.adjust_contrast(image, factor)
        return image, boxes


class RandomSaturation:
    """Randomly adjust image saturation within a factor range."""

    def __init__(self, prob: float = 0.5, factor_range=(0.7, 1.3)):
        self.prob = prob
        self.factor_range = factor_range

    def __call__(self, image, boxes):
        if torch.rand(1) < self.prob:
            low, high = self.factor_range
            factor = torch.empty(1).uniform_(float(low), float(high)).item()
            image = TF.adjust_saturation(image, factor)
        return image, boxes


class RandomHSV:
    """
    Randomly perturb hue, saturation, and value in HSV space.

    This implementation avoids look-up tables and OpenCV usage. It operates via PIL's
    HSV conversion and applies lightweight, vectorized adjustments:
      - Hue: integer cyclic shift in [-hue_gain, +hue_gain] of the 8-bit hue domain (0-255).
      - Saturation/Value: push-pull adjustment around mid-level (128) using a signed
        delta in [-saturation_gain, +saturation_gain] / [-value_gain, +value_gain].
        This preserves extremes better than simple scaling while remaining fast and
        differentiable w.r.t. inputs.

    Args:
        prob (float): Probability to apply the augmentation.
        hue_gain (float): Max fractional hue shift w.r.t. full cycle (0-1 corresponds to 0-255 steps).
        saturation_gain (float): Max signed strength for saturation push-pull around 128.
        value_gain (float): Max signed strength for value push-pull around 128.
    """

    def __init__(self, prob: float = 0.5, *, hue_gain: float = 0.015, saturation_gain: float = 0.7, value_gain: float = 0.4):
        self.prob = prob
        self.hue_gain = float(hue_gain)
        self.saturation_gain = float(saturation_gain)
        self.value_gain = float(value_gain)

    def __call__(self, image: Image.Image, boxes):
        if torch.rand(1) >= self.prob:
            return image, boxes

        # Convert to HSV (8-bit per channel: 0..255)
        hsv = np.array(image.convert("HSV"), dtype=np.uint16)  # use wider type to avoid overflow mid-compute

        h = hsv[..., 0].astype(np.int16)  # allow negative before modulo
        s = hsv[..., 1].astype(np.float32)
        v = hsv[..., 2].astype(np.float32)

        # Hue: sample fractional shift of the 0..255 cycle, then wrap with modulo 256.
        if self.hue_gain > 0:
            dh = int(round(torch.empty(1).uniform_(-self.hue_gain, self.hue_gain).item() * 255.0))
            if dh != 0:
                h = (h + dh) % 256

        # Saturation/Value: push–pull around mid (128). Signed deltas in [-gain, +gain].
        # s' = s + (s - 128) * ds; v' = v + (v - 128) * dv
        if self.saturation_gain > 0:
            ds = float(torch.empty(1).uniform_(-self.saturation_gain, self.saturation_gain).item())
            if ds != 0.0:
                s = s + (s - 128.0) * ds
        if self.value_gain > 0:
            dv = float(torch.empty(1).uniform_(-self.value_gain, self.value_gain).item())
            if dv != 0.0:
                v = v + (v - 128.0) * dv

        # Clip back to valid range and pack
        h = np.clip(h, 0, 255).astype(np.uint8)
        s = np.clip(s, 0.0, 255.0).astype(np.uint8)
        v = np.clip(v, 0.0, 255.0).astype(np.uint8)
        hsv_out = np.stack([h, s, v], axis=-1)

        # Back to RGB PIL Image
        image = Image.fromarray(hsv_out, mode="HSV").convert("RGB")
        return image, boxes


# ===============================
# Albumentations-based augmenters
# ===============================

def _albu_apply(image: Image.Image, aug) -> Image.Image:
    """Apply an Albumentations augmenter to a PIL image safely."""
    arr = np.array(image)
    out = aug(image=arr)
    return Image.fromarray(out["image"])  # type: ignore[index]


class Blur:
    """Albumentations Blur wrapper (image-only)."""

    def __init__(self, prob: float = 0.1, blur_limit=(3, 7)):
        self.prob = prob
        self.blur_limit = blur_limit

    def __call__(self, image, boxes):
        if torch.rand(1) >= self.prob:
            return image, boxes
        try:
            import albumentations as A

            Cls = A.Blur
            params = inspect.signature(Cls.__init__).parameters
            kwargs = {"p": 1.0}
            if "blur_limit" in params:
                kwargs["blur_limit"] = tuple(self.blur_limit)
            aug = Cls(**kwargs)
            image = _albu_apply(image, aug)
            return image, boxes
        except Exception as e:
            raise RuntimeError("Albumentations is required for Blur. Install `albumentations`." ) from e


class MotionBlur:
    """Albumentations MotionBlur wrapper (image-only)."""

    def __init__(self, prob: float = 0.1, blur_limit=(5, 15)):
        self.prob = prob
        self.blur_limit = blur_limit

    def __call__(self, image, boxes):
        if torch.rand(1) >= self.prob:
            return image, boxes
        try:
            import albumentations as A

            Cls = A.MotionBlur
            params = inspect.signature(Cls.__init__).parameters
            kwargs = {"p": 1.0}
            if "blur_limit" in params:
                kwargs["blur_limit"] = tuple(self.blur_limit)
            aug = Cls(**kwargs)
            image = _albu_apply(image, aug)
            return image, boxes
        except Exception as e:
            raise RuntimeError("Albumentations is required for MotionBlur. Install `albumentations`." ) from e


class GaussianBlur:
    """Albumentations GaussianBlur wrapper (image-only)."""

    def __init__(self, prob: float = 0.1, blur_limit=(3, 7), sigma_limit=(0.1, 2.0)):
        self.prob = prob
        self.blur_limit = blur_limit
        self.sigma_limit = sigma_limit

    def __call__(self, image, boxes):
        if torch.rand(1) >= self.prob:
            return image, boxes
        try:
            import albumentations as A

            Cls = A.GaussianBlur
            params = inspect.signature(Cls.__init__).parameters
            kwargs = {"p": 1.0}
            if "blur_limit" in params:
                kwargs["blur_limit"] = tuple(self.blur_limit)
            if "sigma_limit" in params:
                kwargs["sigma_limit"] = tuple(self.sigma_limit)
            aug = Cls(**kwargs)
            image = _albu_apply(image, aug)
            return image, boxes
        except Exception as e:
            raise RuntimeError("Albumentations is required for GaussianBlur. Install `albumentations`." ) from e


class GaussNoise:
    """Albumentations GaussNoise wrapper (image-only)."""

    def __init__(self, prob: float = 0.15, mean: float = 0.0, var_limit=(10.0, 50.0)):
        self.prob = prob
        self.mean = mean
        self.var_limit = var_limit

    def __call__(self, image, boxes):
        if torch.rand(1) >= self.prob:
            return image, boxes
        try:
            import albumentations as A

            Cls = getattr(A, "GaussNoise", None)
            if Cls is None:
                Cls = getattr(A, "GaussianNoise", None)
            if Cls is None:
                raise RuntimeError("Albumentations GaussNoise/GaussianNoise not available")
            params = inspect.signature(Cls.__init__).parameters
            kwargs = {"p": 1.0}
            var = tuple(self.var_limit)
            if "var_limit" in params:
                kwargs["var_limit"] = var
            elif "std_range" in params:
                # Albumentations 2.x expects std_range in [0,1] for float images
                s0 = float(np.sqrt(var[0])) / 255.0
                s1 = float(np.sqrt(var[1])) / 255.0
                std = (max(0.0, min(1.0, s0)), max(0.0, min(1.0, s1)))
                kwargs["std_range"] = std
            if "mean" in params:
                kwargs["mean"] = float(self.mean)
            aug = Cls(**kwargs)
            image = _albu_apply(image, aug)
            return image, boxes
        except Exception as e:
            raise RuntimeError("Albumentations is required for GaussNoise. Install `albumentations`." ) from e


class ImageCompression:
    """Albumentations ImageCompression wrapper (image-only)."""

    def __init__(self, prob: float = 0.25, quality_range=(40, 90)):
        self.prob = prob
        self.quality_range = quality_range

    def __call__(self, image, boxes):
        if torch.rand(1) >= self.prob:
            return image, boxes
        try:
            import albumentations as A

            ql, qu = int(self.quality_range[0]), int(self.quality_range[1])
            Cls = A.ImageCompression
            params = inspect.signature(Cls.__init__).parameters
            kwargs = {"p": 1.0}
            if "quality_range" in params:
                kwargs["quality_range"] = (ql, qu)
            else:
                if "quality_lower" in params:
                    kwargs["quality_lower"] = ql
                if "quality_upper" in params:
                    kwargs["quality_upper"] = qu
            aug = Cls(**kwargs)
            image = _albu_apply(image, aug)
            return image, boxes
        except Exception as e:
            raise RuntimeError("Albumentations is required for ImageCompression. Install `albumentations`." ) from e


class ISONoise:
    """Albumentations ISONoise wrapper (image-only)."""

    def __init__(self, prob: float = 0.2, intensity=(0.05, 0.15), color_shift=(0.01, 0.05)):
        self.prob = prob
        self.intensity = intensity
        self.color_shift = color_shift

    def __call__(self, image, boxes):
        if torch.rand(1) >= self.prob:
            return image, boxes
        try:
            import albumentations as A

            Cls = A.ISONoise
            params = inspect.signature(Cls.__init__).parameters
            kwargs = {"p": 1.0}
            if "color_shift" in params:
                kwargs["color_shift"] = tuple(self.color_shift)
            if "intensity" in params:
                kwargs["intensity"] = tuple(self.intensity)
            elif "intensity_range" in params:
                kwargs["intensity_range"] = tuple(self.intensity)
            aug = Cls(**kwargs)
            image = _albu_apply(image, aug)
            return image, boxes
        except Exception as e:
            raise RuntimeError("Albumentations is required for ISONoise. Install `albumentations`." ) from e


class RandomRain:
    """Albumentations RandomRain wrapper (image-only)."""

    def __init__(
        self,
        prob: float = 0.15,
        slant_range=(-10, 10),
        drop_length=(15, 30),
        drop_width_range=(1, 2),
        density=(0.002, 0.006),  # kept for config compatibility; Albumentations doesn't need it
        blur_value=(3, 5),
        brightness_coefficient=(0.9, 1.0),
    ):
        self.prob = prob
        self.slant_range = slant_range
        self.drop_length = drop_length
        self.drop_width_range = drop_width_range
        self.density = density
        self.blur_value = blur_value
        self.brightness_coefficient = brightness_coefficient

    def __call__(self, image, boxes):
        if torch.rand(1) >= self.prob:
            return image, boxes
        try:
            import albumentations as A

            Cls = A.RandomRain
            params = inspect.signature(Cls.__init__).parameters
            sl_l, sl_u = int(self.slant_range[0]), int(self.slant_range[1])
            dl = int(torch.randint(int(self.drop_length[0]), int(self.drop_length[1]) + 1, (1,)).item())
            dw = int(torch.randint(int(self.drop_width_range[0]), int(self.drop_width_range[1]) + 1, (1,)).item())
            bv = int(torch.randint(int(self.blur_value[0]), int(self.blur_value[1]) + 1, (1,)).item())
            bc = float(torch.empty(1).uniform_(float(self.brightness_coefficient[0]), float(self.brightness_coefficient[1])).item())
            kwargs = {"p": 1.0}
            if "slant_range" in params:
                kwargs["slant_range"] = (sl_l, sl_u)
            else:
                if "slant_lower" in params:
                    kwargs["slant_lower"] = sl_l
                if "slant_upper" in params:
                    kwargs["slant_upper"] = sl_u
            if "drop_length" in params:
                kwargs["drop_length"] = dl
            if "drop_width" in params:
                kwargs["drop_width"] = dw
            if "blur_value" in params:
                kwargs["blur_value"] = bv
            if "brightness_coefficient" in params:
                kwargs["brightness_coefficient"] = bc
            aug = Cls(**kwargs)
            image = _albu_apply(image, aug)
            return image, boxes
        except Exception as e:
            raise RuntimeError("Albumentations is required for RandomRain. Install `albumentations`." ) from e


class RandomFog:
    """Albumentations RandomFog wrapper (image-only)."""

    def __init__(self, prob: float = 0.1, fog_coef=(0.3, 0.6), alpha_coef=(0.05, 0.1)):
        self.prob = prob
        self.fog_coef = fog_coef
        self.alpha_coef = alpha_coef

    def __call__(self, image, boxes):
        if torch.rand(1) >= self.prob:
            return image, boxes
        try:
            import albumentations as A

            Cls = A.RandomFog
            params = inspect.signature(Cls.__init__).parameters
            fog_lower, fog_upper = float(self.fog_coef[0]), float(self.fog_coef[1])
            alpha = float(torch.empty(1).uniform_(float(self.alpha_coef[0]), float(self.alpha_coef[1])).item())
            kwargs = {"p": 1.0}
            if "fog_coef" in params:
                kwargs["fog_coef"] = (fog_lower, fog_upper)
            elif "fog_coef_range" in params:
                kwargs["fog_coef_range"] = (fog_lower, fog_upper)
            else:
                if "fog_coef_lower" in params:
                    kwargs["fog_coef_lower"] = fog_lower
                if "fog_coef_upper" in params:
                    kwargs["fog_coef_upper"] = fog_upper
            if "alpha_coef" in params:
                kwargs["alpha_coef"] = alpha
            aug = Cls(**kwargs)
            image = _albu_apply(image, aug)
            return image, boxes
        except Exception as e:
            raise RuntimeError("Albumentations is required for RandomFog. Install `albumentations`." ) from e


class RandomSunFlare:
    """Albumentations RandomSunFlare wrapper (image-only)."""

    def __init__(self, prob: float = 0.1, src_radius_range=(50, 150), src_intensity=(0.6, 1.0)):
        self.prob = prob
        self.src_radius_range = src_radius_range
        self.src_intensity = src_intensity

    def __call__(self, image, boxes):
        if torch.rand(1) >= self.prob:
            return image, boxes
        try:
            import albumentations as A

            src_radius = int(torch.randint(int(self.src_radius_range[0]), int(self.src_radius_range[1]) + 1, (1,)).item())
            intensity = float(torch.empty(1).uniform_(float(self.src_intensity[0]), float(self.src_intensity[1])).item())
            Cls = A.RandomSunFlare
            params = inspect.signature(Cls.__init__).parameters
            kwargs = {"src_radius": src_radius, "p": 1.0, "flare_roi": (0, 0, 1, 1)}
            if "intensity" in params:
                kwargs["intensity"] = intensity
            elif "intensity_coeff" in params:
                kwargs["intensity_coeff"] = intensity
            aug = Cls(**kwargs)
            image = _albu_apply(image, aug)
            return image, boxes
        except Exception as e:
            raise RuntimeError("Albumentations is required for RandomSunFlare. Install `albumentations`." ) from e


class RandomResizedCrop:
    """Albumentations RandomResizedCrop wrapper that keeps bounding boxes consistent."""

    def __init__(
        self,
        prob: float = 0.5,
        *,
        height: int = 640,
        width: int = 640,
        scale: Sequence[float] = (0.8, 1.0),
        ratio: Sequence[float] = (0.75, 1.33),
        min_visibility: float = 0.0,
        min_area: float = 0.0,
        interpolation: Optional[object] = None,
    ) -> None:
        self.prob = prob
        self.height = int(height)
        self.width = int(width)
        if len(scale) != 2:
            raise ValueError("scale must contain exactly two values (min, max).")
        if len(ratio) != 2:
            raise ValueError("ratio must contain exactly two values (min, max).")
        self.scale = (float(scale[0]), float(scale[1]))
        self.ratio = (float(ratio[0]), float(ratio[1]))
        self.min_visibility = float(min_visibility)
        self.min_area = float(min_area)
        self.interpolation = interpolation
        self._transform = None

    def _resolve_interpolation(self, cv2_module):
        interp = self.interpolation
        if interp is None:
            return cv2_module.INTER_LINEAR
        if isinstance(interp, str):
            attr = f"INTER_{interp.upper()}"
            if hasattr(cv2_module, attr):
                return getattr(cv2_module, attr)
            raise ValueError(f"Unsupported interpolation string: {interp}")
        if isinstance(interp, int):
            return int(interp)
        resampling = getattr(Image, "Resampling", None)
        if resampling is not None and isinstance(interp, resampling):
            mapping = {
                resampling.NEAREST: cv2_module.INTER_NEAREST,
                resampling.BILINEAR: cv2_module.INTER_LINEAR,
                resampling.BICUBIC: cv2_module.INTER_CUBIC,
                resampling.BOX: cv2_module.INTER_AREA,
                resampling.HAMMING: cv2_module.INTER_LINEAR,
                resampling.LANCZOS: cv2_module.INTER_LANCZOS4,
            }
            return mapping.get(interp, cv2_module.INTER_LINEAR)
        # Fallback
        return cv2_module.INTER_LINEAR

    def _get_transform(self):
        if self._transform is None:
            try:
                import albumentations as A
                import cv2
            except Exception as e:
                raise RuntimeError("Albumentations is required for RandomResizedCrop. Install `albumentations`.") from e

            interpolation = self._resolve_interpolation(cv2)
            self._transform = A.Compose(
                [
                    A.RandomResizedCrop(
                        height=self.height,
                        width=self.width,
                        scale=self.scale,
                        ratio=self.ratio,
                        interpolation=interpolation,
                    )
                ],
                bbox_params=A.BboxParams(
                    format="pascal_voc",
                    label_fields=["labels"],
                    min_visibility=self.min_visibility,
                    min_area=self.min_area,
                    clip=True,
                ),
            )
        return self._transform

    def __call__(self, image: Image.Image, boxes: torch.Tensor):
        if torch.rand(1) >= self.prob:
            return image, boxes

        transform = self._get_transform()
        image_np = np.array(image)
        orig_w, orig_h = image.size
        dtype = boxes.dtype if boxes.numel() else torch.float32

        if boxes.numel():
            abs_boxes = boxes.clone()
            abs_boxes[:, [1, 3]] *= orig_w
            abs_boxes[:, [2, 4]] *= orig_h
            bbox_list = abs_boxes[:, 1:5].tolist()
            labels = boxes[:, 0].tolist()
        else:
            bbox_list = []
            labels = []

        augmented = transform(image=image_np, bboxes=bbox_list, labels=labels)
        aug_image = Image.fromarray(augmented["image"])
        aug_w, aug_h = aug_image.size
        aug_boxes = augmented.get("bboxes", [])
        aug_labels = augmented.get("labels", labels)

        if aug_boxes:
            boxes_tensor = torch.tensor(aug_boxes, dtype=dtype)
            labels_tensor = torch.tensor(aug_labels, dtype=dtype)
            result = torch.zeros((boxes_tensor.shape[0], 5), dtype=dtype)
            result[:, 0] = labels_tensor
            result[:, 1] = boxes_tensor[:, 0] / max(aug_w, 1)
            result[:, 2] = boxes_tensor[:, 1] / max(aug_h, 1)
            result[:, 3] = boxes_tensor[:, 2] / max(aug_w, 1)
            result[:, 4] = boxes_tensor[:, 3] / max(aug_h, 1)
            result[:, 1:] = result[:, 1:].clamp(0.0, 1.0)
        else:
            result = torch.zeros((0, 5), dtype=dtype)

        return aug_image, result


class MedianBlur:
    """Albumentations MedianBlur wrapper (image-only)."""

    def __init__(self, prob: float = 0.1, blur_limit: Sequence[int] = (3, 7)):
        self.prob = prob
        self.blur_limit = self._normalize_blur_limit(blur_limit)

    def _normalize_blur_limit(self, blur_limit: Sequence[int]) -> Union[Tuple[int, int], int]:
        if isinstance(blur_limit, Sequence) and not isinstance(blur_limit, (str, bytes)):
            values = [int(v) for v in list(blur_limit)]
            if not values:
                raise ValueError("blur_limit sequence must not be empty")
            if len(values) == 1:
                return values[0]
            return (values[0], values[1])
        return int(blur_limit)

    def _prepare_blur_limit(self) -> Union[Tuple[int, int], int]:
        def _ensure_odd(value: int) -> int:
            value = max(3, value)
            return value if value % 2 else value + 1

        if isinstance(self.blur_limit, tuple):
            low = _ensure_odd(int(self.blur_limit[0]))
            high = _ensure_odd(int(self.blur_limit[1]))
            if high < low:
                high = low
            return (low, high)
        return _ensure_odd(int(self.blur_limit))

    def __call__(self, image, boxes):
        if torch.rand(1) >= self.prob:
            return image, boxes
        try:
            import albumentations as A

            Cls = A.MedianBlur
            kwargs = {"p": 1.0}
            blur_limit = self._prepare_blur_limit()
            if "blur_limit" in inspect.signature(Cls.__init__).parameters:
                kwargs["blur_limit"] = blur_limit
            aug = Cls(**kwargs)
            image = _albu_apply(image, aug)
            return image, boxes
        except Exception as e:
            raise RuntimeError("Albumentations is required for MedianBlur. Install `albumentations`." ) from e


class ToGray:
    """Albumentations ToGray wrapper (image-only)."""

    def __init__(self, prob: float = 0.05):
        self.prob = prob

    def __call__(self, image, boxes):
        if torch.rand(1) >= self.prob:
            return image, boxes
        try:
            import albumentations as A

            aug = A.ToGray(p=1.0)
            image = _albu_apply(image, aug)
            return image, boxes
        except Exception as e:
            raise RuntimeError("Albumentations is required for ToGray. Install `albumentations`." ) from e


class CLAHE:
    """Albumentations CLAHE wrapper (image-only)."""

    def __init__(
        self,
        prob: float = 0.05,
        *,
        clip_limit: float = 4.0,
        tile_grid_size: Sequence[int] = (8, 8),
    ) -> None:
        self.prob = prob
        self.clip_limit = float(clip_limit)
        if len(tile_grid_size) != 2:
            raise ValueError("tile_grid_size must be a sequence of length 2")
        self.tile_grid_size = (int(tile_grid_size[0]), int(tile_grid_size[1]))

    def __call__(self, image, boxes):
        if torch.rand(1) >= self.prob:
            return image, boxes
        try:
            import albumentations as A

            kwargs = {"p": 1.0, "clip_limit": self.clip_limit, "tile_grid_size": self.tile_grid_size}
            aug = A.CLAHE(**kwargs)
            image = _albu_apply(image, aug)
            return image, boxes
        except Exception as e:
            raise RuntimeError("Albumentations is required for CLAHE. Install `albumentations`." ) from e
