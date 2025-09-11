from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
import inspect


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
    def __init__(self, image_size, background_color=(114, 114, 114)):
        """Initialize the object with the target image size."""
        self.target_width, self.target_height = image_size
        self.background_color = background_color

    def set_size(self, image_size: List[int]):
        self.target_width, self.target_height = image_size

    def __call__(self, image: Image, boxes):
        img_width, img_height = image.size
        scale = min(self.target_width / img_width, self.target_height / img_height)
        new_width, new_height = int(img_width * scale), int(img_height * scale)

        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        pad_left = (self.target_width - new_width) // 2
        pad_top = (self.target_height - new_height) // 2
        padded_image = Image.new("RGB", (self.target_width, self.target_height), self.background_color)
        padded_image.paste(resized_image, (pad_left, pad_top))

        boxes[:, [1, 3]] = (boxes[:, [1, 3]] * new_width + pad_left) / self.target_width
        boxes[:, [2, 4]] = (boxes[:, [2, 4]] * new_height + pad_top) / self.target_height

        transform_info = torch.tensor([scale, pad_left, pad_top, pad_left, pad_top])
        return padded_image, boxes, transform_info


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
