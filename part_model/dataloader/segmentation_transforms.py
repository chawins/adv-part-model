"""Transforms that apply to both image and segmentation masks.

This code is copied from
https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL.Image import Image


_TransformOut = Union[Tuple[Image, Image, Dict], Tuple[Image, Image, Dict, Dict[str, Any]]]


def _pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img

# adapted from DINO.datasets.transforms
def crop_bbox(target, region):
    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return target

# adapted from DINO.datasets.transforms
def hflip_bbox(image, target):
    w, h = image.size
    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return target

# adapted from DINO.datasets.transforms
def resize_bbox(image, rescaled_image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    return target


class Compose(object):
    def __init__(self, transforms: list[Callable[..., _TransformOut]]):
        self.transforms: list[Callable[..., _TransformOut]] = transforms

    def __call__(self, *args) -> _TransformOut:
        for t in self.transforms:
            args = t(*args)
        return args


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask_target, bbox_target):
        resized_image = F.resize(image, self.size, antialias=True)
        mask_target = F.resize(
            mask_target,
            self.size,
            interpolation=T.InterpolationMode.NEAREST,
            antialias=True,
        )
        bbox_target = resize_bbox(image, resized_image, bbox_target, self.size)
        return resized_image, mask_target, bbox_target


# class RandomResize(object):
#     def __init__(self, min_size, max_size=None):
#         self.min_size = min_size
#         if max_size is None:
#             max_size = min_size
#         self.max_size = max_size

#     def __call__(self, image, target):
#         size = random.randint(self.min_size, self.max_size)
#         image = F.resize(image, size, antialias=True)
#         target = F.resize(
#             target, size, interpolation=T.InterpolationMode.NEAREST
#         )
#         return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob: float, return_params: bool = False) -> None:
        self.flip_prob: float = flip_prob
        self._return_params: bool = return_params

    def __call__(
        self,
        image: Image,
        mask_target: Image,
        bbox_target: dict,
        params: list[Any] | None = None,
    ) -> _TransformOut:
        is_flip = random.random() < self.flip_prob
        if is_flip:
            image = F.hflip(image)
            mask_target = F.hflip(mask_target)
            bbox_target = hflip_bbox(image, bbox_target)
        if not self._return_params:
            return image, mask_target, bbox_target

        if params is None:
            params = [is_flip]
        else:
            params.append(is_flip)
        return image, mask_target, bbox_target, params


# class RandomCrop(object):
#     def __init__(self, size: int):
#         self.size: int = size

#     def __call__(self, image, target):
#         image = _pad_if_smaller(image, self.size)
#         target = _pad_if_smaller(target, self.size, fill=0)
#         crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
#         image = F.crop(image, *crop_params)
#         target = F.crop(target, *crop_params)
#         return image, target

class RandomResizedCrop(object):
    def __init__(
        self,
        size: int,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (0.75, 1.333333),
        return_params: bool = False,
    ) -> None:
        self.size: int = size
        self.ratio: tuple[float, float] = ratio
        self.scale: tuple[float, float] = scale
        self.return_params: bool = return_params

    def __call__(
        self,
        image: Image,
        mask_target: Image,
        bbox_target: dict,
        params: list[Any] | None = None,
    ) -> _TransformOut:
        crop_params = T.RandomResizedCrop.get_params(
            image, self.scale, self.ratio
        )
        image = F.crop(image, *crop_params)
        mask_target = F.crop(mask_target, *crop_params)
        image = F.resize(image, (self.size, self.size), antialias=True)
        mask_target = F.resize(
            mask_target,
            (self.size, self.size),
            interpolation=T.InterpolationMode.NEAREST,
        )
        bbox_target = crop_bbox(bbox_target, crop_params)
        if self.return_params:
            if params is None:
                params = [crop_params]
            else:
                params.append(crop_params)
            return image, mask_target, bbox_target, params
        return image, mask_target, bbox_target


class CenterCrop(object):
    def __init__(self, size: int):
        self.size = [size, size]

    def __call__(self, image, mask_target, bbox_target):
        resized_image = F.center_crop(image, self.size)
        mask_target = F.center_crop(mask_target, self.size)
        
        image_width, image_height = image.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        bbox_target = crop_bbox(bbox_target, (crop_top, crop_left, crop_height, crop_width))
    
        return resized_image, mask_target, bbox_target


class ToTensor(object):
    """Custom ToTensor for image and segmentation mask."""

    def __call__(self, image: Image, mask_target: Image, bbox_target: dict, *args) -> _TransformOut:
        """Convert image and target to torch.Tensor."""
        image = F.pil_to_tensor(image)
        image = F.convert_image_dtype(image)
        mask_target = torch.as_tensor(np.array(mask_target), dtype=torch.int64)
        return image, mask_target, bbox_target, *args
