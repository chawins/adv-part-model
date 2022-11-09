"""Transforms that apply to both image and segmentation masks.

This code is copied from
https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py
"""

import random
from typing import Any, Callable

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL.Image import Image


_TransformOut = tuple[Image, Image] | tuple[Image, Image, dict[str, Any]]


def _pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


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

    def __call__(self, image, target):
        image = F.resize(image, self.size, antialias=True)
        target = F.resize(
            target,
            self.size,
            interpolation=T.InterpolationMode.NEAREST,
            antialias=True,
        )
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size, antialias=True)
        target = F.resize(
            target, size, interpolation=T.InterpolationMode.NEAREST
        )
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob: float, return_params: bool = False) -> None:
        self.flip_prob: float = flip_prob
        self._return_params: bool = return_params

    def __call__(
        self,
        image: Image,
        target: Image,
        params: list[Any] | None = None,
    ) -> _TransformOut:
        is_flip = random.random() < self.flip_prob
        if is_flip:
            image = F.hflip(image)
            target = F.hflip(target)
        if not self._return_params:
            return image, target

        if params is None:
            params = [is_flip]
        else:
            params.append(is_flip)
        return image, target, params


class RandomCrop(object):
    def __init__(self, size: int):
        self.size: int = size

    def __call__(self, image, target):
        image = _pad_if_smaller(image, self.size)
        target = _pad_if_smaller(target, self.size, fill=0)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


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
        target: Image,
        params: list[Any] | None = None,
    ) -> _TransformOut:
        crop_params = T.RandomResizedCrop.get_params(
            image, self.scale, self.ratio
        )
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        image = F.resize(image, (self.size, self.size), antialias=True)
        target = F.resize(
            target,
            (self.size, self.size),
            interpolation=T.InterpolationMode.NEAREST,
        )
        if self.return_params:
            if params is None:
                params = [crop_params]
            else:
                params.append(crop_params)
            return image, target, params
        return image, target


class CenterCrop(object):
    def __init__(self, size: int):
        self.size = [size, size]

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    """Custom ToTensor for image and segmentation mask."""

    def __call__(self, image: Image, target: Image, *args) -> _TransformOut:
        """Convert image and target to torch.Tensor."""
        image = F.pil_to_tensor(image)
        image = F.convert_image_dtype(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target, *args
