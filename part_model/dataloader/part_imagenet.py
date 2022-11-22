"""PartImageNet dataset."""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch.utils import data
from PIL import Image
from PIL.Image import Image as _ImageType

from part_model.dataloader.segmentation_transforms import (
    CenterCrop,
    Compose,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from part_model.dataloader.util import COLORMAP
from part_model.utils import np_temp_seed
from part_model.utils.eval_sampler import DistributedEvalSampler
from part_model.utils.image import get_seg_type


CLASSES = {
    "Quadruped": 4,
    "Biped": 5,
    "Fish": 4,
    "Bird": 5,
    "Snake": 2,
    "Reptile": 4,
    "Car": 3,
    "Bicycle": 4,
    "Boat": 2,
    "Aeroplane": 5,
    "Bottle": 2,
}


class PartImageNetSegDataset(data.Dataset):
    """PartImageNet Dataset."""

    def __init__(
        self,
        root: str = "~/data/",
        seg_path: str = "~/data/",
        split: str = "train",
        transform: Callable[..., Any] = None,
        use_label: bool = False,
        seg_type: str | None = None,
        seg_fraction: float = 1.0,
        seed: int = 0,
        use_atta: bool = False,
    ) -> None:
        """Load our processed Part-ImageNet dataset.

        Args:
            root: Path to root directory
            split: Data split to load. Defaults to "train".
            transform: Transformations to apply to the images (and
                the segmentation masks if applicable). Defaults to None.
            use_label: Whether to yield class label. Defaults to False.
            seg_type: Specify types of segmentation to load
                ("part", "object", "fg", or None). Defaults to "part".
            seg_fraction: Fraction of segmentation mask to
                provide. The dropped masks are set to all -1. Defaults to 1.
            seed: Random seed. Defaults to 0.
            use_atta: If True, use ATTA (fast adversarial training) and return
                transform params during training.
        """
        self._root: str = root
        self._split: str = split
        self._seg_path: str = os.path.join(seg_path, split)
        self._transform = transform
        self._use_label: bool = use_label
        self._seg_type: str | None = seg_type
        self._use_atta: bool = use_atta

        self.classes: list[str] = self._list_classes()
        self.num_classes: int = len(self.classes)
        self.num_seg_labels: int = sum([CLASSES[c] for c in self.classes])

        # Load data from specified path
        self.images, self.labels, self.masks = self._get_data()
        # Randomly shuffle data
        idx = np.arange(len(self.images))
        with np_temp_seed(seed):
            np.random.shuffle(idx)
        # Randomly drop seg masks if specified for semi-supervised training
        self.seg_drop_idx = idx[: int((1 - seg_fraction) * len(self.images))]

        # Create matrix that maps part segmentation to object segmentation
        part_to_object = [0]
        self.part_to_class = [[0] * (self.num_classes + 1)]
        self.part_to_class[0][0] = 1
        for i, label in enumerate(self.classes):
            part_to_object.extend([i + 1] * CLASSES[label])
            base = [0] * (self.num_classes + 1)
            base[i + 1] = 1
            self.part_to_class.extend([base] * CLASSES[label])
        self.part_to_object = torch.tensor(part_to_object, dtype=torch.long)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        """Get sample at index.

        Args:
            index: Index of data sample to retrieve.

        Returns:
            Image, segmentation label (optional), class label (optional),
            transform params (optional).
        """
        # Collect variables to return
        return_items: list[Any] = []
        atta_data: list[torch.Tensor] | None = None
        _img: _ImageType = Image.open(self.images[index]).convert("RGB")
        _target: _ImageType = Image.open(self.masks[index])
        width, height = _img.size

        if self._transform is not None:
            tf_out = self._transform(_img, _target)
            if len(tf_out) == 3:
                # In ATTA, transform params are also returned
                _img, _target, params = tf_out
                atta_data = [torch.tensor(index), torch.tensor((height, width))]
                for p in params:
                    atta_data.append(torch.tensor(p))
            else:
                _img, _target = tf_out
        return_items.append(_img)

        if self._seg_type is not None:
            if self._seg_type == "object":
                _target = self.part_to_object[_target]
            elif self._seg_type == "fg":
                _target = (_target > 0).long()
            if index in self.seg_drop_idx:
                # Drop segmentation mask by setting all pixels to -1 to ignore
                # later at loss computation
                _target.mul_(0).add_(-1)
            return_items.append(_target)

        if self._use_label:
            _label = self.labels[index]
            return_items.append(_label)
        if atta_data is not None:
            for ad in atta_data:
                return_items.append(ad)
        return return_items

    def _get_data(self):
        images, labels, masks = [], [], []
        for label_idx, label in enumerate(self.classes):
            img_path = os.path.join(self._root, "JPEGImages")
            part_path = os.path.join(self._seg_path, label)
            # Read file names
            with open(f"{self._seg_path}/{label}.txt", "r") as fns:
                filenames = sorted([f.strip() for f in fns.readlines()])
            images.extend([f"{img_path}/{f}.JPEG" for f in filenames])
            masks.extend(
                [f'{part_path}/{f.split("/")[1]}.tif' for f in filenames]
            )
            labels.extend([label_idx] * len(filenames))
        labels = torch.tensor(labels, dtype=torch.long)
        return images, labels, masks

    def _list_classes(self):
        dirs = os.listdir(self._seg_path)
        dirs = [
            d for d in dirs if os.path.isdir(os.path.join(self._seg_path, d))
        ]
        return sorted(dirs)

    def __len__(self):
        return len(self.images)


def get_loader_sampler(args, transform, split: str):
    seg_type: str = get_seg_type(args)
    is_train: bool = split == "train"
    use_atta: bool = args.adv_train == "atta"

    part_imagenet_dataset = PartImageNetSegDataset(
        args.data,
        args.seg_label_dir,
        split=split,
        transform=transform,
        seg_type=seg_type,
        use_label=("semi" in args.experiment) or (seg_type is None),
        seg_fraction=args.semi_label if is_train else 1.0,
        use_atta=use_atta,
    )

    sampler: Optional[torch.utils.data.Sampler] = None
    shuffle: Optional[bool] = is_train
    if args.distributed:
        shuffle = None
        if is_train:
            sampler = torch.utils.data.distributed.DistributedSampler(
                part_imagenet_dataset,
                shuffle=True,
                seed=args.seed,
                drop_last=False,
            )
        else:
            # Use distributed sampler for validation but not testing
            sampler = DistributedEvalSampler(
                part_imagenet_dataset, shuffle=False, seed=args.seed
            )

    batch_size = args.batch_size
    loader = torch.utils.data.DataLoader(
        part_imagenet_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )

    # TODO: can we make this cleaner?
    PART_IMAGENET["part_to_class"] = part_imagenet_dataset.part_to_class
    PART_IMAGENET["num_classes"] = part_imagenet_dataset.num_classes
    PART_IMAGENET["num_seg_labels"] = part_imagenet_dataset.num_seg_labels

    pto = part_imagenet_dataset.part_to_object
    if seg_type == "part":
        seg_labels = len(pto)
    elif seg_type == "fg":
        seg_labels = 2
    else:
        seg_labels = pto.max().item() + 1

    setattr(args, "seg_labels", seg_labels)
    setattr(args, "num_classes", part_imagenet_dataset.num_classes)
    setattr(args, "input_dim", PART_IMAGENET["input_dim"])
    if is_train:
        setattr(args, "num_train", len(part_imagenet_dataset))

    return loader, sampler


def load_part_imagenet(args):

    img_size = PART_IMAGENET["input_dim"][1]
    use_atta: bool = args.adv_train == "atta"

    train_transforms = Compose(
        [
            RandomResizedCrop(img_size, return_params=use_atta),
            RandomHorizontalFlip(0.5, return_params=use_atta),
            ToTensor(),
        ]
    )
    val_transforms = Compose(
        [
            Resize(int(img_size * 256 / 224)),
            CenterCrop(img_size),
            ToTensor(),
        ]
    )

    train_loader, train_sampler = get_loader_sampler(
        args, train_transforms, "train"
    )
    val_loader, _ = get_loader_sampler(args, val_transforms, "val")
    test_loader, _ = get_loader_sampler(args, val_transforms, "test")

    return train_loader, train_sampler, val_loader, test_loader


PART_IMAGENET = {
    "normalize": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "loader": load_part_imagenet,
    "input_dim": (3, 224, 224),
    "colormap": COLORMAP,
}
