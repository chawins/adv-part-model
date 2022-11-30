"""
Code is adapted from 
https://github.com/chenxi116/DeepLabv3.pytorch/blob/master/cityscapes.py
https://github.com/chenxi116/DeepLabv3.pytorch/blob/master/utils.py
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch
import torch.utils.data as data
from part_model.dataloader.util import COLORMAP
from part_model.utils import get_seg_type, np_temp_seed
from part_model.utils.eval_sampler import DistributedEvalSampler
from PIL import Image

from .segmentation_transforms import (
    CenterCrop,
    Compose,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

_FOLDERS_MAP = {
    "image": "images",
    "label": "panoptic-parts",
}

_DATA_FORMAT_MAP = {
    "image": ["png", "jpg"],
    "label": "tif",
}


class PascalPartSegDataset(data.Dataset):
    def __init__(
        self,
        root,
        seg_path,
        split="train",
        transform=None,
        use_label=False,
        seg_type=None,
        seg_fraction=1.0,
        seed=0,
    ):
        self.root = root
        self.transform = transform
        self.split = split
        self.use_label = use_label
        self.seg_type = seg_type
        self.path = os.path.join(seg_path, split)

        self.classes = self._list_classes()
        self.num_classes = len(self.classes)
        # We group all parts such that all classes have three parts
        self.num_seg_labels = self.num_classes * 3
        self.images, self.labels, self.masks = self._get_data()
        idx = np.arange(len(self.images))
        with np_temp_seed(seed):
            np.random.shuffle(idx)
        self.seg_drop_idx = idx[: int((1 - seg_fraction) * len(self.images))]

        # Create matrix that maps part segmentation to object segmentation
        part_to_object = [0]
        self.part_to_class = [[0] * (self.num_classes + 1)]
        self.part_to_class[0][0] = 1
        for i, label in enumerate(self.classes):
            part_to_object.extend([i + 1] * 3)
            base = [0] * (self.num_classes + 1)
            base[i + 1] = 1
            self.part_to_class.extend([base] * 3)
        self.part_to_object = torch.tensor(part_to_object, dtype=torch.long)

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        _target = Image.open(self.masks[index])

        if self.transform is not None:
            _img, _target = self.transform(_img, _target)

        if self.seg_type is not None:
            if self.seg_type == "object":
                _target = self.part_to_object[_target]
            elif self.seg_type == "fg":
                _target = (_target > 0).long()
            if index in self.seg_drop_idx:
                # Drop segmentation mask by setting all pixels to -1 to ignore
                # later at loss computation
                _target.mul_(0).add_(-1)
            if self.use_label:
                _label = self.labels[index]
                return _img, _target, _label
            return _img, _target

        if self.use_label:
            _label = self.labels[index]
            return _img, _label

        return _img

    def _get_files_ext(self, data, ext):
        path = os.path.join(self.root, _FOLDERS_MAP[data], self.split)
        filenames, labels = [], []
        for l, clss in enumerate(self.classes):
            search_files = os.path.join(path, clss, f"*.{ext}")
            search_files = glob.glob(search_files)
            filenames.extend(search_files)
            labels.extend([l] * len(search_files))
        return filenames, labels

    def _get_files(self, data):
        exts = _DATA_FORMAT_MAP[data]
        if not isinstance(exts, list):
            exts = [exts]
        filenames, labels = [], []
        for ext in exts:
            fn, lb = self._get_files_ext(data, ext)
            filenames.extend(fn)
            labels.extend(lb)
        labels = torch.tensor(labels, dtype=torch.long)
        # Sort by only the filenames without path
        only_filenames = [f.split("/")[-1] for f in filenames]
        sorted_idx = np.argsort(only_filenames)
        return np.array(filenames)[sorted_idx], labels[sorted_idx]

    def _get_data(self):
        images, labels = self._get_files("image")
        masks, _ = self._get_files("label")
        assert len(images) == len(labels) == len(masks)
        return images, labels, masks

    def _list_classes(self):
        path = os.path.join(self.root, _FOLDERS_MAP["image"], self.split)
        dirs = os.listdir(path)
        dirs = [d for d in dirs if os.path.isdir(os.path.join(path, d))]
        return sorted(dirs)

    def __len__(self):
        return len(self.images)


def get_loader_sampler(args, transform, split):
    seg_type = get_seg_type(args)
    is_train = split == "train"
    dataset = PascalPartSegDataset(
        args.data,
        args.seg_label_dir,
        split=split,
        transform=transform,
        seg_type=seg_type,
        use_label=("semi" in args.experiment) or (seg_type is None),
        seg_fraction=args.semi_label if is_train else 1.0,
    )

    sampler = None
    if args.distributed:
        shuffle = None
        if is_train:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=True, seed=args.seed, drop_last=False
            )
        else:
            # Use distributed sampler for validation but not testing
            sampler = DistributedEvalSampler(
                dataset, shuffle=False, seed=args.seed
            )
    else:
        shuffle = is_train

    batch_size = args.batch_size
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )

    PASCAL_PART["part_to_class"] = dataset.part_to_class
    PASCAL_PART["num_classes"] = dataset.num_classes
    PASCAL_PART["num_seg_labels"] = dataset.num_seg_labels

    setattr(args, "num_classes", dataset.num_classes)
    pto = dataset.part_to_object
    if seg_type == "part":
        seg_labels = len(pto)
    elif seg_type == "fg":
        seg_labels = 2
    else:
        seg_labels = pto.max().item() + 1
    setattr(args, "seg_labels", seg_labels)

    return loader, sampler


def load_pascal_part(args):

    img_size = PASCAL_PART["input_dim"][1]

    train_transforms = Compose(
        [
            RandomResizedCrop(img_size),
            RandomHorizontalFlip(0.5),
            ToTensor(),
        ]
    )
    val_transforms = Compose(
        [
            Resize(int(img_size * 256 / 224)),
            CenterCrop(
                img_size
            ),  # Added in for prepare_pascal_part_v3.py. Should not affect earlier version.
            ToTensor(),
        ]
    )

    train_loader, train_sampler = get_loader_sampler(
        args, train_transforms, "train"
    )
    val_loader, _ = get_loader_sampler(args, val_transforms, "val")
    test_loader, _ = get_loader_sampler(args, val_transforms, "test")

    return train_loader, train_sampler, val_loader, test_loader


PASCAL_PART = {
    "normalize": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "loader": load_pascal_part,
    "input_dim": (3, 224, 224),
    "colormap": COLORMAP,
}
