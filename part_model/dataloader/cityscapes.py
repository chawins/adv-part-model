"""
Code is adapted from 
https://github.com/chenxi116/DeepLabv3.pytorch/blob/master/cityscapes.py
https://github.com/chenxi116/DeepLabv3.pytorch/blob/master/utils.py
"""
import glob
import os

import numpy as np
import torch
import torch.utils.data as data
from panoptic_parts.utils.format import decode_uids
from part_model.dataloader.util import COLORMAP
from part_model.utils import get_seg_type, np_temp_seed
from PIL import Image

from ..utils.eval_sampler import DistributedEvalSampler
from .segmentation_transforms import (
    CenterCrop,
    Compose,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

CLASSES = {
    "human": 4,
    "vehicle": 5,
}

_FOLDERS_MAP = {
    "image": "images",
    "label": "panoptic-parts",
}

_DATA_FORMAT_MAP = {
    "image": "png",
    "label": "tif",
}

# human: (24, 25), (1: torso, 2: head, 3: arms, 4: legs)
# vehicle: (26, 27, 28), (1: windows, 2: wheels, 3: lights, 4: license plate, 5: chassis)
_LABEL_ASSIGN = {
    (24, 1): 1,
    (24, 2): 2,
    (24, 3): 3,
    (24, 4): 4,
    (25, 1): 1,
    (25, 2): 2,
    (25, 3): 3,
    (25, 4): 4,
    (26, 1): 5,
    (26, 2): 6,
    (26, 3): 7,
    (26, 4): 8,
    (26, 5): 9,
    (27, 1): 5,
    (27, 2): 6,
    (27, 3): 7,
    (27, 4): 8,
    (27, 5): 9,
    (28, 1): 5,
    (28, 2): 6,
    (28, 3): 7,
    (28, 4): 8,
    (28, 5): 9,
}


class MaskToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, mask):
        return torch.from_numpy(np.array(mask).astype(np.int64))


def uids_to_seg_labels(_target):
    _target = np.array(_target)
    target = decode_uids(_target)
    target = np.stack([target[0], target[2]], axis=-1)
    _target = np.zeros_like(_target, dtype=np.uint8)
    for p in _LABEL_ASSIGN:
        _target[np.all(target == p, axis=-1)] = _LABEL_ASSIGN[p]
    return _target


class CityscapesSegDataset(data.Dataset):
    def __init__(
        self,
        root,
        split="train",
        transform=None,
        seg_type=None,
        use_label=False,
        seg_fraction=1.0,
        seed=0,
    ):
        self.root = root
        self.transform = transform
        self.split = split
        self.use_label = use_label
        self.seg_type = seg_type

        self.images = self._get_files("image")
        self.masks = self._get_files("label")
        self.classes = self._list_classes()
        self.num_classes = len(self.classes)
        self.num_seg_labels = sum([CLASSES[c] for c in self.classes])
        idx = np.arange(len(self.images))
        with np_temp_seed(seed):
            np.random.shuffle(idx)
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

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        _target = Image.open(self.masks[index])
        _target = uids_to_seg_labels(_target)
        _target = Image.fromarray(_target)

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
                _label = self.images[index].split("/")[-2]
                _label = self.classes.index(_label)
                return _img, _target, _label
            return _img, _target

        if self.use_label:
            _label = self.images[index].split("/")[-2]
            _label = self.classes.index(_label)
            return _img, _label
        return _img

    def _get_files(self, data):
        pattern = f"*.{_DATA_FORMAT_MAP[data]}"
        search_files = os.path.join(
            self.root, _FOLDERS_MAP[data], self.split, "*", pattern
        )
        filenames = glob.glob(search_files)
        return sorted(filenames)

    def _list_classes(self):
        path = os.path.join(self.root, _FOLDERS_MAP["image"], self.split)
        dirs = os.listdir(path)
        dirs = [d for d in dirs if os.path.isdir(os.path.join(path, d))]
        return sorted(dirs)

    def __len__(self):
        return len(self.images)


def get_loader_sampler(args, transform, split, distributed_sampler=True):
    seg_type = get_seg_type(args)
    is_train = split == "train"
    dataset = CityscapesSegDataset(
        args.data,
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

    CITYSCAPES["part_to_class"] = dataset.part_to_class
    CITYSCAPES["num_classes"] = dataset.num_classes
    CITYSCAPES["num_seg_labels"] = dataset.num_seg_labels

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


def load_cityscapes(args):
    img_size = CITYSCAPES["input_dim"][1]
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


CITYSCAPES = {
    "normalize": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "loader": load_cityscapes,
    "input_dim": (3, 224, 224),
    "colormap": COLORMAP,
}
