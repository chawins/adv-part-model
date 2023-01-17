import os

import numpy as np
import torch
import torch.utils.data as data
from part_model.dataloader.util import COLORMAP
from part_model.utils import get_seg_type, np_temp_seed
from part_model.utils.eval_sampler import DistributedEvalSampler
from part_model.utils.image import get_seg_type
from PIL import Image

from .segmentation_transforms import (
    CenterCrop,
    Compose,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

CLASSES = {
    "n01440764": 4,
    "n01443537": 4,
    "n01484850": 4,
    "n01491361": 4,
    "n01494475": 4,
    "n01608432": 5,
    "n01614925": 5,
    "n01630670": 4,
    "n01632458": 4,
    "n01641577": 4,
    "n01644373": 4,
    "n01644900": 4,
    "n01664065": 4,
    "n01665541": 4,
    "n01667114": 4,
    "n01667778": 4,
    "n01669191": 4,
    "n01685808": 4,
    "n01687978": 4,
    "n01688243": 4,
    "n01689811": 4,
    "n01692333": 4,
    "n01693334": 4,
    "n01694178": 4,
    "n01695060": 4,
    "n01697457": 4,
    "n01698640": 4,
    "n01728572": 2,
    "n01728920": 2,
    "n01729322": 2,
    "n01729977": 2,
    "n01734418": 2,
    "n01735189": 2,
    "n01739381": 2,
    "n01740131": 2,
    "n01742172": 2,
    "n01744401": 2,
    "n01748264": 2,
    "n01749939": 2,
    "n01753488": 2,
    "n01755581": 2,
    "n01756291": 2,
    "n01824575": 5,
    "n01828970": 5,
    "n01843065": 5,
    "n01855672": 5,
    "n02002724": 5,
    "n02006656": 5,
    "n02009229": 5,
    "n02009912": 5,
    "n02017213": 5,
    "n02025239": 5,
    "n02033041": 5,
    "n02058221": 5,
    "n02071294": 4,
    "n02085782": 4,
    "n02089867": 4,
    "n02090379": 4,
    "n02091831": 4,
    "n02092339": 4,
    "n02096177": 4,
    "n02096585": 4,
    "n02097474": 4,
    "n02098105": 4,
    "n02099601": 4,
    "n02100583": 4,
    "n02101006": 4,
    "n02101388": 4,
    "n02102040": 4,
    "n02102973": 4,
    "n02109525": 4,
    "n02109961": 4,
    "n02112137": 4,
    "n02114367": 4,
    "n02120079": 4,
    "n02124075": 4,
    "n02125311": 4,
    "n02128385": 4,
    "n02129604": 4,
    "n02130308": 4,
    "n02132136": 4,
    "n02133161": 4,
    "n02134084": 4,
    "n02134418": 4,
    "n02356798": 4,
    "n02397096": 4,
    "n02403003": 4,
    "n02408429": 4,
    "n02412080": 4,
    "n02415577": 4,
    "n02417914": 4,
    "n02422106": 4,
    "n02422699": 4,
    "n02423022": 4,
    "n02437312": 4,
    "n02441942": 4,
    "n02442845": 4,
    "n02443114": 4,
    "n02444819": 4,
    "n02447366": 4,
    "n02480495": 5,
    "n02480855": 5,
    "n02481823": 5,
    "n02483362": 5,
    "n02483708": 5,
    "n02484975": 5,
    "n02486261": 5,
    "n02486410": 5,
    "n02487347": 5,
    "n02488702": 5,
    "n02489166": 5,
    "n02490219": 5,
    "n02492035": 5,
    "n02492660": 5,
    "n02493509": 5,
    "n02493793": 5,
    "n02494079": 5,
    "n02510455": 4,
    "n02514041": 4,
    "n02536864": 4,
    "n02607072": 4,
    "n02655020": 4,
    "n02690373": 5,
    "n02701002": 3,
    "n02814533": 3,
    "n02823428": 2,
    "n02835271": 4,
    "n02930766": 3,
    "n03100240": 3,
    "n03417042": 3,
    "n03444034": 3,
    "n03445924": 3,
    "n03594945": 3,
    "n03670208": 3,
    "n03769881": 3,
    "n03770679": 3,
    "n03785016": 4,
    "n03791053": 4,
    "n03792782": 4,
    "n03937543": 2,
    "n03947888": 2,
    "n03977966": 3,
    "n03983396": 2,
    "n04037443": 3,
    "n04065272": 3,
    "n04146614": 3,
    "n04147183": 2,
    "n04252225": 3,
    "n04285008": 3,
    "n04465501": 3,
    "n04482393": 4,
    "n04483307": 2,
    "n04487081": 3,
    "n04509417": 4,
    "n04552348": 5,
    "n04557648": 2,
    "n04591713": 2,
    "n04612504": 2,
}


class PartImageNetSegDataset(data.Dataset):
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
        """Load our processed Part-ImageNet dataset

        Args:
            root (str): Path to root directory
            split (str, optional): Data split to load. Defaults to 'train'.
            transform (optional): Transformations to apply to the images (and
                the segmentation masks if applicable). Defaults to None.
            use_label (bool, optional): Whether to yield class label. Defaults to False.
            seg_type (str, optional): Specify types of segmentation to load
                ('part', 'object', or None). Defaults to 'part'.
            seg_fraction (float, optional): Fraction of segmentation mask to
                provide. The dropped masks are set to all -1. Defaults to 1.
            seed (int, optional): Random seed. Defaults to 0.
        """
        self.root = root
        self.split = split
        self.path = os.path.join(seg_path, split)
        self.transform = transform
        self.use_label = use_label
        self.seg_type = seg_type

        self.classes = self._list_classes()
        self.num_classes = len(self.classes)
        self.num_seg_labels = sum([CLASSES[c] for c in self.classes])

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
            part_to_object.extend([i + 1] * CLASSES[label])
            base = [0] * (self.num_classes + 1)
            base[i + 1] = 1
            self.part_to_class.extend([base] * CLASSES[label])
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

    def _get_data(self):
        images, labels, masks = [], [], []
        for l, label in enumerate(self.classes):
            img_path = os.path.join(self.root, "JPEGImages")
            part_path = os.path.join(self.path, label)
            # Read file names
            with open(f"{self.path}/{label}.txt", "r") as fns:
                filenames = sorted([f.strip() for f in fns.readlines()])
            images.extend([f"{img_path}/{f}.JPEG" for f in filenames])
            masks.extend(
                [f'{part_path}/{f.split("/")[1]}.png' for f in filenames]
            )
            labels.extend([l] * len(filenames))
        labels = torch.tensor(labels, dtype=torch.long)
        return images, labels, masks

    def _list_classes(self):
        dirs = os.listdir(self.path)
        dirs = [d for d in dirs if os.path.isdir(os.path.join(self.path, d))]
        return sorted(dirs)

    def __len__(self):
        return len(self.images)


def get_loader_sampler(args, transform, split):
    # TODO: add mpgd if needed
    seg_type = get_seg_type(args)
    is_train = split == "train"

    part_imagenet_dataset = PartImageNetSegDataset(
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
    else:
        # shuffle = is_train
        shuffle = True

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
    PART_IMAGENET_IMAGENET_CLASS[
        "part_to_class"
    ] = part_imagenet_dataset.part_to_class
    PART_IMAGENET_IMAGENET_CLASS[
        "num_classes"
    ] = part_imagenet_dataset.num_classes
    PART_IMAGENET_IMAGENET_CLASS[
        "num_seg_labels"
    ] = part_imagenet_dataset.num_seg_labels

    setattr(args, "num_classes", part_imagenet_dataset.num_classes)
    pto = part_imagenet_dataset.part_to_object
    if seg_type == "part":
        seg_labels = len(pto)
    elif seg_type == "fg":
        seg_labels = 2
    else:
        seg_labels = pto.max().item() + 1
    setattr(args, "seg_labels", seg_labels)

    return loader, sampler


def load_part_imagenet(args):

    img_size = PART_IMAGENET_IMAGENET_CLASS["input_dim"][1]

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


PART_IMAGENET_IMAGENET_CLASS = {
    "normalize": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "loader": load_part_imagenet,
    "input_dim": (3, 224, 224),
    "colormap": COLORMAP,
}
