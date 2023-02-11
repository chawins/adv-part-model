"""ImageNet dataloader with segmentation labels."""

from __future__ import annotations

import torch

from part_model.dataloader import part_imagenet
from part_model.dataloader.segmentation_transforms import (
    CenterCrop,
    Compose,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from part_model.dataloader.util import COLORMAP
from part_model.utils.eval_sampler import DistributedEvalSampler
from part_model.utils.image import get_seg_type


class ImageNetSegDataset(part_imagenet.PartImageNetSegDataset):
    """ImageNet Dataset."""

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


def _get_loader_sampler(args, transform, split: str):
    seg_type: str = get_seg_type(args)
    is_train: bool = split == "train"
    use_atta: bool = args.adv_train == "atta"

    imagenet_dataset = ImageNetSegDataset(
        args.data,
        args.seg_label_dir,
        split=split,
        transform=transform,
        seg_type=seg_type,
        use_label=("semi" in args.experiment) or (seg_type is None),
        seg_fraction=args.semi_label if is_train else 1.0,
        use_atta=use_atta,
    )

    sampler: torch.utils.data.Sampler | None = None
    shuffle: bool | None = is_train
    if args.distributed:
        shuffle = None
        if is_train:
            sampler = torch.utils.data.distributed.DistributedSampler(
                imagenet_dataset,
                shuffle=True,
                seed=args.seed,
                drop_last=False,
            )
        else:
            # Use distributed sampler for validation but not testing
            sampler = DistributedEvalSampler(
                imagenet_dataset, shuffle=False, seed=args.seed
            )

    batch_size = args.batch_size
    loader = torch.utils.data.DataLoader(
        imagenet_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )

    # TODO(chawins@): can we make this cleaner?
    IMAGENET["part_to_class"] = imagenet_dataset.part_to_class
    IMAGENET["num_classes"] = imagenet_dataset.num_classes
    IMAGENET["num_seg_labels"] = imagenet_dataset.num_seg_labels

    pto = imagenet_dataset.part_to_object
    if seg_type == "part":
        seg_labels = len(pto)
    elif seg_type == "fg":
        seg_labels = 2
    else:
        seg_labels = pto.max().item() + 1

    setattr(args, "seg_labels", seg_labels)
    setattr(args, "num_classes", imagenet_dataset.num_classes)
    setattr(args, "input_dim", IMAGENET["input_dim"])
    if is_train:
        setattr(args, "num_train", len(imagenet_dataset))

    return loader, sampler


def load_imagenet(args):
    """Load dataloaders for ImageNetSegDataset."""
    img_size = IMAGENET["input_dim"][1]
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

    train_loader, train_sampler = _get_loader_sampler(
        args, train_transforms, "train"
    )
    val_loader, _ = _get_loader_sampler(args, val_transforms, "val")
    test_loader, _ = _get_loader_sampler(args, val_transforms, "test")

    return train_loader, train_sampler, val_loader, test_loader


IMAGENET = {
    "normalize": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "loader": load_imagenet,
    "input_dim": (3, 224, 224),
    "colormap": COLORMAP,
}
