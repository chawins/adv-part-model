"""Data loading for detection model (e.g., DINO).

TODO(nabeel@): Did you change things in this file significantly? Or did you just
copy it from somewhere? Is there a way we can mostly just import from DINO and
implment as needed?
"""

import os
import json
import random
from pathlib import Path

import PIL
import torch
import torchvision
from torchvision.transforms import RandomResizedCrop

import DINO.datasets.transforms as T
from DINO.datasets.coco import ConvertCocoPolysToMask
from DINO.datasets.transforms import crop, resize
from DINO.util.misc import collate_fn
from part_model.dataloader.util import COLORMAP
from part_model.utils.eval_sampler import DistributedEvalSampler

class RandomCrop(object):
    def __init__(
        self,
        scale=(0.08, 1.0),
        ratio=(0.75, 1.333333),
    ) -> None:
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img: PIL.Image.Image, target: dict):
        region = RandomResizedCrop.get_params(img, self.scale, self.ratio)
        img, target = crop(img, target, region)
        return img, target

class Resize(object):
    def __init__(self, size) -> None:
        self.size: int = size

    def __call__(self, img: PIL.Image.Image, target: dict):
        img, target = resize(img, target, self.size)
        return img, target


class PartImageNetBBOXDataset(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        aux_target_hacks=None,
    ):
        super(PartImageNetBBOXDataset, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(False)
        self.aux_target_hacks = aux_target_hacks

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
        CLASSES = CLASSES.keys()
        CLASSES = sorted(CLASSES)
        
        self.num_classes = len(CLASSES)
        
        self.category_id_to_supercategory = {}
        with open(ann_file) as f:
            annotations = json.load(f)

        for ann in annotations["categories"]:
            category_id = ann["id"]
            supercategory = ann["supercategory"]
            if supercategory in CLASSES:
                self.category_id_to_supercategory[category_id] = CLASSES.index(supercategory)

        self.num_seg_classes = len(self.category_id_to_supercategory)
        
        self.imageid_to_label = {}
        for ann in annotations["annotations"]:
            image_id = ann["image_id"]
            category_id = ann["category_id"]
            if image_id not in self.imageid_to_label:
                self.imageid_to_label[image_id] = self.category_id_to_supercategory[category_id]
            else:
                assert self.imageid_to_label[image_id] == self.category_id_to_supercategory[category_id]

    def __getitem__(self, idx):
        """
        Output:
            - target: dict of multiple items
                - boxes: Tensor[num_box, 4]. \
                    Init type: x0,y0,x1,y1. unnormalized data.
                    Final type: cx,cy,w,h. normalized data. 
        """
        try:
            img, target = super(PartImageNetBBOXDataset, self).__getitem__(idx)
        except:
            print("Error idx: {}".format(idx))
            idx += 1
            img, target = super(PartImageNetBBOXDataset, self).__getitem__(idx)

        image_id = self.ids[idx]

        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)

        if self._transforms is not None:
            img, target = self._transforms(img, target)
       
        class_label = self.imageid_to_label[image_id]
        return img, target, class_label


def get_loader_sampler_bbox(args, transforms, split):
    is_train = split == "train"
    img_folder_path = os.path.join(
        os.path.expanduser(args.data), "JPEGImages"
    )
    
    ann_file_path = os.path.join(
        os.path.expanduser(args.data), "PartBoxSegmentations", f"{split}.json"
    )

    aux_target_hacks_list = None
    part_imagenet_dataset = PartImageNetBBOXDataset(
        img_folder_path,
        ann_file_path,
        transforms,
        aux_target_hacks=aux_target_hacks_list,
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
        collate_fn=collate_fn,  # turns images into NestedTensors
    )

    PART_IMAGENET_BBOX["num_classes"] = part_imagenet_dataset.num_classes
    setattr(args, "num_classes", part_imagenet_dataset.num_classes)

    PART_IMAGENET_BBOX["num_seg_classes"] = part_imagenet_dataset.num_seg_classes
    setattr(args, "seg_labels", part_imagenet_dataset.num_seg_classes)

    return loader, sampler


def load_part_imagenet_bbox(args):
    img_size = PART_IMAGENET_BBOX["input_dim"][1]

    train_transforms = T.Compose(
        [
            RandomCrop(),
            Resize([img_size, img_size]),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0, 0, 0], [1, 1, 1]),
        ]
    )

    val_transforms = T.Compose(
        [
            Resize([int(img_size * 256 / 224), int(img_size * 256 / 224)]),
            T.CenterCrop([img_size, img_size]),
            T.ToTensor(),
            T.Normalize([0, 0, 0], [1, 1, 1]),
        ]
    )

    train_loader, train_sampler = get_loader_sampler_bbox(
        args, train_transforms, "train"
    )
    val_loader, _ = get_loader_sampler_bbox(args, val_transforms, "val")
    test_loader, _ = get_loader_sampler_bbox(args, val_transforms, "test")

    return train_loader, train_sampler, val_loader, test_loader


PART_IMAGENET_BBOX = {
    "normalize": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "loader": load_part_imagenet_bbox,
    "input_dim": (3, 224, 224),
    "colormap": COLORMAP,
}
