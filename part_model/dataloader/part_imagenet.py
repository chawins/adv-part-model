import os

import numpy as np
import json
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
                [f'{part_path}/{f.split("/")[1]}.tif' for f in filenames]
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
    PART_IMAGENET["part_to_class"] = part_imagenet_dataset.part_to_class
    PART_IMAGENET["num_classes"] = part_imagenet_dataset.num_classes
    PART_IMAGENET["num_seg_labels"] = part_imagenet_dataset.num_seg_labels

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

    img_size = PART_IMAGENET["input_dim"][1]

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


PART_IMAGENET = {
    "normalize": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "loader": load_part_imagenet,
    "input_dim": (3, 224, 224),
    "colormap": COLORMAP,
}






















import torchvision
from DINO.datasets.coco import ConvertCocoPolysToMask, dataset_hook_register
from pathlib import Path

class PartImageNetBBOXDataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, class_label_file, ann_file, transforms, aux_target_hacks=None):
        super(PartImageNetBBOXDataset, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(False)
        self.aux_target_hacks = aux_target_hacks

        with open(class_label_file, 'r') as openfile:
            self.image_to_label = json.load(openfile)

        self.classes = list(sorted(set(self.image_to_label.values())))
        self.num_classes = len(self.classes)


    def change_hack_attr(self, hackclassname, attrkv_dict):
        target_class = dataset_hook_register[hackclassname]
        for item in self.aux_target_hacks:
            if isinstance(item, target_class):
                for k,v in attrkv_dict.items():
                    setattr(item, k, v)

    def get_hack(self, hackclassname):
        target_class = dataset_hook_register[hackclassname]
        for item in self.aux_target_hacks:
            if isinstance(item, target_class):
                return item

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
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        # convert to needed format
        if self.aux_target_hacks is not None:
            for hack_runner in self.aux_target_hacks:
                target, img = hack_runner(target, img=img)

        class_label = self.image_to_label[str(idx)]
        return img, target, class_label


# from DINO.datasets.coco import CocoDetection

# class PartImageNetBBOXDataset(CocoDetection):
#     def __init__(
#         self, 
#         img_folder, 
#         class_label_file,
#         ann_file, 
#         transforms, 
#         aux_target_hacks=None
#     ):
#         super(PartImageNetBBOXDataset, self).__init__(img_folder, ann_file, transforms, False, aux_target_hacks)

#         with open(class_label_file, 'r') as openfile:
#             self.image_to_label = json.load(openfile)

#         self.classes = list(sorted(set(self.image_to_label.values())))
#         self.num_classes = len(self.classes)
        
#     def __getitem__(self, index):
#         img, target = super().__getitem__(index)
#         # print(type(img))
#         # qq
#         class_label = self.image_to_label[str(index)]
#         return img, target, class_label



def get_loader_sampler_bbox(args, transforms, split):
    is_train = split == "train"

    # TODO: add as arg
    root = Path('/data/shared/PartImageNet/PartBoxSegmentations')

    PATHS = {
        "train": (root / "train", root / "image_labels" / 'train.json', root / "annotations" / 'train.json'),
        "val": (root / "val", root / "image_labels" / 'val.json', root / "annotations" / 'val.json'),
        "test": (root / "test", root / "image_labels" / 'test.json', root / "annotations" / 'test.json' ),
    }

    img_folder, class_label_file, ann_file = PATHS[split]

    part_imagenet_dataset = PartImageNetBBOXDataset(
        img_folder, 
        class_label_file,
        ann_file, 
        transforms, 
        aux_target_hacks=None
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

    from DINO.util.misc import collate_fn
    batch_size = args.batch_size
    loader = torch.utils.data.DataLoader(
        part_imagenet_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=collate_fn
    )

    # TODO: can we make this cleaner?
    # PART_IMAGENET["part_to_class"] = part_imagenet_dataset.part_to_class
    PART_IMAGENET["num_classes"] = part_imagenet_dataset.num_classes
    # PART_IMAGENET["num_seg_labels"] = part_imagenet_dataset.num_seg_labels

    setattr(args, "num_classes", part_imagenet_dataset.num_classes)
    # pto = part_imagenet_dataset.part_to_object
    # if seg_type == "part":
    #     seg_labels = len(pto)
    # elif seg_type == "fg":
    #     seg_labels = 2
    # else:
    #     seg_labels = pto.max().item() + 1
    # setattr(args, "seg_labels", seg_labels)

    return loader, sampler


import DINO.datasets.transforms as T

def make_coco_transforms(image_set, fix_size=False, strong_aug=False, args=None):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # config the params for data aug
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333
    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]
    
    # update args from config files
    scales = getattr(args, 'data_aug_scales', scales)
    max_size = getattr(args, 'data_aug_max_size', max_size)
    scales2_resize = getattr(args, 'data_aug_scales2_resize', scales2_resize)
    scales2_crop = getattr(args, 'data_aug_scales2_crop', scales2_crop)

    # resize them
    data_aug_scale_overlap = getattr(args, 'data_aug_scale_overlap', None)
    if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
        data_aug_scale_overlap = float(data_aug_scale_overlap)
        scales = [int(i*data_aug_scale_overlap) for i in scales]
        max_size = int(max_size*data_aug_scale_overlap)
        scales2_resize = [int(i*data_aug_scale_overlap) for i in scales2_resize]
        scales2_crop = [int(i*data_aug_scale_overlap) for i in scales2_crop]
    # else:
    #     scales = getattr(args, 'data_aug_scales', scales)
    #     max_size = getattr(args, 'data_aug_max_size', max_size)
    #     scales2_resize = getattr(args, 'data_aug_scales2_resize', scales2_resize)
    #     scales2_crop = getattr(args, 'data_aug_scales2_crop', scales2_crop)

    datadict_for_print = {
        'scales': scales,
        'max_size': max_size,
        'scales2_resize': scales2_resize,
        'scales2_crop': scales2_crop
    }
    print("data_aug_params:", json.dumps(datadict_for_print, indent=2))
        

    if image_set == 'train':
        if fix_size:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResize([(max_size, max(scales))]),
                # T.RandomResize([(512, 512)]),
                normalize,
            ])
        
        if strong_aug:
            import DINO.datasets.sltransform as SLT
            
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose([
                        T.RandomResize(scales2_resize),
                        T.RandomSizeCrop(*scales2_crop),
                        T.RandomResize(scales, max_size=max_size),
                    ])
                ),
                SLT.RandomSelectMulti([
                    SLT.RandomCrop(),
                    # SLT.Rotate(10),
                    SLT.LightingNoise(),
                    SLT.AdjustBrightness(2),
                    SLT.AdjustContrast(2),
                ]),              
                # # for debug only  
                # SLT.RandomCrop(),
                # SLT.LightingNoise(),
                # SLT.AdjustBrightness(2),
                # SLT.AdjustContrast(2),
                # SLT.Rotate(10),
                normalize,
            ])
        
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(scales2_resize),
                    T.RandomSizeCrop(*scales2_crop),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])

    if image_set in ['val', 'eval_debug', 'train_reg', 'test']:

        if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == 'INFO':
            print("Under debug mode for flops calculation only!!!!!!!!!!!!!!!!")
            return T.Compose([
                T.ResizeDebug((1280, 800)),
                normalize,
            ])   

        return T.Compose([
            T.RandomResize([max(scales)], max_size=max_size),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def load_part_imagenet_bbox(args):
    try:
        strong_aug = args.strong_aug
    except:
        strong_aug = False
    train_transforms = make_coco_transforms("train", fix_size=args.fix_size, strong_aug=strong_aug, args=args)
    train_loader, train_sampler = get_loader_sampler_bbox(
        args, train_transforms, "train"
    )
    
    val_transforms = make_coco_transforms("val", fix_size=args.fix_size, strong_aug=False, args=args)
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
