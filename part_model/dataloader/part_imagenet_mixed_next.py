import os

import numpy as np
import torch
import torch.utils.data as data
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

COLORMAP = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.22930429, 0.03104389, 0.1202147],
        [0.29363755, 0.05914433, 0.59998126],
        [0.59566164, 0.54861989, 0.8375412],
        [0.33823151, 0.23849806, 0.96362745],
        [0.34790637, 0.32292447, 0.67780972],
        [0.87156414, 0.00384938, 0.20607524],
        [0.79157960, 0.24604206, 0.53644435],
        [0.37290323, 0.01373695, 0.26237872],
    ]
)

FOLDER_TO_CLASS = {
    "n01440764": "Fish",
    "n01443537": "Fish",
    "n01484850": "Fish",
    "n01491361": "Fish",
    "n01494475": "Fish",
    "n01496331": "Fish",
    "n01498041": "Fish",
    "n01514668": "Bird",
    "n01514859": "Bird",
    "n01518878": "Bird",
    "n01530575": "Bird",
    "n01531178": "Bird",
    "n01532829": "Bird",
    "n01534433": "Bird",
    "n01537544": "Bird",
    "n01558993": "Bird",
    "n01560419": "Bird",
    "n01580077": "Bird",
    "n01582220": "Bird",
    "n01592084": "Bird",
    "n01601694": "Bird",
    "n01608432": "Bird",
    "n01614925": "Bird",
    "n01616318": "Bird",
    "n01622779": "Bird",
    "n01664065": "Reptile",
    "n01665541": "Reptile",
    "n01667114": "Reptile",
    "n01667778": "Reptile",
    "n01669191": "Reptile",
    "n01675722": "Reptile",
    "n01677366": "Reptile",
    "n01682714": "Reptile",
    "n01685808": "Reptile",
    "n01687978": "Reptile",
    "n01688243": "Reptile",
    "n01689811": "Reptile",
    "n01692333": "Reptile",
    "n01693334": "Reptile",
    "n01694178": "Reptile",
    "n01695060": "Reptile",
    "n01697457": "Reptile",
    "n01698640": "Reptile",
    "n01704323": "Reptile",
    "n01728572": "Reptile",
    "n01728920": "Reptile",
    "n01729322": "Reptile",
    "n01729977": "Reptile",
    "n01734418": "Reptile",
    "n01735189": "Reptile",
    "n01737021": "Reptile",
    "n01739381": "Reptile",
    "n01740131": "Reptile",
    "n01742172": "Reptile",
    "n01744401": "Reptile",
    "n01748264": "Reptile",
    "n01749939": "Reptile",
    "n01751748": "Reptile",
    "n01753488": "Reptile",
    "n01755581": "Reptile",
    "n01756291": "Reptile",
    "n01817953": "Bird",
    "n01818515": "Bird",
    "n01819313": "Bird",
    "n01820546": "Bird",
    "n01824575": "Bird",
    "n01828970": "Bird",
    "n01829413": "Bird",
    "n01833805": "Bird",
    "n01843065": "Bird",
    "n01843383": "Bird",
    "n01847000": "Bird",
    "n01855032": "Bird",
    "n01855672": "Bird",
    "n01860187": "Bird",
    "n02002556": "Bird",
    "n02002724": "Bird",
    "n02006656": "Bird",
    "n02007558": "Bird",
    "n02009229": "Bird",
    "n02009912": "Bird",
    "n02011460": "Bird",
    "n02012849": "Bird",
    "n02013706": "Bird",
    "n02017213": "Bird",
    "n02018207": "Bird",
    "n02018795": "Bird",
    "n02025239": "Bird",
    "n02027492": "Bird",
    "n02028035": "Bird",
    "n02033041": "Bird",
    "n02037110": "Bird",
    "n02051845": "Bird",
    "n02056570": "Bird",
    "n02058221": "Bird",
    "n02085620": "Quadruped",
    "n02085782": "Quadruped",
    "n02085936": "Quadruped",
    "n02086079": "Quadruped",
    "n02086240": "Quadruped",
    "n02086646": "Quadruped",
    "n02086910": "Quadruped",
    "n02087046": "Quadruped",
    "n02087394": "Quadruped",
    "n02088094": "Quadruped",
    "n02088238": "Quadruped",
    "n02088364": "Quadruped",
    "n02088466": "Quadruped",
    "n02088632": "Quadruped",
    "n02089078": "Quadruped",
    "n02089867": "Quadruped",
    "n02089973": "Quadruped",
    "n02090379": "Quadruped",
    "n02090622": "Quadruped",
    "n02090721": "Quadruped",
    "n02091244": "Quadruped",
    "n02091467": "Quadruped",
    "n02091635": "Quadruped",
    "n02091831": "Quadruped",
    "n02092002": "Quadruped",
    "n02092339": "Quadruped",
    "n02093256": "Quadruped",
    "n02093428": "Quadruped",
    "n02093647": "Quadruped",
    "n02093754": "Quadruped",
    "n02093859": "Quadruped",
    "n02093991": "Quadruped",
    "n02094114": "Quadruped",
    "n02094258": "Quadruped",
    "n02094433": "Quadruped",
    "n02095314": "Quadruped",
    "n02095570": "Quadruped",
    "n02095889": "Quadruped",
    "n02096051": "Quadruped",
    "n02096177": "Quadruped",
    "n02096294": "Quadruped",
    "n02096437": "Quadruped",
    "n02096585": "Quadruped",
    "n02097047": "Quadruped",
    "n02097130": "Quadruped",
    "n02097209": "Quadruped",
    "n02097298": "Quadruped",
    "n02097474": "Quadruped",
    "n02097658": "Quadruped",
    "n02098105": "Quadruped",
    "n02098286": "Quadruped",
    "n02098413": "Quadruped",
    "n02099267": "Quadruped",
    "n02099429": "Quadruped",
    "n02099601": "Quadruped",
    "n02099712": "Quadruped",
    "n02099849": "Quadruped",
    "n02100236": "Quadruped",
    "n02100583": "Quadruped",
    "n02100735": "Quadruped",
    "n02100877": "Quadruped",
    "n02101006": "Quadruped",
    "n02101388": "Quadruped",
    "n02101556": "Quadruped",
    "n02102040": "Quadruped",
    "n02102177": "Quadruped",
    "n02102318": "Quadruped",
    "n02102480": "Quadruped",
    "n02102973": "Quadruped",
    "n02104029": "Quadruped",
    "n02104365": "Quadruped",
    "n02105056": "Quadruped",
    "n02105162": "Quadruped",
    "n02105251": "Quadruped",
    "n02105412": "Quadruped",
    "n02105505": "Quadruped",
    "n02105641": "Quadruped",
    "n02105855": "Quadruped",
    "n02106030": "Quadruped",
    "n02106166": "Quadruped",
    "n02106382": "Quadruped",
    "n02106550": "Quadruped",
    "n02106662": "Quadruped",
    "n02107142": "Quadruped",
    "n02107312": "Quadruped",
    "n02107574": "Quadruped",
    "n02107683": "Quadruped",
    "n02107908": "Quadruped",
    "n02108000": "Quadruped",
    "n02108089": "Quadruped",
    "n02108422": "Quadruped",
    "n02108551": "Quadruped",
    "n02108915": "Quadruped",
    "n02109047": "Quadruped",
    "n02109525": "Quadruped",
    "n02109961": "Quadruped",
    "n02110063": "Quadruped",
    "n02110185": "Quadruped",
    "n02110341": "Quadruped",
    "n02110627": "Quadruped",
    "n02110806": "Quadruped",
    "n02110958": "Quadruped",
    "n02111129": "Quadruped",
    "n02111277": "Quadruped",
    "n02111500": "Quadruped",
    "n02111889": "Quadruped",
    "n02112018": "Quadruped",
    "n02112137": "Quadruped",
    "n02112350": "Quadruped",
    "n02112706": "Quadruped",
    "n02113023": "Quadruped",
    "n02113186": "Quadruped",
    "n02113624": "Quadruped",
    "n02113712": "Quadruped",
    "n02113799": "Quadruped",
    "n02113978": "Quadruped",
    "n02514041": "Fish",
    "n02526121": "Fish",
    "n02536864": "Fish",
    "n02606052": "Fish",
    "n02607072": "Fish",
    "n02640242": "Fish",
    "n02641379": "Fish",
    "n02643566": "Fish",
    "n02655020": "Fish",
}


class PartImageNetMixedDataset(data.Dataset):
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

        self.images, self.labels = self._get_data()
        self.masks = None
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
        _target = _img.copy()

        if self.transform is not None:
            _img, _ = self.transform(_img, _target)

        if self.use_label:
            _label = self.labels[index]
            return _img, _label
        return _img

    def _get_data(self):
        def getListOfFiles(dirName):
            # create a list of file and sub directories
            # names in the given directory
            listOfFile = os.listdir(dirName)
            allFiles = list()
            # Iterate over all the entries
            for entry in listOfFile:
                # Create full path
                fullPath = os.path.join(dirName, entry)
                # If entry is a directory then get the list of files in this directory
                if os.path.isdir(fullPath):
                    allFiles = allFiles + getListOfFiles(fullPath)
                else:
                    allFiles.append(fullPath)

            return allFiles

        images, labels = [], []

        images = getListOfFiles(os.path.join(self.root, self.split))
        for img in images:
            folderName = img.split("/")[-2]
            className = FOLDER_TO_CLASS[folderName]
            classInd = self.classes.index(className)
            labels.append(classInd)
        labels = torch.tensor(labels, dtype=torch.long)
        return images, labels

    def _list_classes(self):
        dirs = [
            "Aeroplane",
            "Bicycle",
            "Biped",
            "Bird",
            "Boat",
            "Bottle",
            "Car",
            "Fish",
            "Quadruped",
            "Reptile",
            "Snake",
        ]
        return sorted(dirs)

    def __len__(self):
        return len(self.images)


def get_loader_sampler(args, transform, split, distributed_sampler=True):
    # TODO: add mpgd if needed
    seg_type = get_seg_type(args)
    is_train = split == "train"

    part_imagenet_mixed_dataset = PartImageNetMixedDataset(
        args.data,
        args.seg_label_dir,
        split=split,
        transform=transform,
        seg_type=seg_type,
        use_label=("semi" in args.experiment) or (seg_type is None),
        seg_fraction=args.semi_label if is_train else 1.0,
    )

    sampler = None
    if args.distributed and distributed_sampler:
        if is_train:
            sampler = torch.utils.data.distributed.DistributedSampler(
                part_imagenet_mixed_dataset
            )
        else:
            # Use distributed sampler for validation but not testing
            sampler = DistributedEvalSampler(part_imagenet_mixed_dataset)

    batch_size = args.batch_size
    loader = torch.utils.data.DataLoader(
        part_imagenet_mixed_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )

    # TODO: can we make this cleaner?
    PART_IMAGENET_MIXED[
        "part_to_class"
    ] = part_imagenet_mixed_dataset.part_to_class
    PART_IMAGENET_MIXED["num_classes"] = part_imagenet_mixed_dataset.num_classes
    PART_IMAGENET_MIXED[
        "num_seg_labels"
    ] = part_imagenet_mixed_dataset.num_seg_labels

    setattr(args, "num_classes", part_imagenet_mixed_dataset.num_classes)
    pto = part_imagenet_mixed_dataset.part_to_object
    if seg_type == "part":
        seg_labels = len(pto)
    elif seg_type == "fg":
        seg_labels = 2
    else:
        seg_labels = pto.max().item() + 1
    setattr(args, "seg_labels", seg_labels)

    return loader, sampler


def load_part_imagenet(args):

    img_size = PART_IMAGENET_MIXED["input_dim"][1]

    val_transforms = Compose(
        [
            Resize(int(img_size * 256 / 224)),
            CenterCrop(img_size),
            ToTensor(),
        ]
    )

    val_loader, _ = get_loader_sampler(args, val_transforms, "val")
    test_loader, _ = get_loader_sampler(args, val_transforms, "test")

    return None, None, val_loader, test_loader


PART_IMAGENET_MIXED = {
    "normalize": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "loader": load_part_imagenet,
    "input_dim": (3, 224, 224),
    "colormap": COLORMAP,
}
