from part_model.dataloader.cityscapes import CITYSCAPES
from part_model.dataloader.imagenet import IMAGENET
from part_model.dataloader.part_imagenet import PART_IMAGENET
from part_model.dataloader.part_imagenet_bbox import PART_IMAGENET_BBOX
from part_model.dataloader.part_imagenet_corrupt import PART_IMAGENET_CORRUPT
from part_model.dataloader.part_imagenet_geirhos import PART_IMAGENET_GEIRHOS
from part_model.dataloader.part_imagenet_mixed_next import PART_IMAGENET_MIXED
from part_model.dataloader.part_imagenet_imagenet_class import (
    PART_IMAGENET_IMAGENET_CLASS,
)
from part_model.dataloader.part_imagenet_pseudo import PART_IMAGENET_PSEUDO
from part_model.dataloader.part_imagenet_pseudo_imagenet_class import (
    PART_IMAGENET_PSEUDO_IMAGENET_CLASS,
)
from part_model.dataloader.pascal_part import PASCAL_PART
from part_model.dataloader.pascal_voc import PASCAL_VOC
from part_model.dataloader.util import COLORMAP

DATASET_DICT = {
    "cityscapes": CITYSCAPES,
    "pascal-part": PASCAL_PART,
    "pascal-voc": PASCAL_VOC,
    "part-imagenet": PART_IMAGENET,
    "part-imagenet-geirhos": PART_IMAGENET_GEIRHOS,
    "part-imagenet-mixed": PART_IMAGENET_MIXED,
    "part-imagenet-corrupt": PART_IMAGENET_CORRUPT,
    "part-imagenet-pseudo": PART_IMAGENET_PSEUDO,
    "part-imagenet-pseudo-imagenet-class": PART_IMAGENET_PSEUDO_IMAGENET_CLASS,
    "part-imagenet-imagenet-class": PART_IMAGENET_IMAGENET_CLASS,
    "part-imagenet-bbox": PART_IMAGENET_BBOX,
    "imagenet": IMAGENET,
}


def load_dataset(args):
    loader = DATASET_DICT[args.dataset]["loader"]
    return loader(args)
