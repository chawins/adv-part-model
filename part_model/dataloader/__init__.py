from .part_imagenet_corrupt import PART_IMAGENET_CORRUPT
from .part_imagenet_mixed_next import PART_IMAGENET_MIXED
from .cityscapes import CITYSCAPES
from .part_imagenet import PART_IMAGENET
from .part_imagenet_imagenet_class import PART_IMAGENET_IMAGENET_CLASS
from .part_imagenet_geirhos import PART_IMAGENET_GEIRHOS
from .part_imagenet_pseudo import PART_IMAGENET_PSEUDO
from .pascal_part import PASCAL_PART
from .pascal_voc import PASCAL_VOC
from .util import COLORMAP
from .part_imagenet_pseudo_imagenet_class import (
    PART_IMAGENET_PSEUDO_IMAGENET_CLASS,
)


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
}


def load_dataset(args):
    loader = DATASET_DICT[args.dataset]["loader"]
    return loader(args)
