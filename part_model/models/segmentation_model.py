from collections import OrderedDict

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from ..dataloader import DATASET_DICT
from .common import Normalize


def build_deeplabv3(args):
    # FIXME: DeepLabV3 is pretrained on COCO (not ImageNet)
    model = torch.hub.load(
        "pytorch/vision:v0.10.0",
        "deeplabv3_resnet50",
        pretrained=args.pretrained,
    )
    model.classifier = DeepLabHead(2048, args.seg_labels)
    model.aux_classifier = None

    normalize = DATASET_DICT[args.dataset]["normalize"]
    model = nn.Sequential(Normalize(**normalize), model)

    if args.seg_dir != "":
        best_path = f"{args.seg_dir}/checkpoint_best.pt"
        print(f"=> loading best checkpoint for DeepLabv3: {best_path}")
        if args.gpu is None:
            checkpoint = torch.load(best_path)
        else:
            # Map model to be loaded to specified single gpu.
            loc = "cuda:{}".format(args.gpu)
            checkpoint = torch.load(best_path, map_location=loc)

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            name = k[7:]  # remove `module`
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)

    return model


def build_deeplabv3plus(args):

    model = smp.DeepLabV3Plus(
        encoder_name=args.seg_backbone,
        encoder_weights="imagenet" if args.pretrained else None,
        in_channels=3,
        classes=args.seg_labels,
        # Default parameters
        encoder_depth=5,
        encoder_output_stride=16,
        decoder_channels=256,
        decoder_atrous_rates=(12, 24, 36),
        upsampling=4,
        aux_params=None,
    )
    normalize = DATASET_DICT[args.dataset]["normalize"]
    model = nn.Sequential(Normalize(**normalize), model)

    if args.seg_dir != "":
        best_path = f"{args.seg_dir}/checkpoint_best.pt"
        print(f"=> loading best checkpoint for DeepLabv3+: {best_path}")
        if args.gpu is None:
            checkpoint = torch.load(best_path)
        else:
            # Map model to be loaded to specified single gpu.
            loc = "cuda:{}".format(args.gpu)
            checkpoint = torch.load(best_path, map_location=loc)

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            name = k[7:]  # remove `module`
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)

    return model


SEGM_BUILDER = {
    "deeplabv3": build_deeplabv3,
    "deeplabv3plus": build_deeplabv3plus,
}
