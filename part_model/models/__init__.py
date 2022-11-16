"""Model utility."""

import os

import timm
import torch
import torchvision
from torch import nn
from torch.cuda import amp

from part_model.dataloader import DATASET_DICT
from part_model.models.bbox_model import BoundingBoxModel
from part_model.models.clean_mask_model import CleanMaskModel
from part_model.models.common import Normalize
from part_model.models.groundtruth_mask_model import GroundtruthMaskModel
from part_model.models.part_fc_model import PartFCModel
from part_model.models.part_mask_model import PartMaskModel
from part_model.models.part_seg_cat_model import PartSegCatModel
from part_model.models.part_seg_model import PartSegModel
from part_model.models.pixel_count_model import PixelCountModel
from part_model.models.pooling_model import PoolingModel
from part_model.models.segmentation_model import SEGM_BUILDER
from part_model.models.two_head_model import TwoHeadModel
from part_model.models.weighted_bbox_model import WeightedBBoxModel
from part_model.utils.image import get_seg_type


def wrap_distributed(args, model):
    # TODO: When using efficientnet as backbone, pytorch's torchrun complains
    # about unused parameters. This can be suppressed by setting
    # find_unused_parameters to True.
    find_unused_parameters: bool = any(
        "efficientnet" in arch for arch in (args.seg_backbone, args.arch)
    )

    if args.distributed:
        model.cuda(args.gpu)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=find_unused_parameters,
        )
    else:
        model.cuda()
        model = torch.nn.parallel.DataParallel(model)
    return model


def build_classifier(args):

    assert args.dataset in DATASET_DICT

    normalize = DATASET_DICT[args.dataset]["normalize"]
    if args.arch == "resnet101":
        # timm does not have pretrained resnet101
        model = torchvision.models.resnet101(
            pretrained=args.pretrained, progress=True
        )
        rep_dim = 2048
    else:
        model = timm.create_model(
            args.arch, pretrained=args.pretrained, num_classes=0
        )
        with torch.no_grad():
            dummy_input = torch.zeros(
                (2,) + DATASET_DICT[args.dataset]["input_dim"]
            )
            rep_dim = model(dummy_input).size(-1)

    if get_seg_type(args) is not None:
        tokens = args.experiment.split("-")
        model_token = tokens[1]
        exp_tokens = tokens[2:]
        print("=> building segmentation model...")
        segmenter = SEGM_BUILDER[args.seg_arch](args)

        if args.freeze_seg:
            # Froze all weights of the part segmentation model
            for param in segmenter.parameters():
                param.requires_grad = False

        if model_token == "mask":
            model.conv1 = nn.Conv2d(
                args.seg_labels
                + (3 if "inpt" in exp_tokens else 0)
                - (1 if "nobg" in exp_tokens else 0),
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            model.fc = nn.Linear(rep_dim, args.num_classes)
            model = PartMaskModel(args, segmenter, model)
        elif model_token == "seg_cat":
            model.conv1 = nn.Conv2d(
                (args.seg_labels - 1) * 3
                if "nobg" in exp_tokens
                else args.seg_labels * 3,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            model.fc = nn.Linear(rep_dim, args.num_classes)
            model = PartSegCatModel(args, segmenter, model, rep_dim)
        elif model_token == "seg":
            model = PartSegModel(args, segmenter, model, rep_dim, topk=None)
        elif model_token == "clean":
            model.conv1 = nn.Conv2d(
                args.seg_labels
                + (3 if "inpt" in exp_tokens else 0)
                - (1 if "nobg" in exp_tokens else 0),
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            model.fc = nn.Linear(rep_dim, args.num_classes)
            model = CleanMaskModel(args, segmenter, model)
        elif model_token == "groundtruth":
            model.conv1 = nn.Conv2d(
                args.seg_labels
                + (3 if "inpt" in exp_tokens else 0)
                - (1 if "nobg" in exp_tokens else 0),
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            model.fc = nn.Linear(rep_dim, args.num_classes)
            model = GroundtruthMaskModel(args, segmenter, model)
        elif model_token == "2heads_d":
            model = TwoHeadModel(args, segmenter, "d")
        elif model_token == "2heads_e":
            model = TwoHeadModel(args, segmenter, "e")
        elif model_token == "pixel":
            model = PixelCountModel(args, segmenter, None)
        elif model_token == "bbox":
            model = BoundingBoxModel(args, segmenter)
        elif model_token == "wbbox":
            model = WeightedBBoxModel(args, segmenter)
        elif model_token == "fc":
            model = PartFCModel(args, segmenter)
        elif model_token == "pooling":
            model = PoolingModel(args, segmenter)

        n_seg = sum(p.numel() for p in segmenter.parameters()) / 1e6
        nt_seg = (
            sum(p.numel() for p in segmenter.parameters() if p.requires_grad)
            / 1e6
        )
        print(f"=> segmenter params (train/total): {nt_seg:.2f}M/{n_seg:.2f}M")
    else:
        print("=> building a normal classifier (no segmentation)")
        model.fc = nn.Linear(rep_dim, args.num_classes)
        model = nn.Sequential(Normalize(**normalize), model)

    n_model = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"=> total params: {n_model:.2f}M")

    model = wrap_distributed(args, model)

    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if "bias" in n or "ln" in n or "bn" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)

    optim_params = [
        {"params": p_wd, "weight_decay": args.wd},
        {"params": p_non_wd, "weight_decay": 0},
    ]

    if args.optim == "sgd":
        optimizer = torch.optim.SGD(
            optim_params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
        )
    else:
        optimizer = torch.optim.AdamW(
            optim_params,
            lr=args.lr,
            betas=args.betas,
            eps=args.eps,
            weight_decay=args.wd,
        )

    scaler = amp.GradScaler(enabled=not args.full_precision)

    # Optionally resume from a checkpoint
    if args.resume and not args.evaluate or args.resume_if_exist:

        model_path = f"{args.output_dir}/checkpoint_last.pt"
        resume_exists = True
        if not args.resume_if_exist or not os.path.isfile(model_path):
            model_path = args.resume
            resume_exists = False

        if os.path.isfile(model_path):
            print(f"=> loading resume checkpoint {model_path}...")
            if args.gpu is None:
                checkpoint = torch.load(model_path)
            else:
                # Map model to be loaded to specified single gpu.
                loc = f"cuda:{args.gpu}"
                checkpoint = torch.load(model_path, map_location=loc)

            if args.load_from_segmenter:
                print("=> Loading segmenter weight only...")
                state_dict = {}
                for name, params in checkpoint["state_dict"].items():
                    name.replace("module", "module.segmenter")
                    state_dict[name] = params
                model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(checkpoint["state_dict"], strict=True)

            if not args.load_weight_only or resume_exists:
                args.start_epoch = checkpoint["epoch"]
                optimizer.load_state_dict(checkpoint["optimizer"])
                scaler.load_state_dict(checkpoint["scaler"])
            print(f'=> loaded resume checkpoint (epoch {checkpoint["epoch"]})')
        elif args.resume:
            raise FileNotFoundError(f"=> No checkpoint found at {model_path}.")
        else:
            print("=> Tried to resume if exist but found no checkpoint.")
    else:
        print("=> Loaded model without using resumed weights.")

    return model, optimizer, scaler


def build_segmentation(args):
    model = SEGM_BUILDER[args.seg_arch](args)
    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # model_without_ddp = model[1]
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model, device_ids=[args.gpu]
    #     )
    #     model_without_ddp = model.module[1]
    model = wrap_distributed(args, model)
    model_without_ddp = model.module[1]

    backbone_params = list(model_without_ddp.encoder.parameters())
    last_params = list(model_without_ddp.decoder.parameters())
    last_params.extend(list(model_without_ddp.segmentation_head.parameters()))
    optimizer = torch.optim.SGD(
        [
            {"params": filter(lambda p: p.requires_grad, backbone_params)},
            {"params": filter(lambda p: p.requires_grad, last_params)},
        ],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
    )

    scaler = amp.GradScaler(enabled=not args.full_precision)

    # Optionally resume from a checkpoint
    if args.resume and not args.evaluate:
        if os.path.isfile(args.resume):
            print(f"=> loading resume checkpoint {args.resume}...")
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = f"cuda:{args.gpu}"
                checkpoint = torch.load(args.resume, map_location=loc)
            model.load_state_dict(checkpoint["state_dict"])

            if not args.load_weight_only:
                args.start_epoch = checkpoint["epoch"]
                optimizer.load_state_dict(checkpoint["optimizer"])
                scaler.load_state_dict(checkpoint["scaler"])
            print(f'=> loaded resume checkpoint (epoch {checkpoint["epoch"]})')
        else:
            print(f"=> no checkpoint found at {args.resume}")

    return model, optimizer, scaler


def build_model(args):
    if "seg-only" in args.experiment:
        return build_segmentation(args)
    return build_classifier(args)
