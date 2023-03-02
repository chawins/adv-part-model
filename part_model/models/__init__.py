"""Model utility."""

from __future__ import annotations

import logging
import os
from argparse import Namespace

import timm
import torch
import torchvision
from torch import nn
from torch.cuda import amp

from part_model.dataloader import DATASET_DICT
from part_model.models.det_part_models import (
    dino,
    dino_bbox_model,
    multi_head_dino_bbox_model,
)
from part_model.models.model import Classifier, SegClassifier
from part_model.models.seg_part_models import (
    bbox_model,
    clean_mask_model,
    groundtruth_mask_model,
    part_fc_model,
    part_mask_model,
    part_seg_cat_model,
    part_seg_model,
    pixel_count_model,
    pooling_model,
    two_head_model,
    weighted_bbox_model,
)
from part_model.models.seg_part_models.util import SEGM_BUILDER
from part_model.utils.image import get_seg_type

logger = logging.getLogger(__name__)


def wrap_distributed(args, model):
    # When using efficientnet as backbone, pytorch's torchrun complains
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


def load_checkpoint(
    args: Namespace,
    model: nn.Module,
    model_path: str | None = None,
    resume_opt_state: bool = True,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: amp.GradScaler | None = None,
) -> None:
    logger.info("=> Loading resume checkpoint %s...", model_path)
    if args.gpu is None:
        checkpoint = torch.load(model_path)
    else:
        # Map model to be loaded to specified single gpu.
        checkpoint = torch.load(model_path, map_location=f"cuda:{args.gpu}")

    if args.load_from_segmenter:
        logger.info("=> Loading segmenter weight only...")
        state_dict = {}
        for name, params in checkpoint["state_dict"].items():
            name.replace("module", "module.segmenter")
            state_dict[name] = params
        model.load_state_dict(state_dict, strict=False)
    else:
        # Rename state_dict for backward compatibility with old
        # interface (before using Classifier/SegClassifier).
        state_dict = checkpoint["state_dict"]
        new_state_dict = {}
        for name, weights in state_dict.items():
            if "._model." in name:
                new_state_dict[name] = weights
                continue
            new_name = name.replace("module.1.", "module.")
            new_name = new_name.replace("module.", "module._model.")
            new_name = new_name.replace(".segmenter.1.", ".segmenter.")
            new_state_dict[new_name] = weights
        model.load_state_dict(new_state_dict, strict=False)

    if not args.load_weight_only and resume_opt_state:
        args.start_epoch = checkpoint["epoch"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
    logger.info("=> Loaded resume checkpoint (epoch %d)", checkpoint["epoch"])

def build_classifier(args):

    assert args.dataset in DATASET_DICT
    normalize = DATASET_DICT[args.dataset]["normalize"]
    if args.arch == "resnet101":
        # timm does not have pretrained resnet101
        model = torchvision.models.resnet101(
            weights=args.pretrained, progress=True
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

        if args.seg_arch is not None:
            logger.info("=> Building segmentation model...")
            segmenter = SEGM_BUILDER[args.seg_arch](args, normalize=False)
            if args.freeze_seg:
                # Froze all weights of the part segmentation model
                for param in segmenter.parameters():
                    param.requires_grad = False
        elif args.obj_det_arch is not None:
            logger.info("=> Building detection model...")
        
        if args.obj_det_arch == "dino":
            # two options, either sequential or two-headed model
            if model_token == "seq":
                model = dino_bbox_model.DinoBoundingBoxModel(args)
            elif model_token == "2heads":
                model = (
                    multi_head_dino_bbox_model.MultiHeadDinoBoundingBoxModel(
                        args
                    )
                )
            else:
                raise NotImplementedError(
                    f"Invalid model type ({model_token}) for DINO!"
                )

            for param in model.parameters():
                param.requires_grad = True
        elif model_token == "mask":
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
            model = part_mask_model.PartMaskModel(args, segmenter, model)
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
            model = part_seg_cat_model.PartSegCatModel(
                args, segmenter, model, rep_dim
            )
        elif model_token == "seg":
            model = part_seg_model.PartSegModel(
                args, segmenter, model, rep_dim, topk=None
            )
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
            model = clean_mask_model.CleanMaskModel(args, segmenter, model)
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
            model = groundtruth_mask_model.GroundtruthMaskModel(
                args, segmenter, model
            )
        elif model_token == "2heads_d":
            model = two_head_model.TwoHeadModel(args, segmenter, "d")
        elif model_token == "2heads_e":
            model = two_head_model.TwoHeadModel(args, segmenter, "e")
        elif model_token == "pixel":
            model = pixel_count_model.PixelCountModel(args, segmenter, None)
        elif model_token == "bbox_2heads_d":
            model = multi_head_dino_bbox_model.MultiHeadDinoBoundingBoxModel(
                args
            )
        elif model_token == "bbox":
            # two options, either bbox model from object detection or bbox from segmentation model
            if args.obj_det_arch == "dino":
                model = dino_bbox_model.DinoBoundingBoxModel(args)
            else:
                model = bbox_model.BoundingBoxModel(args, segmenter)
        elif model_token == "wbbox":
            model = weighted_bbox_model.WeightedBBoxModel(args, segmenter)
        elif model_token == "fc":
            model = part_fc_model.PartFCModel(args, segmenter)
        elif model_token == "pooling":
            model = pooling_model.PoolingModel(args, segmenter)

        model = SegClassifier(model, normalize=normalize)
        n_seg = sum(p.numel() for p in model.parameters()) / 1e6
        nt_seg = (
            sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        )
        logger.info("=> Model params (train/total): %.2fM/%.2fM", nt_seg, n_seg)
    else:
        logger.info("=> Building a normal classifier...")
        model.fc = nn.Linear(rep_dim, args.num_classes)
        model = Classifier(model, normalize=normalize)
        n_model = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info("=> Total params: %.2fM", n_model)

    # Wrap model again under DistributedDataParallel or just DataParallel
    model = wrap_distributed(args, model)

    if args.obj_det_arch == "dino":
        backbone_params, non_backbone_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(param)
            else:
                non_backbone_params.append(param)
        optim_params = [
            {"params": non_backbone_params},
            {"params": backbone_params, "lr": args.lr_backbone},
        ]
    else:
        p_wd, p_non_wd = [], []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if any(kword in name for kword in ["bias", "ln", "bn"]):
                    p_non_wd.append(param)
                else:
                    p_wd.append(param)
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
    if not (args.evaluate or args.resume or args.resume_if_exist):
        logger.info("=> Model is randomly initialized.")
        return model, optimizer, scaler

    if args.evaluate:
        model_path = f"{args.output_dir}/checkpoint_best.pt"
        resume_opt_state = False
    else:
        # Try to resume training from the last checkpoint
        model_path = f"{args.output_dir}/checkpoint_last.pt"
        resume_opt_state = True
        if not args.resume_if_exist or not os.path.isfile(model_path):
            # If resume_if_exist is False, or the last checkpoint does not
            # exist, use user-specified checkpoint
            model_path = args.resume
            resume_opt_state = False

    if os.path.isfile(model_path):
        load_checkpoint(
            args,
            model,
            model_path=model_path,
            resume_opt_state=resume_opt_state,
            optimizer=optimizer,
            scaler=scaler,
        )
    elif args.resume:
        raise FileNotFoundError(f"=> No checkpoint found at {model_path}.")
    else:
        logger.info(
            "=> resume_if_exist is True, but no checkpoint found at %s",
            model_path,
        )

    return model, optimizer, scaler


def build_segmentation(args):
    model = SEGM_BUILDER[args.seg_arch](args)
    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
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
    # if args.resume and not args.evaluate:
    if args.resume or args.evaluate:
        if os.path.isfile(args.resume):
            logger.info("=> loading resume checkpoint %s...", args.resume)
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
            logger.info(
                "=> loaded resume checkpoint (epoch %d)", checkpoint["epoch"]
            )
        else:
            logger.info("=> no checkpoint found at %s", args.resume)
    
    return model, optimizer, scaler


def build_detector(args):
    model = dino.DinoModel(args)
    model = wrap_distributed(args, model)

    n_det = sum(p.numel() for p in model.parameters()) / 1e6
    nt_det = (
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    )
    print("Model params (train/total): %.2fM/%.2fM", nt_det, n_det)
    
    backbone_params, non_backbone_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            non_backbone_params.append(param)
    optim_params = [
        {"params": non_backbone_params},
        {"params": backbone_params, "lr": args.lr_backbone},
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

    if args.resume or args.evaluate:
        if os.path.isfile(args.resume):
            logger.info("=> loading resume checkpoint %s...", args.resume)
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
            logger.info(
                "=> loaded resume checkpoint (epoch %d)", checkpoint["epoch"]
            )
        else:
            logger.info("=> no checkpoint found at %s", args.resume)

    return model, optimizer, scaler

def build_model(args):
    if "seg-only" in args.experiment:
        if args.obj_det_arch == 'dino':
            return build_detector(args)
        else:
            return build_segmentation(args)
    return build_classifier(args)
