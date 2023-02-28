"""DINO as sequential part model."""

from __future__ import annotations

import logging

import torch
from torch import nn
import torchvision

from DINO.models.dino.dino import (
    DINO,
    build_deformable_transformer,
)
from DINO.models.dino.backbone import Backbone, Joiner, build_position_encoding

from DINO.util.misc import NestedTensor
from part_model.utils.types import BatchImages, Logits

logger = logging.getLogger(__name__)


class DinoModel(nn.Module):
    """DINO as part model."""

    def __init__(self, args):
        """Initialize DinoBoundingBoxModel."""
        logger.info("=> Initializing DinoBoundingBoxModel...")
        super().__init__()

        backbone = _build_backbone(args)
        transformer = build_deformable_transformer(args)

        try:
            match_unstable_error = args.match_unstable_error
            dn_labelbook_size = args.dn_labelbook_size
        except:
            match_unstable_error = True
            dn_labelbook_size = args.seg_labels

        try:
            dec_pred_class_embed_share = args.dec_pred_class_embed_share
        except:
            dec_pred_class_embed_share = True
        try:
            dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
        except:
            dec_pred_bbox_embed_share = True

        self.object_detector = DINO(
            backbone,
            transformer,
            num_classes=args.seg_labels,
            num_queries=args.num_queries,
            aux_loss=True,
            iter_update=True,
            query_dim=4,
            random_refpoints_xy=args.random_refpoints_xy,
            fix_refpoints_hw=args.fix_refpoints_hw,
            num_feature_levels=args.num_feature_levels,
            nheads=args.nheads,
            dec_pred_class_embed_share=dec_pred_class_embed_share,
            dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
            # two stage
            two_stage_type=args.two_stage_type,
            # box_share
            two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
            two_stage_class_embed_share=args.two_stage_class_embed_share,
            decoder_sa_type=args.decoder_sa_type,
            num_patterns=args.num_patterns,
            dn_number=args.dn_number if args.use_dn else 0,
            dn_box_noise_scale=args.dn_box_noise_scale,
            dn_label_noise_ratio=args.dn_label_noise_ratio,
            dn_labelbook_size=dn_labelbook_size,
        )

    def forward(
        self,
        images: BatchImages,
        masks=None,
        dino_targets=None,
        need_tgt_for_training: bool = False,
        return_mask: bool = False,
        return_mask_only: bool = False,
    ) -> Logits | tuple[Logits, torch.Tensor]:
        """Forward pass of sequential DINO part model."""
        # Object Detection part
        nested_tensors = NestedTensor(images, masks)

        if need_tgt_for_training:
            dino_outputs = self.object_detector(nested_tensors, dino_targets)
        else:
            dino_outputs = self.object_detector(nested_tensors)
        return dino_outputs
        


def _build_backbone(args):
    """Build backbone for DINO. Modified from DINO/models/dino/backbone.py.

    Useful args:
        - backbone: backbone name
        - lr_backbone:
        - dilation
        - return_interm_indices: available: [0,1,2,3], [1,2,3], [3]
        - backbone_freeze_keywords:
        - use_checkpoint: for swin only for now
    """
    position_embedding = build_position_encoding(args)
    # This can be removed
    train_backbone = args.lr_backbone > 0
    if not train_backbone:
        raise ValueError("Please set lr_backbone > 0")
    return_interm_indices = args.return_interm_indices
    assert return_interm_indices in [[0, 1, 2, 3], [1, 2, 3], [3]]

    if args.batch_norm_type == "FrozenBatchNorm2d":
        batch_norm_layer = torchvision.ops.FrozenBatchNorm2d
    else:
        batch_norm_layer = getattr(torch.nn, args.batch_norm_type)

    if args.backbone in ["resnet50", "resnet101"]:
        backbone = Backbone(
            args.backbone,
            train_backbone,
            args.dilation,
            return_interm_indices,
            batch_norm=batch_norm_layer,
        )
        bb_num_channels = backbone.num_channels
    else:
        raise NotImplementedError(
            "Only resnet50 and resnet101 are supported for now!"
        )
    assert len(bb_num_channels) == len(return_interm_indices), (
        f"len(bb_num_channels) {len(bb_num_channels)} != "
        f"len(return_interm_indices) {len(return_interm_indices)}"
    )

    model = Joiner(backbone, position_embedding)
    model.num_channels = bb_num_channels
    assert isinstance(
        bb_num_channels, list
    ), f"bb_num_channels is expected to be a List but {type(bb_num_channels)}!"
    return model
