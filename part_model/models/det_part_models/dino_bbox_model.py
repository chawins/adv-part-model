"""DINO as sequential part model."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn

from DINO.models.dino.dino import (
    DINO,
    build_backbone,
    build_deformable_transformer,
)
from DINO.util.misc import NestedTensor
from part_model.utils.types import BatchImages, Logits

logger = logging.getLogger(__name__)


class DinoBoundingBoxModel(nn.Module):
    """DINO as part model."""

    def __init__(self, args):
        """Initialize DinoBoundingBoxModel."""
        logger.info("=> Initializing DinoBoundingBoxModel...")
        super().__init__()
        self._use_conv1d = "conv1d" in args.experiment
        self._sort_dino_outputs = "sort_dino_outputs" in args.experiment
        self._num_queries = args.num_queries

        backbone = build_backbone(args)
        transformer = build_deformable_transformer(args)

        try:
            match_unstable_error = args.match_unstable_error
            dn_labelbook_size = args.dn_labelbook_size
        except:
            match_unstable_error = True
            # dn_labelbook_size = num_classes
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

        input_dim = (
            args.num_queries * (args.seg_labels + 4)
            if not self._use_conv1d
            else 10 * (args.seg_labels)
        )

        # how did we get 50 here
        self.core_model = nn.Sequential(
            nn.Conv1d(args.num_queries, 10, 5)
            if self._use_conv1d
            else nn.Identity(),
            nn.Flatten(),
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, args.num_classes),
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

        if return_mask_only:
            return dino_outputs
        
        # TODO(chawins@): We can consider not taking softmax if we have trouble
        # with attack during adversarial training not working well, or we can
        # use sigmoid to better replicate object detection models.
        dino_probs = F.softmax(dino_outputs["pred_logits"], dim=-1)
        dino_boxes = dino_outputs["pred_boxes"]

        if self._sort_dino_outputs:
            # TODO(nabeel@): Don't leave unused variables (use "_") and
            # remove commneted out line in code.
            batch_size, _, num_classes = dino_probs.shape
            topk_values, topk_indexes = torch.topk(
                dino_probs.view(batch_size, -1), self._num_queries, dim=1
            )
            topk_boxes = topk_indexes // num_classes
            # labels = top_indexes % out_logits.shape[2]
            topk_boxes.unsqueeze_(-1)
            dino_probs = torch.gather(
                dino_probs, 1, topk_boxes.repeat(1, 1, num_classes)
            )
            dino_boxes = torch.gather(dino_boxes, 1, topk_boxes.repeat(1, 1, 4))

        # features = torch.cat([dino_probs, dino_boxes * 2 - 1], dim=2)
        features = torch.cat([dino_probs, dino_boxes], dim=2)

        # Pass to classifier model
        out = self.core_model(features)

        if return_mask:
            return out, dino_outputs
        return out
