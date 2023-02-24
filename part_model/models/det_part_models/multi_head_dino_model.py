"""Multi-head DINO model for bounding box detection and classification."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from DINO.models.dino.dino import DINO, build_deformable_transformer
from DINO.util.misc import NestedTensor
from part_model.models.det_part_models.seq_dino_model import build_backbone
from part_model.utils.types import BatchImages, Logits

logger = logging.getLogger(__name__)


class MultiHeadDinoModel(nn.Module):
    """Multi-head DINO model for bounding box detection and classification."""

    def __init__(self, args) -> None:
        """Initialize MultiHeadDinoModel.

        Args:
            args: Arguments.
        """
        logger.info("=> Initializing MultiHeadDinoModel...")
        super().__init__()

        self.backbone = build_backbone(args)
        transformer = build_deformable_transformer(args)
        try:
            dn_labelbook_size = args.dn_labelbook_size
        except AttributeError:
            dn_labelbook_size = args.seg_labels
        dec_pred_class_embed_share = True
        dec_pred_bbox_embed_share = True

        self.object_detector_head = DINO(
            self.backbone,
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

        # Output of backbone at the last feature level is of shape
        # [batch_size, 2048, 7, 7] for ResNet-50.
        # TODO(nab-126@): Find a way to not hardcode in_features.
        self.classifier_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=args.num_classes),
        )

    def forward(
        self,
        images: BatchImages,
        masks: torch.Tensor | None = None,
        dino_targets: dict[str, Any] | None = None,
        need_tgt_for_training: bool = False,
        return_mask: bool = False,
        return_mask_only: bool = False,
        **kwargs,
    ) -> Logits | tuple[Logits, torch.Tensor]:
        """Forward pass."""
        _ = kwargs  # Unused
        nested_tensors = NestedTensor(images, masks)

        # out[0] is output of backbone, out[1] is output of position encoder.
        out = self.backbone(nested_tensors)

        if need_tgt_for_training:
            out_obj_det_head = self.object_detector_head(
                nested_tensors, dino_targets
            )
        else:
            out_obj_det_head = self.object_detector_head(nested_tensors)

        # out[0][i] is output of i-th feature level which is a nested tensor
        # consisting of tensors and mask.
        out_clf_head = self.classifier_head(out[0][-1].tensors)

        if return_mask_only:
            return out_obj_det_head
        if return_mask:
            return out_clf_head, out_obj_det_head
        return out_clf_head
