from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from DINO.main import build_model_main
from DINO.models.dino.dino import (
    DINO,
    build_backbone,
    build_deformable_transformer,
)
from DINO.util.misc import NestedTensor


class MultiHeadDinoBoundingBoxModel(nn.Module):
    def __init__(self, args):
        print("=> Initializing DinoBoundingBoxModel...")
        super().__init__()

        self.backbone = build_backbone(args)

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

        self.object_detector_head = DINO(
            self.backbone,
            transformer,
            num_classes=args.seg_labels,
            # num_classes=num_classes,
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

        # TODO: don't hardcode
        self.classifier_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(
                in_features=2048, out_features=args.num_classes, bias=True
            ),
        )

        self.num_classes = args.num_classes

    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        dino_targets: dict[str, Any],
        need_tgt_for_training: bool = True,
        return_mask: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        _ = kwargs  # Unused
        nested_tensors = NestedTensor(images, masks)
        out = self.backbone(nested_tensors)

        if need_tgt_for_training:
            out_object_detector_head = self.object_detector_head(
                nested_tensors, dino_targets
            )
        else:
            out_object_detector_head = self.object_detector_head(nested_tensors)

        out_classifier_head = self.classifier_head(out[0][-1].tensors)

        if return_mask:
            return out_classifier_head, out_object_detector_head
        return out_classifier_head
