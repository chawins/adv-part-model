"""Implementation of DINO as part model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from DINO.models.dino.dino import (
    DINO,
    build_backbone,
    build_deformable_transformer,
)
from DINO.util.misc import NestedTensor


class DinoBoundingBoxModel(nn.Module):
    def __init__(self, args):
        print("=> Initializing DinoBoundingBoxModel...")
        super().__init__()

        self.use_conv1d = "conv1d" in args.experiment
        self.sort_dino_outputs = "sort_dino_outputs" in args.experiment
        self.num_queries = args.num_queries

        backbone = build_backbone(args)

        transformer = build_deformable_transformer(args)

        dn_labelbook_size = args.seg_labels + 1

        # TODO(nabeel@): Why is this try-except needed? Catch all like this is
        # not recommended.
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
            if not self.use_conv1d
            else 10 * (args.seg_labels)
        )

        self.core_model = nn.Sequential(
            nn.Conv1d(args.num_queries, 10, 5)
            if self.use_conv1d
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
        images,
        **kwargs,
    ):
        masks = kwargs["masks"]
        dino_targets = kwargs["dino_targets"]
        need_tgt_for_training = kwargs["need_tgt_for_training"]
        return_mask = kwargs["return_mask"]
        # Object Detection part
        nested_tensors = NestedTensor(images, masks)

        if need_tgt_for_training:
            dino_outputs = self.object_detector(nested_tensors, dino_targets)
        else:
            dino_outputs = self.object_detector(nested_tensors)

        dino_probs = F.softmax(dino_outputs["pred_logits"], dim=-1)
        dino_boxes = dino_outputs["pred_boxes"]

        if self.sort_dino_outputs:
            # TODO(nabeel@): Don't leave unused variables and commneted out line
            # in code.
            topk_values, topk_indexes = torch.topk(
                dino_probs.view(dino_probs.shape[0], -1),
                self.num_queries,
                dim=1,
            )
            topk_boxes = topk_indexes // dino_probs.shape[2]
            # labels = top_indexes % out_logits.shape[2]
            dino_probs = torch.gather(
                dino_probs,
                1,
                topk_boxes.unsqueeze(-1).repeat(1, 1, dino_probs.shape[2]),
            )
            dino_boxes = torch.gather(
                dino_boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4)
            )

        features = torch.cat(
            [dino_probs, dino_boxes],
            dim=2,
        )

        out = self.core_model(features)

        if return_mask:
            return out, dino_outputs
        return out
