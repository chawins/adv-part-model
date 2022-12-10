from __future__ import annotations

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


class DinoBoundingBoxModel(nn.Module):
    def __init__(self, args):
        print("=> Initializing DinoBoundingBoxModel...")
        super().__init__()

        self.use_conv1d = "conv1d" in args.experiment

        # TODO: load weights if args.load_from_segmenter
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

        input_dim = (
            args.num_queries * (args.seg_labels + 4)
            if not self.use_conv1d
            else 10 * (args.seg_labels)
        )

        self.core_model = nn.Sequential(
            nn.Conv1d(args.num_queries, 10, 5) if self.use_conv1d else nn.Identity(),
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
        masks,
        dino_targets,
        need_tgt_for_training,
        return_mask=False,
        **kwargs,
    ):
        # Object Detection part
        nested_tensors = NestedTensor(images, masks)

        if need_tgt_for_training:
            dino_outputs = self.object_detector(nested_tensors, dino_targets)
        else:
            dino_outputs = self.object_detector(nested_tensors)

        # concatenate softmax'd logits and bounding box predictions
        features = torch.cat(
            [
                F.softmax(dino_outputs["pred_logits"], dim=1),
                dino_outputs["pred_boxes"],
            ],
            dim=2,
        )

        out = self.core_model(features)

        if return_mask:
            return out, dino_outputs
            # return out, outputs['pred_logits'], outputs['pred_boxes']
        return out
