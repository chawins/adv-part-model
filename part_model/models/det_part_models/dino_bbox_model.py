"""DINO as sequential part model."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from DINO.models.dino.backbone import Backbone, Joiner, build_position_encoding
from DINO.models.dino.dino import DINO, build_deformable_transformer
from DINO.util.misc import NestedTensor
from part_model.utils.types import BatchImages, Logits

logger = logging.getLogger(__name__)


class FeatureExtractor(nn.Module):
    """Feature extractor for DINO."""

    def __init__(self, args, hidden_dim: int = 16) -> None:
        """Initialize FeatureExtractor."""
        super().__init__()
        feature_dim: int = args.seg_labels + 4
        self.layer, self.pooling = None, None
        self.output_dim = feature_dim * args.num_queries
        # TODO(chawins@): Consider self-attention layer.
        if "conv1d" in args.experiment:
            self.layer = nn.Conv1d(feature_dim, hidden_dim, 1)
            self.output_dim = hidden_dim * args.num_queries
        if "pool" in args.experiment:
            self.pooling = nn.AdaptiveMaxPool1d(1)
            self.output_dim = hidden_dim

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        """Forward pass.

        Args:
            inputs: [batch_size, num_queries, feature_dim]

        Returns:
            Outputs of vary shapes depending on the layers.
        """
        # x.shape: [batch_size, num_queries, feature_dim]
        if self.layer is not None:
            inputs = inputs.permute(0, 2, 1)
            # Output shape: [batch_size, hidden_dim, num_queries]
            inputs = self.layer(inputs)
        if self.pooling is not None:
            if self.layer is None:
                inputs = inputs.permute(0, 2, 1)
            # Output shape: [batch_size, hidden_dim, 1]
            inputs = self.pooling(inputs)
        return inputs


class DinoBoundingBoxModel(nn.Module):
    """DINO as part model."""

    def __init__(self, args):
        """Initialize DinoBoundingBoxModel."""
        logger.info("=> Initializing DinoBoundingBoxModel...")
        super().__init__()
        self._use_conv1d = "conv1d" in args.experiment
        self._sort_dino_outputs = "sort_dino_outputs" in args.experiment
        self._num_queries = args.num_queries

        # Temporary model options
        self._use_sigmoid = "sig" in args.experiment

        backbone = _build_backbone(args)
        transformer = build_deformable_transformer(args)

        # TODO(nab-126@): Remove the try/except block. Try to never use
        # catch-all except.
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

        hidden_dim1: int = 16
        hidden_dim2: int = 64
        first_layer = FeatureExtractor(args, hidden_dim1)

        # Input to core model is [batch_size, num_queries, num_classes + 4]
        self.core_model = nn.Sequential(
            first_layer,
            nn.Flatten(),
            nn.BatchNorm1d(first_layer.output_dim),
            nn.Linear(first_layer.output_dim, hidden_dim2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim2),
            nn.Linear(hidden_dim2, args.num_classes),
        )

    def forward(
        self,
        images: BatchImages,
        masks=None,
        dino_targets=None,
        need_tgt_for_training: bool = False,
        return_mask: bool = False,
    ) -> Logits | tuple[Logits, torch.Tensor]:
        """Forward pass of sequential DINO part model."""
        # Object Detection part
        nested_tensors = NestedTensor(images, masks)

        if need_tgt_for_training:
            dino_outputs = self.object_detector(nested_tensors, dino_targets)
        else:
            dino_outputs = self.object_detector(nested_tensors)

        if self._use_sigmoid:
            # We can consider not taking softmax if we have trouble
            # with attack during adversarial training not working well, or we can
            # use sigmoid to better replicate object detection models.
            dino_probs = F.sigmoid(dino_outputs["pred_logits"])
        else:
            dino_probs = F.softmax(dino_outputs["pred_logits"], dim=-1)
        dino_boxes = dino_outputs["pred_boxes"]

        if self._sort_dino_outputs:
            # TODO(nabeel@): Don't leave unused variables (use "_") and
            # remove commneted out line in code.
            batch_size, _, num_classes = dino_probs.shape
            _, topk_indexes = torch.topk(
                dino_probs.view(batch_size, -1), self._num_queries, dim=1
            )
            topk_boxes = topk_indexes // num_classes
            topk_boxes.unsqueeze_(-1)
            dino_probs = torch.gather(
                dino_probs, 1, topk_boxes.repeat(1, 1, num_classes)
            )
            dino_boxes = torch.gather(dino_boxes, 1, topk_boxes.repeat(1, 1, 4))

        features = torch.cat([dino_probs, dino_boxes], dim=2)

        # Pass to classifier model
        out = self.core_model(features)

        if return_mask:
            return out, dino_outputs
        return out


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
