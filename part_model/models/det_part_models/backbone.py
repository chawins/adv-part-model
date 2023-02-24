"""Utility functions for DINO part models."""

import logging

import torch
import torchvision

from DINO.models.dino.backbone import (
    BackboneBase,
    Joiner,
    build_position_encoding,
)

logger = logging.getLogger(__name__)


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        dilation: bool,
        return_interm_indices: list,
        batch_norm=torchvision.ops.FrozenBatchNorm2d,
    ) -> None:
        """Initialize backbone.

        Args:
            name: Name of backbone architecture.
            train_backbone: Whether backbone will be trained.
            dilation: Dilation.
            return_interm_indices: Indices of intermediate layers to return.
            batch_norm: Batch normalization layer. Defaults to
                torchvision.ops.FrozenBatchNorm2d.

        Raises:
            NotImplementedError: Invalid backbone name.
        """
        if name in ["resnet50", "resnet101"]:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                norm_layer=batch_norm,
                weights="IMAGENET1K_V1",
            )
        else:
            raise NotImplementedError(
                f"Only resnet50 and resnet101 are available ({name} given)."
            )
        assert return_interm_indices in [[0, 1, 2, 3], [1, 2, 3], [3]]
        num_channels_all = [256, 512, 1024, 2048]
        num_channels = num_channels_all[4 - len(return_interm_indices) :]
        super().__init__(
            backbone, train_backbone, num_channels, return_interm_indices
        )


def build_backbone(args):
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
