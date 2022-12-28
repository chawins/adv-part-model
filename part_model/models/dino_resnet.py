from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from DINO.models.dino.dino import (
    DINO,
    build_backbone,
    build_deformable_transformer,
)
from DINO.util.misc import NestedTensor


class ResNet(nn.Module):
    def __init__(self, args):
        print("=> Initializing Dino Resnet Model...")
        super().__init__()

        self.backbone = build_backbone(args)
        self.classifier_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            # nn.Flatten(),
            nn.Flatten(start_dim=-3, end_dim=-1),
            nn.Linear(
                in_features=2048, out_features=args.num_classes, bias=True
            ),
        )

    def forward(
        self,
        images,
        masks,
        dino_targets,
        need_tgt_for_training,
        return_mask=False,
        **kwargs,
    )-> torch.Tensor:
        _ = kwargs  # Unused
        nested_tensors = NestedTensor(images, masks)
        out = self.backbone(nested_tensors)

        out_classifier_head = self.classifier_head(out[0][-1].tensors)

        if return_mask:
            return out_classifier_head, None
        return out_classifier_head
