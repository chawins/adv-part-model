"""Multi-head architecture."""

from __future__ import annotations

import torch
from torch import nn


class TwoHeadModel(nn.Module):
    def __init__(self, args, segmenter, mode):
        super().__init__()
        self.mode = mode
        if self.mode == "d":
            segmenter.segmentation_head = Heads(
                segmenter, args.num_classes
            )
        else:
            latent_dim = 2048  # TODO: depends on backbone
            pool_size = 4
            fc_dim = 64
            segmenter.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.BatchNorm2d(latent_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(latent_dim, fc_dim, (pool_size, pool_size)),
                nn.BatchNorm2d(fc_dim),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(fc_dim, args.num_classes),
            )
        self._segmenter = segmenter

    def forward(
        self, images: torch.Tensor, return_mask: bool = False, **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        _ = kwargs  # Unused
        # Segmentation part
        out = self._segmenter(images)
        if return_mask:
            return out
        return out[0]


class Heads(nn.Module):
    def __init__(self, segmentor, num_classes):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(256, 50, (3, 3), (1, 1)),
                    nn.BatchNorm2d(50),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(50, 10, (1, 1), (1, 1)),
                    nn.BatchNorm2d(10),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Flatten(),
                    nn.Linear(1690, 100),
                    nn.ReLU(),
                    nn.Linear(100, num_classes),
                ),
                segmentor.segmentation_head,
            ]
        )

    def forward(
        self, images: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return [head(images) for head in self.heads]
