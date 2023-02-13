"""Downsampled (pooling) part model."""

from __future__ import annotations

from argparse import Namespace

import torch
import torch.nn.functional as F
from torch import nn


class PoolingFeatureExtractor(nn.Module):
    """Feature extraction layer for Downsampled part model."""

    def __init__(self, no_bg: bool) -> None:
        """Initialize PoolingFeatureExtractor.

        Args:
            no_bg: If True, background channel of the mask is dropped.
        """
        super().__init__()
        self._no_bg: bool = no_bg

    def forward(
        self, logits_masks: torch.Tensor, from_logits: bool = True
    ) -> torch.Tensor:
        """Extract features.

        Args:
            logits_masks: Predicted masks to extract features from.
            from_logits: If True, expect logits_masks to be logits. Otherwise,
                expect softmax/probability mask.

        Returns:
            Extracted features.
        """
        # masks: [B, num_segs (including background), H, W]
        if from_logits:
            masks = F.softmax(logits_masks, dim=1)
        else:
            masks = logits_masks
        # Remove background
        if self._no_bg:
            masks = masks[:, 1:]
        return masks


class PoolingModel(nn.Module):
    """Downsampled (or pooling) part model."""

    def __init__(self, args: Namespace, segmenter: nn.Module):
        """Initialize Downsampled part model."""
        print("=> Initializing PoolingModel...")
        super().__init__()
        self._segmenter = segmenter
        no_bg = "nobg" in args.experiment
        use_bn_after_pooling = "bn" in args.experiment
        input_dim = args.seg_labels - (1 if no_bg else 0)

        idx = args.experiment.find("pooling")
        pool_size = int(args.experiment[idx:].split("-")[1])
        var_per_mask = 5
        print(
            f"Creating a downsampled part model (no_bg: {no_bg}, input_dim: "
            f"{input_dim}, pool_size: {pool_size})..."
        )

        batchnorm = []
        if use_bn_after_pooling:
            batchnorm = [nn.BatchNorm2d(input_dim), nn.ReLU(inplace=True)]

        self.core_model = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            *batchnorm,
            nn.Conv2d(
                input_dim, input_dim * var_per_mask, (pool_size, pool_size)
            ),
            nn.BatchNorm2d(input_dim * var_per_mask),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(input_dim * var_per_mask, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(inplace=True),
            nn.Linear(50, args.num_classes),
        )
        self.feature_extactor = PoolingFeatureExtractor(no_bg)

    def get_classifier(self):
        """Get model that takes logit mask and returns classification output."""
        return nn.Sequential(self.feature_extactor, self.core_model)

    def forward(
        self, images: torch.Tensor, return_mask: bool = False, **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """End-to-end prediction from images.

        Args:
            images:
            return_mask: If True, return predicted mask together with
                classification outputs.

        Returns:
            Predicted classes and segmentation maskes (if return_mask is True)
            in logit form.
        """
        _ = kwargs  # Unused
        # Segmentation part
        logits_masks = self._segmenter(images)
        masks = self.feature_extactor(logits_masks, from_logits=True)
        out = self.core_model(masks)
        if return_mask:
            return out, logits_masks
        return out
