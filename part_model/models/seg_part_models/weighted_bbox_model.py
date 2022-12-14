"""Weighted bounding-box part model."""

from __future__ import annotations

from argparse import Namespace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from part_model.dataloader import DATASET_DICT

_EPS = 1e-6


class WeightedBBoxFeatureExtractor(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        norm_by_img: bool,
        no_score: bool,
        use_conv1d: bool,
    ) -> None:
        """Initialize WeightedBBoxFeatureExtractor.

        Args:
            height: _description_
            width: _description_
            norm_by_img: _description_
            no_score: _description_
            use_conv1d: _description_
        """
        super().__init__()
        self.height = height
        self.width = width
        self.norm_by_img = norm_by_img
        self.no_score = no_score
        self.use_conv1d = use_conv1d
        grid = torch.arange(height)[None, None, :]
        self.register_buffer("grid", grid, persistent=False)

    def forward(
        self,
        logits_masks: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # masks: [B, num_segs (including background), H, W]
        masks = F.softmax(logits_masks, dim=1)
        # Remove background
        masks = masks[:, 1:]

        # Compute foreground/background mask (fg_score - bg_score)
        fg_mask = (
            logits_masks[:, 1:].sum(1, keepdim=True) - logits_masks[:, 0:1]
        )
        fg_mask = torch.sigmoid(fg_mask)
        fg_mask = fg_mask / fg_mask.sum((2, 3), keepdim=True).clamp_min(_EPS)
        # weighted_logits_masks = logits_masks[:, 1:] * fg_mask
        # masks = F.softmax(weighted_logits_masks, dim=1)

        # out: [batch_size, num_classes]
        class_scores = (logits_masks[:, 1:] * fg_mask).sum((2, 3))

        # Compute mean and sd for part mask
        mask_sums = torch.sum(masks, [2, 3]) + _EPS
        mask_sumsX = torch.sum(masks, 2) + _EPS
        mask_sumsY = torch.sum(masks, 3) + _EPS

        # Part centroid is standardized by object's centroid and sd
        centerX = (mask_sumsX * self.grid).sum(2) / mask_sums
        centerY = (mask_sumsY * self.grid).sum(2) / mask_sums
        sdX = (mask_sumsX * (self.grid - centerX.unsqueeze(-1)) ** 2).sum(
            2
        ) / mask_sums
        sdY = (mask_sumsY * (self.grid - centerY.unsqueeze(-1)) ** 2).sum(
            2
        ) / mask_sums
        sdX = sdX.sqrt()
        sdY = sdY.sqrt()

        if self.norm_by_img:
            # Normalize centers to [-1, 1]
            centerX = centerX / self.width * 2 - 1
            centerY = centerY / self.height * 2 - 1
            # Max sdX is W / 2 (two pixels on 0 and W-1). Normalize to [0, 1]
            sdX = sdX / self.width * 2
            sdY = sdY / self.height * 2

        if self.no_score:
            segOut = [centerX, centerY, sdX, sdY]
        else:
            segOut = [class_scores, centerX, centerY, sdX, sdY]
        # segOut: [batch_size, num_classes/parts, num_features (4 or 5)]
        segOut = torch.cat([s.unsqueeze(-1) for s in segOut], dim=2)
        if self.use_conv1d:
            segOut = segOut.permute(0, 2, 1)
        return segOut


class WeightedBBoxModel(nn.Module):
    def __init__(self, args: Namespace, segmenter: nn.Module) -> None:
        print("=> Initializing WeightedBBoxModel...")
        super().__init__()
        self.segmenter = segmenter
        self.use_conv1d = "conv1d" in args.experiment
        self.no_score = "no_score" in args.experiment
        dim = 4 if self.no_score else 5
        dim_per_bbox = 10 if self.use_conv1d else dim
        input_dim = (args.seg_labels - 1) * dim_per_bbox
        datasetDict = DATASET_DICT[args.dataset]

        self.return_centroid = "centroid" in args.experiment
        part_to_class = datasetDict["part_to_class"]
        part_to_class = torch.tensor(part_to_class, dtype=torch.float32)
        bg_idx = 1
        self.register_buffer(
            "part_to_class_mat",
            part_to_class[bg_idx:, bg_idx:][None, :, :, None, None],
            persistent=False,
        )

        self.norm_by_img = "norm_img" in args.experiment
        _, height, width = datasetDict["input_dim"]
        self.height = height
        self.width = width
        self.totalPixels = height * width

        self.core_model = nn.Sequential(
            nn.Conv1d(dim, 10, 1) if self.use_conv1d else nn.Identity(),
            nn.Flatten(),
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, args.num_classes),
        )

        grid = torch.arange(height)[None, None, :]
        self.register_buffer("grid", grid, persistent=False)

        # TODO: find a clean way to reuse feature_extactor in forward
        self.feature_extactor = WeightedBBoxFeatureExtractor(
            height, width, self.norm_by_img, self.no_score, self.use_conv1d
        )

    def get_classifier(self) -> nn.Module:
        return nn.Sequential(self.feature_extactor, self.core_model)

    def forward(
        self, images: torch.Tensor, return_mask: bool = False, **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        # Segmentation part
        logits_masks = self.segmenter(images)
        segOut = self.feature_extactor(logits_masks)
        out = self.core_model(segOut)

        if return_mask:
            if self.return_centroid:
                # Get softmax mask and remove background
                masks = F.softmax(logits_masks, dim=1)
                masks = masks[:, 1:]
                object_masks = masks.unsqueeze(2) * self.part_to_class_mat
                object_masks = object_masks.sum(1)
                object_masks_sums = torch.sum(object_masks, [2, 3]) / (
                    self.height * self.width
                )
                centerX = segOut[:, :, -4]
                centerY = segOut[:, :, -3]
                logits_masks = (
                    logits_masks,
                    centerX,
                    centerY,
                    object_masks_sums,
                )
            return out, logits_masks
        return out
