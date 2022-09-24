import torch
import torch.nn as nn
import torch.nn.functional as F


class PoolingFeatureExtractor(nn.Module):
    def __init__(self, no_bg: bool):
        super().__init__()
        self.no_bg = no_bg

    def forward(self, logits_masks: torch.Tensor) -> torch.Tensor:
        # masks: [B, num_segs (including background), H, W]
        masks = F.softmax(logits_masks, dim=1)
        # Remove background
        if self.no_bg:
            masks = masks[:, 1:]
        return masks


class PoolingModel(nn.Module):
    def __init__(self, args, segmenter):
        print("=> Initializing PoolingModel...")
        super().__init__()
        self.segmenter = segmenter
        self.no_bg = "nobg" in args.experiment
        use_bn_after_pooling = "bn" in args.experiment
        input_dim = args.seg_labels - (1 if self.no_bg else 0)

        idx = args.experiment.find("pooling")
        pool_size = int(args.experiment[idx:].split("-")[1])
        var_per_mask = 5

        if use_bn_after_pooling:
            bn = [nn.BatchNorm2d(input_dim), nn.ReLU(inplace=True)]
        else:
            bn = []

        self.core_model = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            *bn,
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
        self.feature_extactor = PoolingFeatureExtractor(self.no_bg)

    def get_classifier(self):
        return nn.Sequential(self.feature_extactor, self.core_model)

    def forward(self, images, return_mask=False, **kwargs):
        # Segmentation part
        logits_masks = self.segmenter(images)
        masks = self.feature_extactor(logits_masks)
        out = self.core_model(masks)
        if return_mask:
            return out, logits_masks
        return out
