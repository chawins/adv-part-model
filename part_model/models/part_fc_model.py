import torch
import torch.nn as nn
import torch.nn.functional as F
from part_model.dataloader import DATASET_DICT


class PartFCModel(nn.Module):
    def __init__(self, args, segmenter):
        print("=> Initializing BoundingBoxModel...")
        super(PartFCModel, self).__init__()
        self.segmenter = segmenter
        self.no_bg = "nobg" in args.experiment
        input_dim = (
            (args.seg_labels - int(self.no_bg))
            if "condense" in args.experiment
            else (args.seg_labels - int(self.no_bg)) * 5
        )

        self.fc = nn.Sequential(
            nn.Conv2d(14, 1, (50, 50), 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3481, 200),
            nn.ReLU(),
            nn.Linear(200, input_dim),
            nn.Sigmoid(),
        )

        self.core_model = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, args.num_classes),
        )

    def forward(self, images, return_mask=False, **kwargs):
        # Segmentation part
        logits_masks = self.segmenter(images)
        # masks: [B, num_segs (including background), H, W]
        masks = F.softmax(logits_masks, dim=1)

        # Remove background
        if self.no_bg:
            masks = masks[:, 1:]

        condensed = self.fc(masks)

        out = self.core_model(condensed)

        if return_mask:
            return out, logits_masks
        return out
