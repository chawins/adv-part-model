import torch
import torch.nn as nn
import torch.nn.functional as F


class GroundtruthMaskModel(nn.Module):
    def __init__(self, args, segmenter, core_model):
        super(GroundtruthMaskModel, self).__init__()
        self.core_model = core_model
        self.concat_input = "inpt" in args.experiment
        self.no_bg = "nobg" in args.experiment
        self.seg_labels = args.seg_labels

    def forward(self, images, segs=None, return_mask=False, **kwargs):
        # Segmentation part

        bg_idx = 1 if self.no_bg else 0

        part_masks = []
        for i in range(bg_idx, self.seg_labels):
            part_masks.append(segs == i)
        masks = torch.stack(part_masks, dim=1).float()

        masks = masks[:, bg_idx:]
        if self.concat_input:
            masks = torch.cat([images, masks], dim=1)

        # Classify directly based on the mask
        out = self.core_model(masks)
        return out
