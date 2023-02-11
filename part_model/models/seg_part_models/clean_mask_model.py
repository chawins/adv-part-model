import torch
import torch.nn as nn
import torch.nn.functional as F


class CleanMaskModel(nn.Module):
    def __init__(self, args, segmenter, core_model):
        super(CleanMaskModel, self).__init__()
        self.segmenter = segmenter
        self.core_model = core_model
        self.concat_input = "inpt" in args.experiment
        self.hard_label_mask = "hard" in args.experiment
        self.no_bg = "nobg" in args.experiment
        self.seg_labels = args.seg_labels
        self.detach_mask = "detach" in args.experiment
        self.clean_masks = None

    def forward(self, images, return_mask=False, clean=False, **kwargs):
        # Segmentation part
        if clean:
            logits_masks = self.segmenter(images)
            masks = F.softmax(logits_masks, dim=1)
            self.clean_masks = masks.detach()
        else:
            masks = self.clean_masks

        if self.detach_mask:
            masks = masks.detach()

        bg_idx = 1 if self.no_bg else 0

        if self.hard_label_mask:
            masks = masks.argmax(1)
            # Use all parts in order and exclude background
            part_masks = []
            for i in range(bg_idx, self.seg_labels):
                part_masks.append(masks == i)
            masks = torch.stack(part_masks, dim=1).float()

        masks = masks[:, bg_idx:]
        if self.concat_input:
            masks = torch.cat([images, masks], dim=1)

        # Classify directly based on the mask
        out = self.core_model(masks)
        if return_mask:
            return out, self.clean_masks
        return out
