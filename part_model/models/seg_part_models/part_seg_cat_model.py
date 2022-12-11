import torch
import torch.nn as nn
import torch.nn.functional as F


class PartSegCatModel(nn.Module):
    def __init__(self, args, segmentor, core_model, rep_dim):
        super(PartSegCatModel, self).__init__()
        self.segmentor = segmentor
        self.core_model = core_model
        self.num_parts = args.seg_labels - 1

    def forward(self, images, return_mask=False, **kwargs):
        batch_size = images.size(0)

        # Segmentation part
        logits_masks = self.segmentor(images)
        masks = F.softmax(logits_masks, dim=1).argmax(1)

        # Use all parts in order and exclude background
        part_masks = F.one_hot(masks, num_classes=self.num_parts + 1).permute(
            0, 3, 1, 2
        )
        images = images[:, None] * part_masks[:, 1:, None]
        images = images.reshape(
            (batch_size, self.num_parts * 3, images.size(-2), images.size(-1))
        )

        out = self.core_model(images)
        if return_mask:
            return out, logits_masks
        return out
