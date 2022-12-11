from operator import matmul

import torch
import torch.nn as nn
import torch.nn.functional as F
from part_model.dataloader import DATASET_DICT


class PixelCountModel(nn.Module):
    def __init__(self, args, segmenter, core_model):
        super(PixelCountModel, self).__init__()
        print("=> Initializing PixelCountModel...")
        self.segmenter = segmenter
        exp_tokens = args.experiment.split("-")
        self.hard_label_mask = "hard" in exp_tokens
        if "softmax_mask" in exp_tokens:
            self.mode = "softmax_mask"
        elif "logits_mask" in exp_tokens:
            self.mode = "logits_mask"
        else:
            self.mode = None
        self.seg_labels = args.seg_labels
        self.detach_mask = "detach" in exp_tokens
        self.group_part_by_class = "part" in exp_tokens
        dataset_params = DATASET_DICT[args.dataset]
        if self.group_part_by_class:
            part_to_class = dataset_params["part_to_class"]
            self.register_buffer(
                "part_to_class_mat",
                torch.tensor(part_to_class, dtype=torch.float32),
            )

    def forward(self, images, return_mask=False, **kwargs):
        # Segmentation part
        logits_masks = self.segmenter(images)
        masks = F.softmax(logits_masks, dim=1)

        if self.detach_mask:
            masks = masks.detach()

        if self.group_part_by_class:
            # masks: [B, NP, H, W], part_to_class_mat: [NP, NC]
            masks = torch.matmul(
                masks.permute(0, 2, 3, 1), self.part_to_class_mat
            ).permute(0, 3, 1, 2)
            class_logits_masks = torch.matmul(
                logits_masks.permute(0, 2, 3, 1), self.part_to_class_mat
            ).permute(0, 3, 1, 2)

        if self.hard_label_mask:
            masks = masks.argmax(1)
            masks = F.one_hot(masks, num_classes=self.seg_labels).to(
                images.dtype
            )
            masks = masks.permute(0, 3, 1, 2)

        if self.mode is not None:
            # fg_score - bg_score
            fg_mask = (
                logits_masks[:, 1:].sum(1, keepdim=True) - logits_masks[:, 0:1]
            )
            fg_mask = torch.sigmoid(fg_mask)
            fg_mask = fg_mask / fg_mask.sum((2, 3), keepdim=True).clamp_min(
                1e-6
            )
            if self.mode == "softmax_mask":
                # FIXME: This will see double softmax
                class_mask = F.softmax(class_logits_masks[:, 1:], dim=1)
                # class_mask = logits_masks[:, 1:].clamp(-1e2, 1e2)
            else:
                class_mask = class_logits_masks[:, 1:]
            # out: [batch_size, num_classes]
            out = (class_mask * fg_mask).sum((2, 3))
        else:
            masks = class_logits_masks[:, 1:]
            # Classify directly based on the mask
            out = masks.mean((2, 3))  # TODO: can use sum (or temperature) here

        if return_mask:
            return out, logits_masks
        return out
