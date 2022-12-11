import torch
import torch.nn as nn
import torch.nn.functional as F
from part_model.dataloader import DATASET_DICT


class PartSegModel(nn.Module):
    def __init__(self, args, segmenter, core_model, rep_dim, topk=None):
        super(PartSegModel, self).__init__()
        self.segmenter = segmenter
        # self.segmentor = segmenter
        self.core_model = core_model
        dataset_params = DATASET_DICT[args.dataset]
        self.use_soft_mask = "soft" in args.experiment
        self.group_part_by_class = "group" in args.experiment
        self.no_bg = "nobg" in args.experiment

        self.k = topk if topk is not None else args.seg_labels
        if self.group_part_by_class:
            part_to_class = dataset_params["part_to_class"]
            self.part_to_class_mat = torch.tensor(
                part_to_class, dtype=torch.float32, device=args.gpu
            )
            self.k = self.part_to_class_mat.size(-1)
        self.mask = None
        self.temperature = args.temperature

        # Aggregation layer
        self.linear_dim = rep_dim * (self.k - self.no_bg)
        self.linear = nn.Linear(self.linear_dim, args.num_classes)

    def forward(self, images, return_mask=False, **kwargs):
        batch_size = images.size(0)
        images = self._apply_mask(images)

        # Segmentation part
        logits_masks = self.segmenter(images)
        masks = F.softmax(logits_masks / self.temperature, dim=1)
        bg_idx = 1 if self.no_bg else 0

        if self.group_part_by_class:
            masks = (
                masks.unsqueeze(2)
                * self.part_to_class_mat[None, :, :, None, None]
            )
            masks = masks.sum(1)

        if self.use_soft_mask:
            images = images.unsqueeze(1) * masks[:, bg_idx:, :, :].unsqueeze(2)
        else:
            label_masks = F.one_hot(masks.argmax(1), num_classes=self.k)[
                :, :, :, bg_idx:
            ]
            images = images.unsqueeze(1) * label_masks.permute(
                0, 3, 1, 2
            ).unsqueeze(2)

        images = images.reshape(
            (batch_size * (self.k - bg_idx),) + images.shape[2:]
        )

        out = self.core_model(images)
        out = out.view(batch_size, self.linear_dim)
        out = self.linear(out)
        if return_mask:
            return out, logits_masks
        return out

    def set_mask(self, x):
        self.mask = (x != -1).float()

    def _apply_mask(self, x):
        if self.mask is None:
            return x
        return x * self.mask
