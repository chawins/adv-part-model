import torch
import torch.nn as nn
import torch.nn.functional as F
from part_model.dataloader import DATASET_DICT


class BoundingBoxModel(nn.Module):
    def __init__(self, args, segmenter):
        print("=> Initializing BoundingBoxModel...")
        super(BoundingBoxModel, self).__init__()
        self.segmenter = segmenter
        self.no_bg = "nobg" in args.experiment
        self.use_conv1d = "conv1d" in args.experiment
        dim_per_bbox = 10 if self.use_conv1d else 5
        input_dim = (args.seg_labels - int(self.no_bg)) * dim_per_bbox
        datasetDict = DATASET_DICT[args.dataset]

        self.norm_by_obj = "norm_obj" in args.experiment
        if self.norm_by_obj:
            # Get matrix that maps parts to object class
            part_to_class = datasetDict["part_to_class"]
            part_to_class = torch.tensor(part_to_class, dtype=torch.float32)
            bg_idx = 1 if self.no_bg else 0
            self.register_buffer(
                "part_to_class_mat",
                part_to_class[bg_idx:, bg_idx:],
                persistent=False,
            )

        self.norm_by_img = "norm_img" in args.experiment
        if self.norm_by_img:
            _, height, width = datasetDict["input_dim"]
            self.height = height
            self.width = width
            self.totalPixels = height * width

        self.core_model = nn.Sequential(
            nn.Conv1d((args.seg_labels - 1), 10, 5)
            if self.use_conv1d
            else nn.Identity(),
            nn.Flatten(),
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, args.num_classes),
        )

        grid = torch.arange(height)[None, None, :]
        self.register_buffer("grid", grid, persistent=False)

    def forward(self, images, return_mask=False, **kwargs):
        # Segmentation part
        logits_masks = self.segmenter(images)
        # masks: [B, num_segs (including background), H, W]
        masks = F.softmax(logits_masks, dim=1)

        # Remove background
        if self.no_bg:
            masks = masks[:, 1:]

        # Compute mean and sd for part mask
        mask_sums = torch.sum(masks, [2, 3])
        mask_sumsX = torch.sum(masks, 2)
        mask_sumsY = torch.sum(masks, 3)

        # Part centroid is standardized by object's centroid and sd
        centerX = (mask_sumsX * self.grid).sum(2) / mask_sums
        centerY = (mask_sumsY * self.grid).sum(2) / mask_sums
        sdX = (mask_sumsX * (self.grid - centerX.unsqueeze(-1)) ** 2).sum(2) / mask_sums
        sdY = (mask_sumsY * (self.grid - centerY.unsqueeze(-1)) ** 2).sum(2) / mask_sums
        sdX = sdX.sqrt()
        sdY = sdY.sqrt()

        # Normalize part location and size by the object's
        if self.norm_by_obj:
            # Get object mask from part mask
            # TODO: this assumes that bbox model is always used with part
            object_masks = (
                masks.unsqueeze(2) * self.part_to_class_mat[None, :, :, None, None]
            )
            object_masks = object_masks.sum(1)

            # Compute mean and sd for object mask
            object_mask_sums = torch.sum(object_masks, [2, 3])
            object_mask_sumsX = torch.sum(object_masks, 2)
            object_mask_sumsY = torch.sum(object_masks, 3)

            object_centerX = (object_mask_sumsX * self.grid).sum(2) / object_mask_sums
            object_centerY = (object_mask_sumsY * self.grid).sum(2) / object_mask_sums
            object_sdX = (
                object_mask_sumsX * (self.grid - object_centerX.unsqueeze(-1)) ** 2
            ).sum(2) / object_mask_sums
            object_sdY = (
                object_mask_sumsY * (self.grid - object_centerY.unsqueeze(-1)) ** 2
            ).sum(2) / object_mask_sums
            # object_sdX.sqrt_()
            # object_sdY.sqrt_()
            object_sdX = object_sdX.sqrt()
            object_sdY = object_sdY.sqrt()

            object_centerX = torch.matmul(object_centerX, self.part_to_class_mat.T)
            object_centerY = torch.matmul(object_centerY, self.part_to_class_mat.T)
            object_sdX = torch.matmul(object_sdX, self.part_to_class_mat.T)
            object_sdY = torch.matmul(object_sdY, self.part_to_class_mat.T)

            # Normalize part's centroid and sd by object's
            # TODO: this can be changed to in-place ops?
            centerX_norm = (centerX - object_centerX) / object_sdX
            centerY_norm = (centerY - object_centerY) / object_sdY
            # TODO: does it make difference to compute sd in the normalized
            # space? (should be NO)
            sdX = sdX / object_sdX
            sdY = sdY / object_sdY
            mask_sums_norm = mask_sums / object_mask_sums
            segOut = torch.cat(
                [
                    mask_sums_norm[:, None, :],
                    centerX_norm[:, None, :],
                    centerY_norm[:, None, :],
                    sdX[:, None, :],
                    sdY[:, None, :],
                ],
                dim=1,
            )

        elif self.norm_by_img:
            # representation: [B, num_parts * 5]
            mask_sums_norm = mask_sums / self.totalPixels
            # Normalize centers to [-1, 1]
            centerX_norm = centerX / self.width * 2 - 1
            centerY_norm = centerY / self.height * 2 - 1
            # Max sdX is W / 2 (two pixels on 0 and W-1). Normalize to [0, 1]
            sdX_norm = sdX / self.width * 2
            sdY_norm = sdY / self.height * 2
            segOut = torch.cat(
                [
                    mask_sums_norm[:, None, :],
                    centerX_norm[:, None, :],
                    centerY_norm[:, None, :],
                    sdX_norm[:, None, :],
                    sdY_norm[:, None, :],
                ],
                dim=1,
            )

        else:
            # No normalization at all
            segOut = torch.cat(
                [
                    mask_sums[:, None, :],
                    centerX[:, None, :],
                    centerY[:, None, :],
                    sdX[:, None, :],
                    sdY[:, None, :],
                ],
                dim=1,
            )

        if self.use_conv1d:
            segOut = segOut.permute(0, 2, 1)
        out = self.core_model(segOut)

        if return_mask:
            return out, logits_masks
        return out
