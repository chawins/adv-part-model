from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def trades_loss(cl_logits, adv_logits, targets, beta):
    cl_loss = F.cross_entropy(cl_logits, targets, reduction="mean")
    cl_probs = F.softmax(cl_logits, dim=1)
    adv_lprobs = F.log_softmax(adv_logits, dim=1)
    adv_loss = F.kl_div(adv_lprobs, cl_probs, reduction="batchmean")
    return cl_loss + beta * adv_loss


def mat_loss(cl_logits, adv_logits, targets, beta):
    cl_loss = F.cross_entropy(cl_logits, targets, reduction="mean")
    adv_loss = F.cross_entropy(adv_logits, targets, reduction="mean")
    return (1 - beta) * cl_loss + beta * adv_loss


def semi_seg_loss(seg_mask, seg_targets):
    if seg_mask.size(0) == 2 * seg_targets.size(0):
        seg_targets = torch.cat([seg_targets, seg_targets], dim=0)
    # Ignore targets that were set to -1 (hack to simulate semi-supervised
    # segmentation)
    semi_mask = seg_targets[:, 0, 0] >= 0
    seg_loss = 0
    if semi_mask.any():
        seg_loss = F.cross_entropy(
            seg_mask[semi_mask], seg_targets[semi_mask], reduction="none"
        )
        return seg_loss.mean((1, 2))
    return seg_loss


def semi_keypoint_loss(seg_mask, seg_targets, targets):
    # hard-coded for ALL PartImageNet
    CLASSES = {
        "Quadruped": 4,
        "Biped": 5,
        "Fish": 4,
        "Bird": 5,
        "Snake": 2,
        "Reptile": 4,
        "Car": 3,
        "Bicycle": 4,
        "Boat": 2,
        "Aeroplane": 5,
        "Bottle": 2,
    }
    grid = torch.arange(seg_mask.shape[3])[None, None, :].cuda()
    masks = F.softmax(seg_mask, dim=1)
    # Remove background
    # masks = masks[:, 1:]

    # out: [batch_size, num_classes]
    sum_scores = (masks).sum((2, 3))
    class_scores = (sum_scores / sum_scores.sum((1,))[:,None])[:, 1:]
    # print(class_scores)

    no_bg_mask = masks[:, 1:]
    # Compute mean and sd for part mask
    mask_sums = torch.sum(no_bg_mask, [2, 3])
    mask_sumsX = torch.sum(no_bg_mask, 2)
    mask_sumsY = torch.sum(no_bg_mask, 3)

    # Part centroid is standardized by object's centroid and sd
    centerX = (mask_sumsX * grid).sum(2) / mask_sums
    centerY = (mask_sumsY * grid).sum(2) / mask_sums

    centerX /= seg_mask.shape[3]
    centerY /= seg_mask.shape[3]

    targets = []
    for i in range(seg_mask.shape[1]):
        targets.append(torch.where(seg_targets == i, 1.0, 0.0))
    target_masks = torch.stack(targets).permute(1, 0, 2, 3)
    target_masks = target_masks[:, 1:]
    present_part = torch.where(torch.sum(target_masks, (2, 3)) > 0, 1.0, 0.0)
    present_part /= torch.sum(present_part)
    target_mask_sums = torch.sum(target_masks, [2, 3])
    target_mask_sumsX = torch.sum(target_masks, 2)
    target_mask_sumsY = torch.sum(target_masks, 3)

    # Part centroid is standardized by object's centroid and sd
    target_centerX = (target_mask_sumsX * grid).sum(2) / target_mask_sums
    target_centerY = (target_mask_sumsY * grid).sum(2) / target_mask_sums

    target_centerX /= seg_mask.shape[3]
    target_centerY /= seg_mask.shape[3]

    loss = torch.nn.BCELoss()(class_scores, present_part)
    loss = F.mse_loss(
        target_centerX[present_part > 0], centerX[present_part > 0]
    ) + F.mse_loss(target_centerY[present_part > 0], centerY[present_part > 0])
    return loss


class SemiKeypointLoss(nn.Module):
    def __init__(self, seg_const: float = 0.5, reduction: str = "mean"):
        super(SemiKeypointLoss, self).__init__()
        assert 0 <= seg_const <= 1
        self.seg_const = seg_const
        self.reduction = reduction
        

    def forward(
        self,
        logits: Union[list, tuple],
        targets: torch.Tensor,
        seg_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        logits, seg_mask = logits
        loss = 0
        if self.seg_const < 1:
            clf_loss = F.cross_entropy(logits, targets, reduction="none")
            loss += (1 - self.seg_const) * clf_loss
        if self.seg_const > 0:
            semi_mask = seg_targets[:, 0, 0] >= 0
            seg_loss = torch.zeros_like(semi_mask, dtype=logits.dtype)
            seg_loss[semi_mask] = semi_keypoint_loss(seg_mask, seg_targets, targets)
            loss += self.seg_const * seg_loss
        if self.reduction == "mean":
            return loss.mean()
        return loss


class PixelwiseCELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        if self.reduction == "pixelmean":
            loss = F.cross_entropy(logits, targets, reduction="none")
            return loss.mean((1, 2))
        return F.cross_entropy(logits, targets, reduction=self.reduction)


class TRADESLoss(nn.Module):
    def __init__(self, beta):
        super(TRADESLoss, self).__init__()
        self.beta = beta

    def forward(self, logits, targets):
        batch_size = logits.size(0) // 2
        cl_logits, adv_logits = logits[:batch_size], logits[batch_size:]
        loss = trades_loss(cl_logits, adv_logits, targets, self.beta)
        return loss


class MATLoss(nn.Module):
    def __init__(self, beta):
        super(MATLoss, self).__init__()
        self.beta = beta

    def forward(self, logits, targets):
        batch_size = logits.size(0) // 2
        cl_logits, adv_logits = logits[:batch_size], logits[batch_size:]
        loss = mat_loss(cl_logits, adv_logits, targets, self.beta)
        return loss


class KLDLoss(nn.Module):
    def __init__(self, reduction="none"):
        super(KLDLoss, self).__init__()
        assert reduction in ("none", "mean", "sum-non-batch")
        self.reduction = reduction

    def forward(self, cl_logits, adv_logits):
        cl_probs = F.softmax(cl_logits, dim=1)
        adv_lprobs = F.log_softmax(adv_logits, dim=1)
        if self.reduction in ("none", "mean"):
            return F.kl_div(adv_lprobs, cl_probs, reduction=self.reduction)
        loss = F.kl_div(adv_lprobs, cl_probs, reduction="none")
        dims = tuple(range(1, loss.ndim))
        return loss.sum(dims)


class SingleSegGuidedCELoss(nn.Module):
    def __init__(
        self,
        guide_masks: torch.Tensor,
        loss_masks: torch.Tensor = None,
        const: float = 1.0,
    ):
        super(SingleSegGuidedCELoss, self).__init__()
        self.guide_masks = guide_masks
        self.loss_masks = loss_masks
        self.loss_scales = torch.tensor(
            [m.sum().item() for m in loss_masks], device=guide_masks.device
        )
        self.const = const

    def forward(
        self, logits: Union[list, tuple, torch.Tensor], targets: torch.Tensor
    ) -> torch.Tensor:
        loss = 0
        if isinstance(logits, (list, tuple)):
            seg_mask = logits[1]
            logits = logits[0]
            # CE loss on segmentation mask
            seg_loss = F.cross_entropy(
                seg_mask, self.guide_masks[targets], reduction="none"
            )
            seg_loss *= self.loss_masks[targets]
            loss = -seg_loss.sum((1, 2)) / self.loss_scales[targets]
        clf_loss = F.cross_entropy(logits, targets, reduction="none")
        return clf_loss + self.const * loss


class SegGuidedCELoss(nn.Module):
    def __init__(self, const: float = 1.0):
        super(SegGuidedCELoss, self).__init__()
        self.const = const

    def forward(
        self,
        logits: Union[list, tuple],
        targets: torch.Tensor,
        guides: Optional[torch.Tensor] = None,
        guide_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = 0
        if guides is not None:
            seg_mask = logits[1]
            logits = logits[0]
            # CE loss on segmentation mask
            seg_loss = F.cross_entropy(seg_mask, guides, reduction="none")
            seg_loss *= guide_masks
            loss = seg_loss.sum((1, 2)) / guide_masks.sum((1, 2))
        # TODO: assume binary label
        clf_loss = -F.cross_entropy(logits, 1 - targets, reduction="none")
        return clf_loss - self.const * loss


class SemiSumLoss(nn.Module):
    def __init__(self, seg_const: float = 0.5, reduction: str = "mean"):
        super(SemiSumLoss, self).__init__()
        assert 0 <= seg_const <= 1
        self.seg_const = seg_const
        self.reduction = reduction

    def forward(
        self,
        logits: Union[list, tuple],
        targets: torch.Tensor,
        seg_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        logits, seg_mask = logits
        loss = 0
        if self.seg_const < 1:
            clf_loss = F.cross_entropy(logits, targets, reduction="none")
            loss += (1 - self.seg_const) * clf_loss
        if self.seg_const > 0:
            semi_mask = seg_targets[:, 0, 0] >= 0
            seg_loss = torch.zeros_like(semi_mask, dtype=logits.dtype)
            seg_loss[semi_mask] = semi_seg_loss(seg_mask, seg_targets)
            loss += self.seg_const * seg_loss
        if self.reduction == "mean":
            return loss.mean()
        return loss


class SemiSegTRADESLoss(nn.Module):
    def __init__(self, const: float = 1.0, beta: float = 1.0):
        super(SemiSegTRADESLoss, self).__init__()
        self.const = const
        self.beta = beta
        self.adv_only = True

    def forward(
        self,
        logits: Union[list, tuple],
        targets: torch.Tensor,
        seg_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        logits, seg_mask = logits
        batch_size = logits.size(0) // 2
        cl_logits, adv_logits = logits[:batch_size], logits[batch_size:]
        if self.adv_only:
            seg_mask = seg_mask[batch_size:]
        seg_loss = semi_seg_loss(seg_mask, seg_targets).mean()
        clf_loss = F.cross_entropy(cl_logits, targets, reduction="mean")
        cl_probs = F.softmax(cl_logits, dim=1)
        adv_lprobs = F.log_softmax(adv_logits, dim=1)
        adv_loss = F.kl_div(adv_lprobs, cl_probs, reduction="batchmean")
        loss = (
            (1 - self.const) * clf_loss + self.const * seg_loss + self.beta * adv_loss
        )
        return loss


class SemiSegMATLoss(nn.Module):
    def __init__(self, const: float = 1.0, beta: float = 1.0):
        super(SemiSegMATLoss, self).__init__()
        self.const = const
        self.beta = beta
        self.adv_only = True

    def forward(
        self,
        logits: Union[list, tuple],
        targets: torch.Tensor,
        seg_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        logits, seg_mask = logits
        batch_size = logits.size(0) // 2
        cl_logits, adv_logits = logits[:batch_size], logits[batch_size:]
        if self.adv_only:
            seg_mask = seg_mask[batch_size:]
        seg_loss = semi_seg_loss(seg_mask, seg_targets).mean()
        clf_loss = mat_loss(cl_logits, adv_logits, targets, self.beta)
        return (1 - self.const) * clf_loss + self.const * seg_loss


def get_train_criterion(args):
    if "seg-only" in args.experiment:
        criterion = PixelwiseCELoss().cuda(args.gpu)
    else:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    train_criterion = criterion
    if args.adv_train == "trades":
        if "semi" in args.experiment:
            train_criterion = SemiSegTRADESLoss(args.seg_const_trn, args.adv_beta)
        else:
            train_criterion = TRADESLoss(args.adv_beta)
    elif args.adv_train == "mat":
        if "semi" in args.experiment:
            train_criterion = SemiSegMATLoss(args.seg_const_trn, args.adv_beta)
        else:
            train_criterion = MATLoss(args.adv_beta)
    elif "semi" in args.experiment:
        if "keypoint" in args.experiment:
            train_criterion = SemiKeypointLoss(seg_const=args.seg_const_trn)
        else:
            train_criterion = SemiSumLoss(seg_const=args.seg_const_trn)
    train_criterion = train_criterion.cuda(args.gpu)
    return criterion, train_criterion
