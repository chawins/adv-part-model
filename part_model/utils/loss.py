"""Definition of all loss functions."""

from __future__ import annotations

import copy
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from DINO.models.dino.dino import SetCriterion
from DINO.models.dino.matcher import build_matcher

_EPS = 1e-6
_LARGE_NUM = 1e8


def _cw_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Linear magin loss (Carlini-Wagner loss)."""
    targets_oh = F.one_hot(targets, num_classes=logits.shape[1])
    if targets.ndim == 1:
        target_logits = torch.index_select(logits, dim=-1, index=targets)
    else:
        # For segmentatio mask (assume targets.ndim == 3)
        targets_oh = targets_oh.permute(0, 3, 1, 2)
        target_logits = (targets_oh * logits).sum(1)
    max_other_logits = (logits - targets_oh * _LARGE_NUM).max(1)[0]
    loss = max_other_logits - target_logits
    loss.clamp_max_(1e-3)
    return loss


def _trades_loss(
    cl_logits: torch.Tensor,
    adv_logits: torch.Tensor,
    targets: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    cl_loss = F.cross_entropy(cl_logits, targets, reduction="mean")
    cl_probs = F.softmax(cl_logits, dim=1)
    adv_lprobs = F.log_softmax(adv_logits, dim=1)
    adv_loss = F.kl_div(adv_lprobs, cl_probs, reduction="batchmean")
    return cl_loss + beta * adv_loss


def _mat_loss(
    cl_logits: torch.Tensor,
    adv_logits: torch.Tensor,
    targets: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    cl_loss = F.cross_entropy(cl_logits, targets, reduction="mean")
    adv_loss = F.cross_entropy(adv_logits, targets, reduction="mean")
    return (1 - beta) * cl_loss + beta * adv_loss


def _seg_loss(
    seg_mask: torch.Tensor, seg_targets: torch.Tensor, loss_fn: str = "ce"
) -> torch.Tensor:
    """Compute segmentation loss.

    If CE loss is used, ignore targets (and corresponding masks) that are set to
    -1.
    """
    if seg_mask.size(0) == 2 * seg_targets.size(0):
        seg_targets = torch.cat([seg_targets, seg_targets], dim=0)
    # Ignore targets that were set to -1 (hack to simulate semi-supervised
    # segmentation)
    if loss_fn == "ce":
        semi_mask = seg_targets[:, 0, 0] >= 0
        seg_loss = 0
        if semi_mask.any():
            seg_loss = F.cross_entropy(
                seg_mask[semi_mask], seg_targets[semi_mask], reduction="none"
            ).mean((1, 2))
        return seg_loss
    # Expect seg_mask to be logits
    seg_loss = F.kl_div(seg_mask, seg_targets, reduction="none").mean((1, 2))
    return seg_loss


def _semi_keypoint_loss(
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    object_masks_sums: torch.Tensor,
    seg_targets: torch.Tensor,
    label_targets: torch.Tensor,
) -> torch.Tensor:
    grid = torch.arange(seg_targets.shape[2])[None, None, :].cuda()
    targets = F.one_hot(seg_targets, num_classes=center_x.shape[1] + 1)
    target_masks = targets.permute(0, 3, 2, 1)
    target_masks = target_masks[:, 1:]
    target_mask_sums = torch.sum(target_masks, [2, 3]) + _EPS
    target_mask_sums_x = torch.sum(target_masks, 2) + _EPS
    target_mask_sums_y = torch.sum(target_masks, 3) + _EPS

    # Part centroid is standardized by object's centroid and sd
    target_center_x = (target_mask_sums_x * grid).sum(
        2
    ) / target_mask_sums / seg_targets.shape[1] * 2 - 1
    target_center_y = (target_mask_sums_y * grid).sum(
        2
    ) / target_mask_sums / seg_targets.shape[2] * 2 - 1

    # Only penalize parts that exist in seg_targets
    present_part = torch.sum(target_masks, (2, 3)) > 0
    keypoint_loss_x = F.mse_loss(
        target_center_x[present_part], center_x[present_part]
    )
    keypoint_loss_y = F.mse_loss(
        target_center_y[present_part], center_y[present_part]
    )
    keypoint_loss = keypoint_loss_x + keypoint_loss_y

    # TODO: This loss probably drives all pixels to one part/class
    cls_loss = F.cross_entropy(object_masks_sums, label_targets)
    return cls_loss + keypoint_loss


class PixelwiseCELoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self._reduction: str = reduction

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        if self._reduction == "pixelmean":
            loss = F.cross_entropy(logits, targets, reduction="none")
            return loss.mean((1, 2))
        return F.cross_entropy(logits, targets, reduction=self._reduction)


class TRADESLoss(nn.Module):
    def __init__(self, beta: float) -> None:
        super().__init__()
        self._beta: float = beta

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        batch_size = logits.size(0) // 2
        cl_logits, adv_logits = logits[:batch_size], logits[batch_size:]
        loss = _trades_loss(cl_logits, adv_logits, targets, self._beta)
        return loss


class MATLoss(nn.Module):
    def __init__(self, beta: float) -> None:
        super().__init__()
        self._beta: float = beta

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        batch_size = logits.size(0) // 2
        cl_logits, adv_logits = logits[:batch_size], logits[batch_size:]
        loss = _mat_loss(cl_logits, adv_logits, targets, self._beta)
        return loss


class KLDLoss(nn.Module):
    def __init__(self, reduction: str = "none") -> None:
        super().__init__()
        assert reduction in ("none", "mean", "sum-non-batch")
        self._reduction: str = reduction

    def forward(
        self, cl_logits: torch.Tensor, adv_logits: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        cl_probs = F.softmax(cl_logits, dim=1)
        adv_lprobs = F.log_softmax(adv_logits, dim=1)
        if self._reduction in ("none", "mean"):
            return F.kl_div(adv_lprobs, cl_probs, reduction=self._reduction)
        loss = F.kl_div(adv_lprobs, cl_probs, reduction="none")
        dims = tuple(range(1, loss.ndim))
        return loss.sum(dims)


class SingleSegGuidedCELoss(nn.Module):
    def __init__(
        self,
        guide_masks: torch.Tensor,
        loss_masks: torch.Tensor = None,
        const: float = 1.0,
    ) -> None:
        super().__init__()
        self.guide_masks = guide_masks
        self.loss_masks = loss_masks
        self.loss_scales = torch.tensor(
            [m.sum().item() for m in loss_masks], device=guide_masks.device
        )
        self.const = const

    def forward(
        self,
        logits: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
        **kwargs,
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
    def __init__(self, const: float = 1.0) -> None:
        super().__init__()
        self._const: float = const

    def forward(
        self,
        logits: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
        guides: torch.Tensor | None = None,
        guide_masks: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError("DEPRECATED")
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
        return clf_loss - self._const * loss


class SemiSumLoss(nn.Module):
    def __init__(
        self,
        seg_const: float = 0.5,
        reduction: str = "mean",
        targeted_seg: bool = False,
        seg_loss_fn: str = "ce",
    ) -> None:
        super().__init__()
        assert 0 <= seg_const <= 1
        self._seg_const: float = seg_const
        self._reduction: str = reduction
        self._targeted_seg: bool = targeted_seg
        self._seg_loss_fn: str = seg_loss_fn

    def forward(
        self,
        logits: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
        seg_targets: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute CE segmentation + classification loss."""
        logits, seg_mask = logits
        loss = 0
        if self._seg_const < 1:
            clf_loss = F.cross_entropy(logits, targets, reduction="none")
            loss += (1 - self._seg_const) * clf_loss

        if self._seg_const > 0:
            if self._seg_loss_fn == "ce":
                # Check whether target masks contain any negative number. We use
                # -1 to specify masks that we want to drop when seg_frac < 1.
                semi_mask = seg_targets[:, 0, 0] >= 0
                seg_loss = torch.zeros_like(semi_mask, dtype=torch.float32)
                seg_loss[semi_mask] = _seg_loss(
                    seg_mask, seg_targets, loss_fn="ce"
                )
            else:
                seg_loss = _seg_loss(seg_mask, seg_targets, loss_fn="kld")
            if self._targeted_seg:
                seg_loss *= -1
            loss += self._seg_const * seg_loss

        if self._reduction == "mean":
            return loss.mean()

        return loss


class SemiSumLinearLoss(nn.Module):
    def __init__(
        self,
        seg_const: float = 0.5,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        assert 0 <= seg_const <= 1
        self._seg_const: float = seg_const
        self._reduction: str = reduction

    def forward(
        self,
        logits: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
        seg_targets: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute linear segmentation + classification loss."""
        logits, seg_logits = logits
        loss = 0
        if self._seg_const < 1:
            clf_loss = _cw_loss(logits, targets)
            loss += (1 - self._seg_const) * clf_loss
        if self._seg_const > 0:
            # Check whether target masks contain any negative number. We use
            # -1 to specify masks that we want to drop when seg_frac < 1.
            semi_mask = seg_targets[:, 0, 0] >= 0
            seg_loss = torch.zeros_like(semi_mask, dtype=torch.float32)
            seg_loss[semi_mask] = _cw_loss(seg_logits, seg_targets).mean((1, 2))
            loss += self._seg_const * seg_loss
        if self._reduction == "mean":
            return loss.mean()
        return loss


class SemiKeypointLoss(nn.Module):
    def __init__(self, seg_const: float = 0.5, reduction: str = "mean") -> None:
        super().__init__()
        assert 0 <= seg_const <= 1
        self._seg_const: float = seg_const
        self._reduction: str = reduction

    def forward(
        self,
        logits: Union[list, tuple],
        targets: torch.Tensor,
        seg_targets: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute keypooint + classification loss."""
        logits, seg_mask = logits
        seg_mask, centerX, centerY, object_masks_sums = seg_mask
        loss = 0
        if self._seg_const < 1:
            clf_loss = F.cross_entropy(logits, targets, reduction="none")
            loss += (1 - self._seg_const) * clf_loss
        if self._seg_const > 0:
            semi_mask = seg_targets[:, 0, 0] >= 0
            seg_loss = torch.zeros_like(semi_mask, dtype=logits.dtype)
            seg_loss[semi_mask] = _semi_keypoint_loss(
                centerX, centerY, object_masks_sums, seg_targets, targets
            )
            loss += self._seg_const * seg_loss
        if self._reduction == "mean":
            return loss.mean()
        return loss


class SemiSegTRADESLoss(nn.Module):
    def __init__(self, const: float = 1.0, beta: float = 1.0) -> None:
        super().__init__()
        self.const: float = const
        self.beta: float = beta
        self.adv_only: bool = True

    def forward(
        self,
        logits: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
        seg_targets: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute segmentation + TRADES loss."""
        logits, seg_mask = logits
        batch_size = logits.size(0) // 2
        cl_logits, adv_logits = logits[:batch_size], logits[batch_size:]
        if self.adv_only:
            seg_mask = seg_mask[batch_size:]
        seg_loss = _seg_loss(seg_mask, seg_targets).mean()
        clf_loss = F.cross_entropy(cl_logits, targets, reduction="mean")
        cl_probs = F.softmax(cl_logits, dim=1)
        adv_lprobs = F.log_softmax(adv_logits, dim=1)
        adv_loss = F.kl_div(adv_lprobs, cl_probs, reduction="batchmean")
        loss = (
            (1 - self.const) * clf_loss
            + self.const * seg_loss
            + self.beta * adv_loss
        )
        return loss


class SemiSegMATLoss(nn.Module):
    def __init__(self, const: float = 1.0, beta: float = 1.0) -> None:
        super().__init__()
        self._const: float = const
        self._beta: float = beta
        self._adv_only: bool = True

    def forward(
        self,
        logits: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
        seg_targets: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute segmentation + MAT loss."""
        logits, seg_mask = logits
        batch_size = logits.size(0) // 2
        cl_logits, adv_logits = logits[:batch_size], logits[batch_size:]
        if self._adv_only:
            seg_mask = seg_mask[batch_size:]
        seg_loss = _seg_loss(seg_mask, seg_targets).mean()
        clf_loss = _mat_loss(cl_logits, adv_logits, targets, self._beta)
        return (1 - self._const) * clf_loss + self._const * seg_loss


def get_train_criterion(args):
    if "seg-only" in args.experiment:
        if args.obj_det_arch == "dino":
            matcher, weight_dict, losses = get_dino_loss_params(args)
            # criterion = SetCriterion(args.seg_labels, matcher, weight_dict, args.focal_alpha, losses)
            criterion = BBOXLoss(
                args.seg_labels,
                matcher=matcher,
                weight_dict=weight_dict,
                focal_alpha=args.focal_alpha,
                losses=losses,
            )
        else:
            criterion = PixelwiseCELoss().cuda(args.gpu)
    else:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    train_criterion = criterion
    if args.adv_train == "trades" and args.obj_det_arch != "dino":
        if "semi" in args.experiment:
            train_criterion = SemiSegTRADESLoss(
                args.seg_const_trn, args.adv_beta
            )
        else:
            train_criterion = TRADESLoss(args.adv_beta)
    elif args.adv_train == "mat":
        if "semi" in args.experiment:
            train_criterion = SemiSegMATLoss(args.seg_const_trn, args.adv_beta)
        else:
            train_criterion = MATLoss(args.adv_beta)
    elif "semi" in args.experiment:
        if "centroid" in args.experiment:
            train_criterion = SemiKeypointLoss(seg_const=args.seg_const_trn)

        if args.obj_det_arch == "dino":
            matcher, weight_dict, losses = get_dino_loss_params(args)
            if args.adv_train == "trades":
                train_criterion = SemiBBOXTRADESLoss(
                    args.seg_labels,
                    matcher=matcher,
                    weight_dict=weight_dict,
                    focal_alpha=args.focal_alpha,
                    losses=losses,
                    seg_const=args.seg_const_trn,
                    const=args.seg_const_trn,
                    beta=args.adv_beta,
                )
            else:
                train_criterion = SemiBBOXLoss(
                    args.seg_labels,
                    matcher=matcher,
                    weight_dict=weight_dict,
                    focal_alpha=args.focal_alpha,
                    losses=losses,
                    seg_const=args.seg_const_trn,
                )
        else:
            train_criterion = SemiSumLoss(seg_const=args.seg_const_trn)
    train_criterion = train_criterion.cuda(args.gpu)
    return criterion, train_criterion


class BBOXLoss(SetCriterion):
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        focal_alpha,
        losses,
        reduction: str = "mean",
    ):
        super().__init__(num_classes, matcher, weight_dict, focal_alpha, losses)
        self.weight_dict = weight_dict

    def forward(
        self,
        dino_outputs: dict,
        dino_targets: list,
        return_indices=False,
        **kwargs,
    ):
        """This performs the loss computation.
        Parameters:
            outputs: dict of tensors, see the output specification of the model for the format
            targets: list of dicts, such that len(targets) == batch_size.
                    The expected keys in each dict depends on the losses applied, see each loss' doc

            return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        loss_dict = super().forward(dino_outputs, dino_targets, return_indices)
        bbox_loss = sum(
            loss_dict[k] * self.weight_dict[k]
            for k in loss_dict.keys()
            if k in self.weight_dict
        )
        return bbox_loss


class SemiBBOXLoss(SetCriterion):
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        focal_alpha,
        losses,
        seg_const: float = 0.5,
        reduction: str = "mean",
    ):
        super().__init__(num_classes, matcher, weight_dict, focal_alpha, losses)
        assert 0 <= seg_const <= 1
        self.seg_const = seg_const
        self.reduction = reduction
        self.weight_dict = weight_dict

    def forward(
        self,
        logits: Union[list, tuple],
        dino_outputs: dict,
        dino_targets: list,
        targets: torch.Tensor,
        return_indices=False,
        **kwargs,
    ):
        """This performs the loss computation.
        Parameters:
            outputs: dict of tensors, see the output specification of the model for the format
            targets: list of dicts, such that len(targets) == batch_size.
                    The expected keys in each dict depends on the losses applied, see each loss' doc

            return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        loss = 0
        if self.seg_const < 1:
            clf_loss = F.cross_entropy(logits, targets, reduction="none")
            loss += (1 - self.seg_const) * clf_loss
        if self.seg_const > 0:
            loss_dict = super().forward(
                dino_outputs, dino_targets, return_indices
            )
            bbox_loss = sum(
                loss_dict[k] * self.weight_dict[k]
                for k in loss_dict.keys()
                if k in self.weight_dict
            )
            loss += self.seg_const * bbox_loss

        if self.reduction == "mean":
            return loss.mean()

        return loss


class SemiBBOXTRADESLoss(SetCriterion):
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        focal_alpha,
        losses,
        seg_const: float = 0.5,
        reduction: str = "mean",
        const: float = 1.0,
        beta: float = 1.0,
    ):
        super().__init__(num_classes, matcher, weight_dict, focal_alpha, losses)
        self.seg_const = seg_const
        self.reduction = reduction
        self.weight_dict = weight_dict

        # TRADES
        self.const = const
        self.beta = beta
        self.adv_only = True

    def truncate_dictionary(self, dictionary, k):
        for key, value in dictionary.items():
            if isinstance(value, list) or torch.is_tensor(value):
                dictionary[key] = value[k:]
            elif isinstance(value, dict):
                dictionary[key] = self.truncate_dictionary(value, k)
        return dictionary

    def forward(
        self,
        logits: Union[list, tuple],
        dino_outputs: dict,
        dino_targets: list,
        targets: torch.Tensor,
        return_indices=False,
        **kwargs,
    ):
        """This performs the loss computation.
        Parameters:
            outputs: dict of tensors, see the output specification of the model for the format
            targets: list of dicts, such that len(targets) == batch_size.
                    The expected keys in each dict depends on the losses applied, see each loss' doc

            return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        batch_size = logits.size(0) // 2
        cl_logits, adv_logits = logits[:batch_size], logits[batch_size:]
        if self.adv_only:
            dino_outputs = self.truncate_dictionary(dino_outputs, batch_size)
            dino_targets = dino_targets[batch_size:]
        loss_dict = super().forward(dino_outputs, dino_targets, return_indices)
        bbox_loss = sum(
            loss_dict[k] * self.weight_dict[k]
            for k in loss_dict.keys()
            if k in self.weight_dict
        )

        clf_loss = F.cross_entropy(cl_logits, targets, reduction="mean")
        cl_probs = F.softmax(cl_logits, dim=1)
        adv_lprobs = F.log_softmax(adv_logits, dim=1)
        adv_loss = F.kl_div(adv_lprobs, cl_probs, reduction="batchmean")
        loss = (
            (1 - self.const) * clf_loss
            + self.const * bbox_loss
            + self.beta * adv_loss
        )

        return loss


def get_dino_loss_params(args):
    losses = ["labels", "boxes", "cardinality"]

    matcher = build_matcher(args)
    # prepare weight dict
    weight_dict = {
        "loss_ce": args.cls_loss_coef,
        "loss_bbox": args.bbox_loss_coef,
    }
    weight_dict["loss_giou"] = args.giou_loss_coef
    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

    # for DN training
    if args.use_dn:
        weight_dict["loss_ce_dn"] = args.cls_loss_coef
        weight_dict["loss_bbox_dn"] = args.bbox_loss_coef
        weight_dict["loss_giou_dn"] = args.giou_loss_coef

    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    clean_weight_dict = copy.deepcopy(weight_dict)

    # TODO(nab-126@): this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update(
                {k + f"_{i}": v for k, v in clean_weight_dict.items()}
            )
        weight_dict.update(aux_weight_dict)

    if args.two_stage_type != "no":
        interm_weight_dict = {}
        try:
            no_interm_box_loss = args.no_interm_box_loss
        except:
            no_interm_box_loss = False
        _coeff_weight_dict = {
            "loss_ce": 1.0,
            "loss_bbox": 1.0 if not no_interm_box_loss else 0.0,
            "loss_giou": 1.0 if not no_interm_box_loss else 0.0,
        }
        try:
            interm_loss_coef = args.interm_loss_coef
        except:
            interm_loss_coef = 1.0
        interm_weight_dict.update(
            {
                f"{k}_interm": v * interm_loss_coef * _coeff_weight_dict[k]
                for k, v in clean_weight_dict_wo_dn.items()
            }
        )
        weight_dict.update(interm_weight_dict)

    return matcher, weight_dict, losses
