from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from DINO.models.dino.matcher import build_matcher
from DINO.models.dino.dino import SetCriterion
import copy

_EPS = 1e-6


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


def semi_keypoint_loss(centerX, centerY, object_masks_sums, seg_targets, label_targets):
    grid = torch.arange(seg_targets.shape[2])[None, None, :].cuda()
    targets = F.one_hot(seg_targets, num_classes=centerX.shape[1] + 1)
    target_masks = targets.permute(0, 3, 2, 1)
    target_masks = target_masks[:, 1:]
    target_mask_sums = torch.sum(target_masks, [2, 3]) + _EPS
    target_mask_sumsX = torch.sum(target_masks, 2) + _EPS
    target_mask_sumsY = torch.sum(target_masks, 3) + _EPS
    # Part centroid is standardized by object's centroid and sd
    target_centerX = (target_mask_sumsX * grid).sum(
        2
    ) / target_mask_sums / seg_targets.shape[1] * 2 - 1
    target_centerY = (target_mask_sumsY * grid).sum(
        2
    ) / target_mask_sums / seg_targets.shape[2] * 2 - 1
    # TODO: This probably doesn't need sqrt?
    # loss = torch.sqrt(
    #     F.mse_loss(target_centerX[present_part > 0], centerX[present_part > 0])
    #     + F.mse_loss(
    #         target_centerY[present_part > 0], centerY[present_part > 0]
    #     )
    # )
    # loss += F.nll_loss(object_masks_sums, label_targets)
    # Only penalize parts that exist in seg_targets
    present_part = torch.sum(target_masks, (2, 3)) > 0
    keypoint_loss_x = F.mse_loss(target_centerX[present_part], centerX[present_part])
    keypoint_loss_y = F.mse_loss(target_centerY[present_part], centerY[present_part])
    keypoint_loss = keypoint_loss_x + keypoint_loss_y
    # TODO: This loss probably drives all pixels to one part/class
    cls_loss = F.cross_entropy(object_masks_sums, label_targets)
    return cls_loss + keypoint_loss


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
            # Check whether target masks contain any negative number. We use
            # -1 to specify masks that we want to drop when seg_frac < 1.
            semi_mask = seg_targets[:, 0, 0] >= 0
            seg_loss = torch.zeros_like(semi_mask, dtype=logits.dtype)
            seg_loss[semi_mask] = semi_seg_loss(seg_mask, seg_targets)
            loss += self.seg_const * seg_loss
        if self.reduction == "mean":
            return loss.mean()
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
        seg_mask, centerX, centerY, object_masks_sums = seg_mask
        loss = 0
        if self.seg_const < 1:
            clf_loss = F.cross_entropy(logits, targets, reduction="none")
            loss += (1 - self.seg_const) * clf_loss
        if self.seg_const > 0:
            semi_mask = seg_targets[:, 0, 0] >= 0
            seg_loss = torch.zeros_like(semi_mask, dtype=logits.dtype)
            seg_loss[semi_mask] = semi_keypoint_loss(
                centerX, centerY, object_masks_sums, seg_targets, targets
            )
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
    if args.adv_train == "trades" and args.obj_det_arch != "dino":
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
        if "centroid" in args.experiment:
            train_criterion = SemiKeypointLoss(seg_const=args.seg_const_trn)

        if args.obj_det_arch == "dino": 
            losses = ['labels', 'boxes', 'cardinality']
        
            matcher = build_matcher(args)
            # prepare weight dict
            weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
            weight_dict['loss_giou'] = args.giou_loss_coef
            clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)
            
            # for DN training
            if args.use_dn:
                weight_dict['loss_ce_dn'] = args.cls_loss_coef
                weight_dict['loss_bbox_dn'] = args.bbox_loss_coef
                weight_dict['loss_giou_dn'] = args.giou_loss_coef

            if args.masks:
                weight_dict["loss_mask"] = args.mask_loss_coef
                weight_dict["loss_dice"] = args.dice_loss_coef
            clean_weight_dict = copy.deepcopy(weight_dict)

            # TODO this is a hack
            if args.aux_loss:
                aux_weight_dict = {}
                for i in range(args.dec_layers - 1):
                    aux_weight_dict.update({k + f'_{i}': v for k, v in clean_weight_dict.items()})
                weight_dict.update(aux_weight_dict)

            if args.two_stage_type != 'no':
                interm_weight_dict = {}
                try:
                    no_interm_box_loss = args.no_interm_box_loss
                except:
                    no_interm_box_loss = False
                _coeff_weight_dict = {
                    'loss_ce': 1.0,
                    'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
                    'loss_giou': 1.0 if not no_interm_box_loss else 0.0,
                }
                try:
                    interm_loss_coef = args.interm_loss_coef
                except:
                    interm_loss_coef = 1.0
                interm_weight_dict.update({k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k] for k, v in clean_weight_dict_wo_dn.items()})
                weight_dict.update(interm_weight_dict)

            if args.adv_train == "trades":
                train_criterion = SemiBBOXTRADESLoss(args.num_classes, matcher=matcher, weight_dict=weight_dict,
                                        focal_alpha=args.focal_alpha, losses=losses, seg_const=args.seg_const_trn,
                                        const=args.seg_const_trn, beta=args.adv_beta)
            else:
                train_criterion = SemiBBOXLoss(args.num_classes, matcher=matcher, weight_dict=weight_dict,
                                        focal_alpha=args.focal_alpha, losses=losses, seg_const=args.seg_const_trn)
        
        else:
            train_criterion = SemiSumLoss(seg_const=args.seg_const_trn)
    train_criterion = train_criterion.cuda(args.gpu)
    return criterion, train_criterion


class SemiBBOXLoss(SetCriterion):
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses, seg_const: float = 0.5, reduction: str = "mean"):
        super(SemiBBOXLoss, self).__init__(num_classes, matcher, weight_dict, focal_alpha, losses)
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
        return_indices=False):
        """ This performs the loss computation.
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
            loss_dict = super().forward(dino_outputs, dino_targets, return_indices)
            bbox_loss = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)
            loss += self.seg_const * bbox_loss
        if self.reduction == "mean":
            return loss.mean()

        return loss


class SemiBBOXTRADESLoss(SetCriterion):
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses, seg_const: float = 0.5, reduction: str = "mean", const: float = 1.0, beta: float = 1.0):
        super(SemiBBOXTRADESLoss, self).__init__(num_classes, matcher, weight_dict, focal_alpha, losses)
        self.seg_const = seg_const
        self.reduction = reduction
        self.weight_dict = weight_dict
        
        # TRADES
        self.const = const
        self.beta = beta
        self.adv_only = True

    def truncate_dictionary(
        self,
        dictionary,
        k
    ):
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
        return_indices=False):
        """ This performs the loss computation.
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
        bbox_loss = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)

        clf_loss = F.cross_entropy(cl_logits, targets, reduction="mean")
        cl_probs = F.softmax(cl_logits, dim=1)
        adv_lprobs = F.log_softmax(adv_logits, dim=1)
        adv_loss = F.kl_div(adv_lprobs, cl_probs, reduction="batchmean")
        loss = (
            (1 - self.const) * clf_loss + self.const * bbox_loss + self.beta * adv_loss
        )

        return loss