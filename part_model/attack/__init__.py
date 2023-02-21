"""Utility functions for setting up attack modules."""

from __future__ import annotations

import math

from torch import nn

import part_model.models as pm_models
from part_model.attack.auto import AutoAttackModule
from part_model.attack.auto_square import AutoAttackSPModule
from part_model.attack.corruption_benchmark import CorruptionBenchmarkModule
from part_model.attack.hsj import HopSkipJump
from part_model.attack.masked_pgd import MaskedPGDAttack
from part_model.attack.mat import MATAttack
from part_model.attack.none import NoAttack
from part_model.attack.pgd import PGDAttack
from part_model.attack.rays import RayS
from part_model.attack.seg_guide import SegGuidedAttack
from part_model.attack.seg_inverse import SegInverseAttack
from part_model.attack.trades import TRADESAttack
from part_model.utils.loss import (
    BBOXLoss,
    PixelwiseCELoss,
    SegGuidedCELoss,
    SemiSumLinearLoss,
    SemiSumLoss,
    get_dino_loss_params,
)


def _get_loss(args, option):
    if "seg-only" in args.experiment:
        if args.obj_det_arch == "dino":
            matcher, weight_dict, losses = get_dino_loss_params(args)
            loss = BBOXLoss(
                args.seg_labels, matcher, weight_dict, args.focal_alpha, losses
            )
        else:
            loss = PixelwiseCELoss(reduction="pixelmean").cuda(args.gpu)
    elif option == "both":
        loss = [
            SemiSumLoss(seg_const=0).cuda(args.gpu),
            SemiSumLoss(seg_const=1).cuda(args.gpu),
        ]
    else:
        loss = {
            "ce": nn.CrossEntropyLoss(reduction="none"),
            # TODO: Rename to be less ambiguous
            "seg-sum": SemiSumLoss(
                seg_const=args.seg_const_atk, reduction="none"
            ),
            "seg-ce": SemiSumLoss(seg_const=1, reduction="none"),
            "seg-linear": SemiSumLinearLoss(seg_const=1, reduction="none"),
            "seg-guide": SegGuidedCELoss(const=args.seg_const_atk),
            # 'single-seg': setup_seg_guide_loss(args),
        }[option].cuda(args.gpu)
    return loss


def setup_eval_attacker(args, model, num_classes=None, guide_dataloader=None):

    if num_classes is None:
        num_classes = args.num_classes
    eps = float(args.epsilon)
    norm = args.atk_norm
    num_steps = 100
    attack_config = {
        "pgd_steps": num_steps,
        "pgd_step_size": max(0.001, eps / 4 / (num_steps / 10)),
        "num_restarts": 5,
    }

    no_attack = NoAttack(None, None, None, norm, eps)
    attack_list = [("no_attack", no_attack)]
    if args.eval_attack == "":
        return attack_list

    for atk in args.eval_attack.split(","):
        if atk == "pgd":
            attack = PGDAttack(
                attack_config, model, _get_loss(args, "ce"), norm, eps
            )
        elif atk == "aa":
            attack = AutoAttackModule(
                None,
                model,
                None,
                norm,
                eps,
                verbose=True,
                num_classes=num_classes,
            )
        elif atk == "aasp":
            # AutoAttack - Square+
            attack = AutoAttackSPModule(
                None,
                model,
                None,
                norm,
                eps,
                verbose=True,
                num_classes=num_classes,
            )
        elif "seg-guide" in atk:
            # seg-guide/<selection_method>/<seg_const>/<ts> (optional)
            seg_atk_tokens = atk.split("/")
            guide_selection_method = seg_atk_tokens[1]
            seg_const = float(seg_atk_tokens[2])
            use_two_stages = seg_atk_tokens[-1] == "ts"
            attack = SegGuidedAttack(
                {
                    "pgd_steps": num_steps,
                    "pgd_step_size": max(0.001, eps / 4 / (num_steps / 10)),
                    "num_restarts": 5,
                    "guide_selection": guide_selection_method,
                    "seg_const": seg_const,
                    "use_two_stages": use_two_stages,
                },
                model,
                _get_loss(args, "seg-ce"),
                norm,
                eps,
                classifier=pm_models.wrap_distributed(
                    args, model.module.get_classifier()
                ),
                dataloader=guide_dataloader,
                seg_labels=args.seg_labels,
            )
        elif atk == "single-seg":
            attack = PGDAttack(
                attack_config,
                model,
                _get_loss(args, "ce"),
                norm,
                eps,
                forward_args={"return_mask": True},
            )
        elif atk == "seg-inverse":
            # loss_fn is defined in SegInverseAttackModule
            attack_config["seg_const"] = args.seg_const_atk
            # TODO: Better define norm?
            # L2-norm of sqrt(d) in logit space
            attack_config["mask_l2_eps"] = 224 * math.sqrt(args.seg_labels)
            attack_config["num_restarts"] = 2
            attack = SegInverseAttack(attack_config, model, None, norm, eps)
        elif "seg-" in atk:
            attack = PGDAttack(
                attack_config, model, _get_loss(args, atk), norm, eps
            )
        elif atk == "mpgd":
            attack = MaskedPGDAttack(
                attack_config, model, _get_loss(args, "ce"), norm, eps
            )
        # elif atk == 'hsja':
        #     attack = HopSkipJumpAttack(None, model, None, norm, eps)
        elif atk == "rays":
            attack = RayS(None, model, None, norm, eps, num_classes=num_classes)
        elif atk == "hsj":
            attack = HopSkipJump(
                None, model, None, norm, eps, num_classes=num_classes
            )
        elif "corrupt" in atk:
            attack = CorruptionBenchmarkModule(
                None, None, None, norm, None, int(atk[7:])
            )
        elif atk == "longer-pgd":
            num_steps = 300
            attack_config = {
                "pgd_steps": num_steps,
                "pgd_step_size": max(0.001, eps / 4 / (num_steps / 10)),
                "num_restarts": 2,
            }
            attack = PGDAttack(
                attack_config, model, _get_loss(args, "ce"), norm, eps
            )
        attack_list.append((atk, attack))

    return attack_list


def setup_train_attacker(args, model):

    eps: float = float(args.epsilon)
    use_atta: bool = args.adv_train == "atta"
    norm: str = args.atk_norm
    attack_config = {
        "pgd_steps": 1 if use_atta else args.atk_steps,
        "pgd_step_size": eps / args.atk_steps * 1.25,
        "num_restarts": 1,
    }

    attack = {
        "none": NoAttack(None, None, None, norm, eps),
        "pgd": PGDAttack(
            attack_config, model, _get_loss(args, "ce"), norm, eps
        ),
        "pgd-semi-sum": PGDAttack(
            attack_config, model, _get_loss(args, "seg-sum"), norm, eps
        ),
        "pgd-semi-seg": PGDAttack(
            attack_config, model, _get_loss(args, "seg-ce"), norm, eps
        ),
        "pgd-semi-both": PGDAttack(
            attack_config, model, _get_loss(args, "both"), norm, eps
        ),
        "trades": TRADESAttack(
            attack_config, model, _get_loss(args, "ce"), norm, eps
        ),
        "mat": MATAttack(
            attack_config, model, _get_loss(args, "ce"), norm, eps
        ),
        "mpgd": MaskedPGDAttack(
            attack_config, model, _get_loss(args, "ce"), norm, eps
        ),
        "atta": PGDAttack(
            attack_config, model, _get_loss(args, "ce"), norm, eps
        ),
    }[args.adv_train]

    return attack


def setup_aa_attacker(args, model, num_classes=None):
    """Set up AutoAttack for validation."""
    if num_classes is None:
        num_classes = args.num_classes
    eps = float(args.epsilon)
    norm = args.atk_norm
    attack = AutoAttackModule(
        None, model, None, norm, eps, verbose=False, num_classes=num_classes
    )
    return attack


def setup_val_attacker(args, model):
    eps = float(args.epsilon)
    norm = args.atk_norm
    attack_config = {
        "pgd_steps": 50,
        "pgd_step_size": 0.002,
        "num_restarts": 1,
    }
    if args.adv_train == "mpgd":
        return MaskedPGDAttack(
            attack_config, model, _get_loss(args, "ce"), norm, eps
        )
    # TODO: special case for hard-label pixel model
    if "pixel" in args.experiment and "hard" in args.experiment:
        return PGDAttack(
            attack_config, model, _get_loss(args, "seg-ce"), norm, eps
        )

    return PGDAttack(attack_config, model, _get_loss(args, "ce"), norm, eps)
