"""Utility functions for setting up attack modules."""

import torch
import torch.nn as nn

import part_model.models as pm_models
from part_model.utils.loss import (
    PixelwiseCELoss,
    SegGuidedCELoss,
    SemiSumLoss,
)

from part_model.attack.auto import AutoAttackModule
from part_model.attack.auto_square import AutoAttackSPModule
from part_model.attack.corruption_benchmark import CorruptionBenchmarkModule

from part_model.attack.hsj import HopSkipJump
from part_model.attack.masked_pgd import MaskedPGDAttackModule
from part_model.attack.mat import MATAttackModule
from part_model.attack.none import NoAttackModule
from part_model.attack.pgd import PGDAttackModule
from part_model.attack.rays import RayS
from part_model.attack.seg_guide import SegGuidedAttackModule
from part_model.attack.seg_pgd import SegPGDAttackModule
from part_model.attack.trades import TRADESAttackModule


# def setup_seg_guide_loss(args):
#     from part_model.dataloader.cityscapes import seg_file_to_mask

#     # TODO
#     guide_images = ["./figures/00092.tif", "./figures/00033.tif"]
#     guide_masks, loss_masks = [], []
#     for i in range(args.num_classes):
#         guide_mask, mask = seg_file_to_mask(guide_images[i])
#         guide_masks.append(guide_mask.cuda(args.gpu))
#         # TODO: 0 -> 1, 1 -> 0
#         loss_masks.append(mask[:, :, 1 - i].cuda(args.gpu))
#     guide_masks = torch.stack(guide_masks, dim=0)
#     loss_masks = torch.stack(loss_masks, dim=0)
#     loss = SingleSegGuidedCELoss(
#         guide_masks, loss_masks=loss_masks, const=args.seg_loss_const
#     )
#     return loss


def get_loss(args, option):
    if "seg-only" in args.experiment:
        loss = PixelwiseCELoss(reduction="pixelmean").cuda(args.gpu)
    elif option == "both":
        loss = [
            SemiSumLoss(seg_const=0).cuda(args.gpu),
            SemiSumLoss(seg_const=1).cuda(args.gpu),
        ]
    else:
        loss = {
            "ce": nn.CrossEntropyLoss(reduction="none"),
            "seg": SemiSumLoss(seg_const=1, reduction="none"),
            "sum": SemiSumLoss(seg_const=args.seg_const_atk, reduction="none"),
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

    no_attack = NoAttackModule(None, None, None, norm, eps)
    attack_list = [("no_attack", no_attack)]
    if args.eval_attack == "":
        return attack_list

    for atk in args.eval_attack.split(","):
        if atk == "pgd":
            attack = PGDAttackModule(
                attack_config, model, get_loss(args, "ce"), norm, eps
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
            attack = SegGuidedAttackModule(
                {
                    "pgd_steps": num_steps,
                    "pgd_step_size": max(0.001, eps / 4 / (num_steps / 10)),
                    "num_restarts": 5,
                    "guide_selection": guide_selection_method,
                    "seg_const": seg_const,
                    "use_two_stages": use_two_stages,
                },
                model,
                get_loss(args, "seg"),
                norm,
                eps,
                classifier=pm_models._wrap_distributed(
                    args, model.module.get_classifier()
                ),
                dataloader=guide_dataloader,
                seg_labels=args.seg_labels,
            )
        elif atk == "single-seg":
            attack = PGDAttackModule(
                attack_config,
                model,
                get_loss(args, "ce"),
                norm,
                eps,
                forward_args={"return_mask": True},
            )
        elif atk == "seg":
            attack = SegPGDAttackModule(
                attack_config, model, get_loss(args, "seg"), norm, eps
            )
        elif atk == "seg-sum":
            attack = SegPGDAttackModule(
                attack_config, model, get_loss(args, "sum"), norm, eps
            )
        elif atk == "mpgd":
            attack = MaskedPGDAttackModule(
                attack_config, model, get_loss(args, "ce"), norm, eps
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
            attack = PGDAttackModule(
                attack_config, model, get_loss(args, "ce"), norm, eps
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
        # 'pgd_steps': 5,
        # 'pgd_step_size': eps / 3,
        # 'pgd_steps': 3,
        # 'pgd_step_size': eps / 2,
        "num_restarts": 1,
    }

    attack = {
        "none": NoAttackModule(None, None, None, norm, eps),
        "pgd": PGDAttackModule(
            attack_config, model, get_loss(args, "ce"), norm, eps
        ),
        "pgd-semi-sum": SegPGDAttackModule(
            attack_config, model, get_loss(args, "sum"), norm, eps
        ),
        "pgd-semi-seg": SegPGDAttackModule(
            attack_config, model, get_loss(args, "seg"), norm, eps
        ),
        "pgd-semi-both": SegPGDAttackModule(
            attack_config, model, get_loss(args, "both"), norm, eps
        ),
        "trades": TRADESAttackModule(
            attack_config, model, get_loss(args, "ce"), norm, eps
        ),
        "mat": MATAttackModule(
            attack_config, model, get_loss(args, "ce"), norm, eps
        ),
        "mpgd": MaskedPGDAttackModule(
            attack_config, model, get_loss(args, "ce"), norm, eps
        ),
        "atta": PGDAttackModule(
            attack_config, model, get_loss(args, "ce"), norm, eps
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
        "pgd_steps": 100,
        "pgd_step_size": 0.002,
        "num_restarts": 1,
    }
    if args.adv_train == "mpgd":
        return MaskedPGDAttackModule(
            attack_config, model, get_loss(args, "ce"), norm, eps
        )
    # TODO: special case for hard-label pixel model
    if "pixel" in args.experiment and "hard" in args.experiment:
        return SegPGDAttackModule(
            attack_config, model, get_loss(args, "seg"), norm, eps
        )

    return PGDAttackModule(
        attack_config, model, get_loss(args, "ce"), norm, eps
    )
