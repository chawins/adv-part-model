"""Main script for both training and evaluation.

The template of this code is generously provided by Norman Mu (@normster).
The original version is from https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""

import argparse
import json
import math
import os
import pickle
import sys
import time
import PIL
from PIL import Image

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import wandb
from torch.distributed.elastic.multiprocessing.errors import record

# from torchmetrics import IoU as IoU   # Use this for older version of torchmetrics
from torchmetrics import JaccardIndex as IoU
from torchvision.utils import save_image
from torchvision import transforms

from part_model.attack import (
    setup_eval_attacker,
    setup_train_attacker,
    setup_val_attacker,
)
from part_model.attack.pgd import PGDAttackModule
from part_model.dataloader import COLORMAP, load_dataset
from part_model.models import build_model
from part_model.utils import (
    AverageMeter,
    ProgressMeter,
    adjust_learning_rate,
    dist_barrier,
    get_compute_acc,
    get_rank,
    init_distributed_mode,
    is_main_process,
    pixel_accuracy,
    save_on_master,
)
from part_model.utils.loss import get_train_criterion

class_to_channels = {
        "n01440764": [1, 2, 3, 4],
        "n01443537": [5, 6, 7, 8],
        "n01484850": [9, 10, 11, 12],
        "n01491361": [13, 14, 15, 16],
        "n01494475": [17, 18, 19, 20],
        "n01608432": [21, 22, 23, 24, 25],
        "n01614925": [26, 27, 28, 29, 30],
        "n01630670": [31, 32, 33, 34],
        "n01632458": [35, 36, 37, 38],
        "n01641577": [39, 40, 41, 42],
        "n01644373": [43, 44, 45, 46],
        "n01644900": [47, 48, 49, 50],
        "n01664065": [51, 52, 53, 54],
        "n01665541": [55, 56, 57, 58],
        "n01667114": [59, 60, 61, 62],
        "n01667778": [63, 64, 65, 66],
        "n01669191": [67, 68, 69, 70],
        "n01685808": [71, 72, 73, 74],
        "n01687978": [75, 76, 77, 78],
        "n01688243": [79, 80, 81, 82],
        "n01689811": [83, 84, 85, 86],
        "n01692333": [87, 88, 89, 90],
        "n01693334": [91, 92, 93, 94],
        "n01694178": [95, 96, 97, 98],
        "n01695060": [99, 100, 101, 102],
        "n01697457": [103, 104, 105, 106],
        "n01698640": [107, 108, 109, 110],
        "n01728572": [111, 112],
        "n01728920": [113, 114],
        "n01729322": [115, 116],
        "n01729977": [117, 118],
        "n01734418": [119, 120],
        "n01735189": [121, 122],
        "n01739381": [123, 124],
        "n01740131": [125, 126],
        "n01742172": [127, 128],
        "n01744401": [129, 130],
        "n01748264": [131, 132],
        "n01749939": [133, 134],
        "n01753488": [135, 136],
        "n01755581": [137, 138],
        "n01756291": [139, 140],
        "n01824575": [141, 142, 143, 144, 145],
        "n01828970": [146, 147, 148, 149, 150],
        "n01843065": [151, 152, 153, 154, 155],
        "n01855672": [156, 157, 158, 159, 160],
        "n02002724": [161, 162, 163, 164, 165],
        "n02006656": [166, 167, 168, 169, 170],
        "n02009229": [171, 172, 173, 174, 175],
        "n02009912": [176, 177, 178, 179, 180],
        "n02017213": [181, 182, 183, 184, 185],
        "n02025239": [186, 187, 188, 189, 190],
        "n02033041": [191, 192, 193, 194, 195],
        "n02058221": [196, 197, 198, 199, 200],
        "n02071294": [201, 202, 203, 204],
        "n02085782": [205, 206, 207, 208],
        "n02089867": [209, 210, 211, 212],
        "n02090379": [213, 214, 215, 216],
        "n02091831": [217, 218, 219, 220],
        "n02092339": [221, 222, 223, 224],
        "n02096177": [225, 226, 227, 228],
        "n02096585": [229, 230, 231, 232],
        "n02097474": [233, 234, 235, 236],
        "n02098105": [237, 238, 239, 240],
        "n02099601": [241, 242, 243, 244],
        "n02100583": [245, 246, 247, 248],
        "n02101006": [249, 250, 251, 252],
        "n02101388": [253, 254, 255, 256],
        "n02102040": [257, 258, 259, 260],
        "n02102973": [261, 262, 263, 264],
        "n02109525": [265, 266, 267, 268],
        "n02109961": [269, 270, 271, 272],
        "n02112137": [273, 274, 275, 276],
        "n02114367": [277, 278, 279, 280],
        "n02120079": [281, 282, 283, 284],
        "n02124075": [285, 286, 287, 288],
        "n02125311": [289, 290, 291, 292],
        "n02128385": [293, 294, 295, 296],
        "n02129604": [297, 298, 299, 300],
        "n02130308": [301, 302, 303, 304],
        "n02132136": [305, 306, 307, 308],
        "n02133161": [309, 310, 311, 312],
        "n02134084": [313, 314, 315, 316],
        "n02134418": [317, 318, 319, 320],
        "n02356798": [321, 322, 323, 324],
        "n02397096": [325, 326, 327, 328],
        "n02403003": [329, 330, 331, 332],
        "n02408429": [333, 334, 335, 336],
        "n02412080": [337, 338, 339, 340],
        "n02415577": [341, 342, 343, 344],
        "n02417914": [345, 346, 347, 348],
        "n02422106": [349, 350, 351, 352],
        "n02422699": [353, 354, 355, 356],
        "n02423022": [357, 358, 359, 360],
        "n02437312": [361, 362, 363, 364],
        "n02441942": [365, 366, 367, 368],
        "n02442845": [369, 370, 371, 372],
        "n02443114": [373, 374, 375, 376],
        "n02444819": [377, 378, 379, 380],
        "n02447366": [381, 382, 383, 384],
        "n02480495": [385, 386, 387, 388, 389],
        "n02480855": [390, 391, 392, 393, 394],
        "n02481823": [395, 396, 397, 398, 399],
        "n02483362": [400, 401, 402, 403, 404],
        "n02483708": [405, 406, 407, 408, 409],
        "n02484975": [410, 411, 412, 413, 414],
        "n02486261": [415, 416, 417, 418, 419],
        "n02486410": [420, 421, 422, 423, 424],
        "n02487347": [425, 426, 427, 428, 429],
        "n02488702": [430, 431, 432, 433, 434],
        "n02489166": [435, 436, 437, 438, 439],
        "n02490219": [440, 441, 442, 443, 444],
        "n02492035": [445, 446, 447, 448, 449],
        "n02492660": [450, 451, 452, 453, 454],
        "n02493509": [455, 456, 457, 458, 459],
        "n02493793": [460, 461, 462, 463, 464],
        "n02494079": [465, 466, 467, 468, 469],
        "n02510455": [470, 471, 472, 473],
        "n02514041": [474, 475, 476, 477],
        "n02536864": [478, 479, 480, 481],
        "n02607072": [482, 483, 484, 485],
        "n02655020": [486, 487, 488, 489],
        "n02690373": [490, 491, 492, 493, 494],
        "n02701002": [495, 496, 497],
        "n02814533": [498, 499, 500],
        "n02823428": [501, 502],
        "n02835271": [503, 504, 505, 506],
        "n02930766": [507, 508, 509],
        "n03100240": [510, 511, 512],
        "n03417042": [513, 514, 515],
        "n03444034": [516, 517, 518],
        "n03445924": [519, 520, 521],
        "n03594945": [522, 523, 524],
        "n03670208": [525, 526, 527],
        "n03769881": [528, 529, 530],
        "n03770679": [531, 532, 533],
        "n03785016": [534, 535, 536, 537],
        "n03791053": [538, 539, 540, 541],
        "n03792782": [542, 543, 544, 545],
        "n03937543": [546, 547],
        "n03947888": [548, 549],
        "n03977966": [550, 551, 552],
        "n03983396": [553, 554],
        "n04037443": [555, 556, 557],
        "n04065272": [558, 559, 560],
        "n04146614": [561, 562, 563],
        "n04147183": [564, 565],
        "n04252225": [566, 567, 568],
        "n04285008": [569, 570, 571],
        "n04465501": [572, 573, 574],
        "n04482393": [575, 576, 577, 578],
        "n04483307": [579, 580],
        "n04487081": [581, 582, 583],
        "n04509417": [584, 585, 586, 587],
        "n04552348": [588, 589, 590, 591, 592],
        "n04557648": [593, 594],
        "n04591713": [595, 596],
        "n04612504": [597, 598],
    }


def _get_args_parser():
    parser = argparse.ArgumentParser(description="Part classification", add_help=False)
    parser.add_argument("--data", default="~/data/shared/", type=str)
    parser.add_argument("--arch", default="resnet18", type=str)
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Load pretrained model on ImageNet-1k",
    )
    parser.add_argument("--output-dir", default="./", type=str, help="output dir")
    parser.add_argument(
        "-j",
        "--workers",
        default=10,
        type=int,
        metavar="N",
        help="number of data loading workers per process",
    )
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument(
        "--batch-size",
        default=256,
        type=int,
        help="mini-batch size per device.",
    )
    parser.add_argument("--full-precision", action="store_true")
    parser.add_argument("--warmup-epochs", default=0, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--wd", default=1e-4, type=float)
    parser.add_argument("--optim", default="sgd", type=str)
    parser.add_argument("--betas", default=(0.9, 0.999), nargs=2, type=float)
    parser.add_argument("--eps", default=1e-8, type=float)
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument(
        "--resume", default="", type=str, help="path to latest checkpoint"
    )
    parser.add_argument(
        "--load-weight-only",
        action="store_true",
        help="Resume checkpoint by loading model weights only",
    )
    parser.add_argument("--evaluate", action="store_true", help="Evaluate only")
    parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://localhost:10001",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--no-distributed", action="store_true", help="Disable distributed mode"
    )
    parser.add_argument("--dist-backend", default="nccl", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB")
    parser.add_argument(
        "--resume-if-exist",
        action="store_true",
        help=(
            "Override --resume option and resume from the "
            "current best checkpoint in the same dir if exists"
        ),
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    # TODO
    parser.add_argument("--dataset", required=True, type=str, help="Dataset")
    parser.add_argument("--num-classes", default=10, type=int, help="Number of classes")
    parser.add_argument(
        "--experiment",
        required=True,
        type=str,
        help="Type of experiment to run",
    )
    parser.add_argument(
        "--parts",
        default=0,
        type=int,
        help="Number of parts (default: 0 = use full images)",
    )
    parser.add_argument(
        "--use-part-idx",
        action="store_true",
        help="Model also takes part indices as input",
    )
    parser.add_argument(
        "--seg-label-dir",
        default="",
        type=str,
        help="Path to segmentation labels",
    )
    parser.add_argument(
        "--seg-labels",
        default=10,
        type=int,
        help="Number of segmentation classes including background",
    )
    parser.add_argument(
        "--seg-dir",
        default="",
        type=str,
        help="Path to weight of segmentation model",
    )
    parser.add_argument(
        "--freeze-seg",
        action="store_true",
        help="Freeze weights in segmentation model",
    )
    parser.add_argument(
        "--seg-arch",
        default="deeplabv3plus",
        type=str,
        help="Architecture of segmentation model",
    )
    parser.add_argument(
        "--seg-backbone",
        default="resnet18",
        type=str,
        help="Architecture of backbone model",
    )
    parser.add_argument(
        "--epsilon",
        default=8 / 255,
        type=float,
        help="Perturbation norm for attacks (default: 8/255)",
    )
    # Adversarial training
    parser.add_argument(
        "--adv-train",
        default="none",
        type=str,
        help="Use adversarial training (default: none = normal training)",
    )
    parser.add_argument(
        "--atk-steps", default=10, type=int, help="Number of attack iterations"
    )
    parser.add_argument(
        "--atk-norm",
        default="Linf",
        type=str,
        help="Lp-norm of adversarial perturbation (default: Linf)",
    )
    parser.add_argument(
        "--adv-beta",
        default=6.0,
        type=float,
        help="Beta parameter for TRADES or MAT (default: 6)",
    )
    parser.add_argument(
        "--eval-attack",
        default="",
        type=str,
        help="Attacks to evaluate with, comma-separated (default: pgd,aa)",
    )
    parser.add_argument(
        "--seg-const-trn",
        default=0,
        type=float,
        help="Constant in front of seg loss used during training (default: 0)",
    )
    parser.add_argument(
        "--seg-const-atk",
        default=0,
        type=float,
        help="Constant in front of seg loss used during attack (default: 0)",
    )
    parser.add_argument(
        "--semi-label",
        default=1.0,
        type=float,
        help="Fraction of segmentation labels to use in semi-supervised training (default: 1)",
    )
    parser.add_argument(
        "--load-from-segmenter",
        action="store_true",
        help="Resume checkpoint by loading only segmenter weights",
    )
    parser.add_argument(
        "--temperature",
        default=1,
        type=float,
        help="Softmax temperature for part-seg model",
    )
    parser.add_argument("--save-all-epochs", action="store_true")
    parser.add_argument(
        "--prediction-path",
        help="path of predicted masks",
        default="/data/kornrapatp/test",
        type=str,
    )
    return parser


best_acc1 = 0


@record
def main(args):
    """Main function."""
    init_distributed_mode(args)

    global best_acc1

    # Fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data loading code
    print("=> creating dataset")
    loaders = load_dataset(args)
    train_loader, train_sampler, val_loader, test_loader = loaders

    # Create model
    print("=> creating model")
    model, optimizer, scaler = build_model(args)
    cudnn.benchmark = True

    # Define loss function
    criterion, train_criterion = get_train_criterion(args)

    # Logging
    if is_main_process():
        logfile = open(os.path.join(args.output_dir, "log.txt"), "a")
        logfile.write(str(args) + "\n")
        logfile.flush()
        if args.wandb:
            wandb_id = os.path.split(args.output_dir)[-1]
            wandb.init(project="part-model", id=wandb_id, config=args, resume="allow")
            print("wandb step:", wandb.run.step)

    eval_attack = setup_eval_attacker(args, model)
    no_attack = eval_attack[0][1]
    train_attack = setup_train_attacker(args, model)
    val_attack = setup_val_attacker(args, model)
    save_metrics = {
        "train": [],
        "test": [],
    }

    print(args)

    if args.evaluate:
        if args.resume:
            load_path = args.resume
        else:
            load_path = f"{args.output_dir}/checkpoint_best.pt"
    else:
        print("=> beginning training")
        val_stats = {}
        for epoch in range(args.start_epoch, args.epochs):
            is_best = False
            if args.distributed:
                train_sampler.set_epoch(epoch)
            lr = adjust_learning_rate(optimizer, epoch, args)
            print(f"=> lr @ epoch {epoch}: {lr:.2e}")

            # Train for one epoch
            train_stats = _train(
                train_loader,
                model,
                train_criterion,
                train_attack,
                optimizer,
                scaler,
                epoch,
                args,
            )

            if (epoch + 1) % 2 == 0:
                val_stats = _validate(val_loader, model, criterion, no_attack, args)
                clean_acc1, acc1 = val_stats["acc1"], None
                is_best = clean_acc1 > best_acc1
                if args.adv_train != "none":
                    val_stats = _validate(
                        val_loader, model, criterion, val_attack, args
                    )
                    acc1 = val_stats["acc1"]
                    is_best = acc1 > best_acc1 and clean_acc1 >= acc1

                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_acc1": best_acc1,
                    "args": args,
                }

                if is_best:
                    print("=> Saving new best checkpoint")
                    save_on_master(save_dict, args.output_dir, is_best=True)
                    best_acc1 = (
                        max(clean_acc1, best_acc1)
                        if acc1 is None
                        else max(acc1, best_acc1)
                    )
                save_epoch = epoch + 1 if args.save_all_epochs else None
                save_on_master(
                    save_dict, args.output_dir, is_best=False, epoch=save_epoch
                )

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in val_stats.items()},
                "epoch": epoch,
            }

            if is_main_process():
                save_metrics["train"].append(log_stats)
                if args.wandb:
                    wandb.log(log_stats)
                logfile.write(json.dumps(log_stats) + "\n")
                logfile.flush()

        # Compute stats of best model after training
        dist_barrier()
        load_path = f"{args.output_dir}/checkpoint_best.pt"

    print(f"=> loading checkpoint from {load_path}...")
    if args.gpu is None:
        checkpoint = torch.load(load_path)
    else:
        # Map model to be loaded to specified single gpu.
        loc = "cuda:{}".format(args.gpu)
        checkpoint = torch.load(load_path, map_location=loc)
    model.load_state_dict(checkpoint["state_dict"])

    # Running evaluation
    for attack in eval_attack:
        # Use DataParallel (not distributed) model for AutoAttack.
        # Otherwise, DDP model can get timeout or c10d failure.
        stats = _validate(test_loader, model, criterion, attack[1], args)
        print(f"=> {attack[0]}: {stats}")
        stats["attack"] = str(attack[0])
        dist_barrier()
        if is_main_process():
            save_metrics["test"].append(stats)
            if args.wandb:
                wandb.log(stats)
            logfile.write(json.dumps(stats) + "\n")

    if is_main_process():
        # Save metrics to pickle file if not exists else append
        pkl_path = os.path.join(args.output_dir, "metrics.pkl")
        if os.path.exists(pkl_path):
            metrics = pickle.load(open(pkl_path, "rb"))
            metrics.append(save_metrics)
        else:
            pickle.dump([save_metrics], open(pkl_path, "wb"))

        last_path = f"{args.output_dir}/checkpoint_last.pt"
        if os.path.exists(last_path):
            os.remove(last_path)
        logfile.close()


def _train(train_loader, model, criterion, attack, optimizer, scaler, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    mem = AverageMeter("Mem (GB)", ":6.1f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, mem],
        prefix="Epoch: [{}]".format(epoch),
    )
    compute_acc = get_compute_acc(args)
    seg_only = "seg-only" in args.experiment

    # Switch to train mode
    model.train()

    end = time.time()
    for i, samples in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        if len(samples) == 2:
            images, targets = samples
            segs = None
        elif seg_only:
            # If training segmenter only, `targets` is segmentation mask
            images, targets, _ = samples
            segs = None
        else:
            images, segs, targets = samples
            segs = segs.cuda(args.gpu, non_blocking=True)

        images = images.cuda(args.gpu, non_blocking=True)
        targets = targets.cuda(args.gpu, non_blocking=True)
        batch_size = images.size(0)

        # Compute output
        with amp.autocast(enabled=not args.full_precision):
            if attack.use_mask:
                # Attack for part models where both class and segmentation
                # labels are used
                images = attack(images, targets, segs)
                if attack.dual_losses:
                    targets = torch.cat([targets, targets], axis=0)
                    segs = torch.cat([segs, segs], axis=0)
            else:
                # Attack for either classifier or segmenter alone
                images = attack(images, targets)

            if segs is None or seg_only:
                outputs = model(images)
                loss = criterion(outputs, targets)
            elif "groundtruth" in args.experiment:
                outputs = model(images, segs=segs)
                loss = criterion(outputs, targets)
            else:
                outputs = model(images, return_mask=True)
                loss = criterion(outputs, targets, segs)
                outputs = outputs[0]

            if args.adv_train in ("trades", "mat"):
                outputs = outputs[batch_size:]

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        # Measure accuracy and record loss
        acc1 = compute_acc(outputs, targets)
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)

        # Compute gradient and do SGD step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if i % args.print_freq == 0:
            if is_main_process() and args.wandb:
                wandb.log(
                    {
                        "acc": acc1.item(),
                        "loss": loss.item(),
                        "scaler": scaler.get_scale(),
                    }
                )
            progress.display(i)

    progress.synchronize()
    return {
        "acc1": top1.avg,
        "loss": losses.avg,
        "lr": optimizer.param_groups[0]["lr"],
    }


def _validate(val_loader, model, criterion, attack, args):
    seg_only = "seg-only" in args.experiment
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    pacc = AverageMeter("PixelAcc", ":6.2f")
    mem = AverageMeter("Mem (GB)", ":6.1f")
    iou = AverageMeter("IoU", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses, top1, mem],
        prefix="Test: ",
    )
    compute_acc = get_compute_acc(args)
    compute_iou = IoU(args.seg_labels).cuda(args.gpu)
    
    # switch to evaluate mode
    model.eval()
    count = 0
    end = time.time()
    for i, samples in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if len(samples) == 3:
            images, targets, filenames = samples
            segs = None
        elif seg_only:
            images, targets, _, filenames = samples
            segs = None
        else:
            images, segs, targets, filenames = samples
            segs = segs.cuda(args.gpu, non_blocking=True)

        # DEBUG
        if args.debug:
            save_image(COLORMAP[segs].permute(0, 3, 1, 2), "gt.png")
            save_image(images, "test.png")
        # save_image(COLORMAP[targets].permute(0, 3, 1, 2), "gt.png")
        # save_image(images, "test.png")

        images = images.cuda(args.gpu, non_blocking=True)
        targets = targets.cuda(args.gpu, non_blocking=True)
        batch_size = images.size(0)

        # DEBUG: fixed clean segmentation masks
        if "clean" in args.experiment:
            model(images, clean=True)

        def save_pil_image(img, path):
            image_path = os.path.join(path)
            pil_img = PIL.Image.fromarray(img)
            pil_img.save(image_path)

        # compute output
        with torch.no_grad():
            if attack.use_mask:
                images = attack(images, targets, segs)
            else:
                images = attack(images, targets)

            # Need to duplicate segs and targets to match images expanded by
            # image corruption attack
            if images.shape[0] != targets.shape[0]:
                ratio = images.shape[0] // targets.shape[0]
                targets = targets.repeat((ratio,) + (1,) * (len(targets.shape) - 1))
                if segs:
                    segs = segs.repeat((ratio,) + (1,) * (len(segs.shape) - 1))

            if segs is None or "normal" in args.experiment or seg_only:
                outputs = model(images)
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                # suppress remaining channels
                fill_val = torch.full(outputs.shape[2:], 0)
                for j in range(len(filenames)):
                    name = filenames[j].split("/")[0]
                    channels = class_to_channels[name] + [0]
                    for part_channel in range(outputs[j].shape[0]):
                        if part_channel not in channels:
                            outputs[j][part_channel] = fill_val
                    sum_scores = torch.sum(outputs[j], dim=0)  # 224 by 224
                    target_scores = 1 - outputs[j][0]
                    for c in channels[:-1]:
                        outputs[j][c] = outputs[j][c] * target_scores / sum_scores

                out_masks = outputs.argmax(1)
                for j in range(len(filenames)):
                    count += 1
                    print(count)
                    original_image = Image.open(
                        f"{args.data}/JPEGImages/{filenames[j]}.JPEG"
                    )
                    original_image = transforms.ToTensor()(original_image)
                    # print(original_image.shape)
                    _, H, W = original_image.shape

                    pseudo_mask = out_masks[j].unsqueeze(0)
                    # print(pseudo_mask.shape)
                    pseudo_mask = transforms.functional.resize(
                        pseudo_mask,
                        (H, W),
                        interpolation=transforms.InterpolationMode.NEAREST,
                    ).squeeze(0)
                    # print(pseudo_mask.shape)

                    name = f'{filenames[j].split("/")[1]}.tif'
                    save_pil_image(
                        pseudo_mask.detach().cpu().numpy().astype(np.int16),
                        os.path.join(args.prediction_path, name),
                    )
                    continue

            elif "groundtruth" in args.experiment:
                outputs = model(images, segs=segs)
                loss = criterion(outputs, targets)
            else:
                outputs, masks = model(images, return_mask=True)
                if "centroid" in args.experiment:
                    masks, _, _, _ = masks
                pixel_acc = pixel_accuracy(masks, segs)
                pacc.update(pixel_acc.item(), batch_size)
            loss = criterion(outputs, targets)
        # save_image(
        #     COLORMAP[outputs.argmax(1)].permute(0, 3, 1, 2),
        #     "pred_seg_clean.png",
        # )
        # 0 / 0

        # DEBUG
        # if args.debug and isinstance(attack, PGDAttackModule):
        if args.debug:
            save_image(
                COLORMAP[masks.argmax(1)].permute(0, 3, 1, 2),
                "pred_seg_clean.png",
            )
            print(targets == outputs.argmax(1))
            import pdb

            pdb.set_trace()

        # measure accuracy and record loss
        acc1 = compute_acc(outputs, targets)
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        if seg_only:
            iou.update(compute_iou(outputs, targets).item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if i % args.print_freq == 0:
            progress.display(i)

    # TODO: this should also be done with the ProgressMeter
    progress.synchronize()
    print(f" * Acc@1 {top1.avg:.3f}")

    if pacc.count > 0:
        pacc.synchronize()
        print(f"Pixelwise accuracy: {pacc.avg:.4f}")
    if seg_only:
        iou.synchronize()
        print(f"IoU: {iou.avg:.4f}")
    return {"acc1": top1.avg, "loss": losses.avg, "pixel-acc": pacc.avg}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Part Classification", parents=[_get_args_parser()]
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
