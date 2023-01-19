#!/usr/bin/env python
# The template of this code is generously provided by Norman Mu (@normster)
# The original version is from https://github.com/pytorch/examples/blob/master/imagenet/main.py
import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import wandb
from torch.distributed.elastic.multiprocessing.errors import record

# from torchmetrics import IoU as IoU   # Use this for older version of torchmetrics
from torchmetrics import JaccardIndex as IoU
from torchvision.utils import save_image

from part_model.attack import setup_eval_attacker
from part_model.dataloader import COLORMAP, load_dataset
from part_model.models import build_model
from part_model.utils import (
    AverageMeter,
    ProgressMeter,
    dist_barrier,
    get_compute_acc,
    get_rank,
    init_distributed_mode,
    is_main_process,
    pixel_accuracy,
)
from part_model.utils.argparse import get_args_parser
from part_model.utils.loss import get_train_criterion

best_acc1 = 0


def main():
    init_distributed_mode(args)

    global best_acc1

    # Fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data loading code
    print("=> creating dataset")
    loaders = load_dataset(args)
    _, _, _, test_loader = loaders

    # Create model
    print("=> creating model")
    model, _, _ = build_model(args)
    cudnn.benchmark = True

    # Define loss function
    criterion, _ = get_train_criterion(args)

    # Load model weights
    if args.resume:
        load_path = args.resume
        print(f"=> loading resume checkpoint {load_path}")
    else:
        load_path = f"{args.output_dir}/checkpoint_best.pt"
        print(f"=> loading best checkpoint {load_path}")
    if args.gpu is None:
        checkpoint = torch.load(load_path)
    else:
        # Map model to be loaded to specified single gpu.
        loc = "cuda:{}".format(args.gpu)
        checkpoint = torch.load(load_path, map_location=loc)
    model.load_state_dict(checkpoint["state_dict"])

    # Logging
    if is_main_process():
        logfile = open(os.path.join(args.output_dir, "log.txt"), "a")
        logfile.write(str(args) + "\n")
        logfile.flush()
        if args.wandb:
            wandb_id = os.path.split(args.output_dir)[-1]
            wandb.init(
                project="part-model", id=wandb_id, config=args, resume="allow"
            )
            print("wandb step:", wandb.run.step)

    model.eval()
    eval_attack = setup_eval_attacker(args, model, guide_dataloader=test_loader)

    for attack in eval_attack:
        # Use DataParallel (not distributed) model for AutoAttack.
        # Otherwise, DDP model can get timeout or c10d failure.
        stats = validate(test_loader, model, criterion, attack[1])
        print(f"=> {attack[0]}: {stats}")
        stats["attack"] = str(attack[0])
        dist_barrier()
        if is_main_process():
            if args.wandb:
                wandb.log(stats)
            logfile.write(json.dumps(stats) + "\n")

    if is_main_process():
        # Save metrics to pickle file if not exists else append
        pkl_path = os.path.join(args.output_dir, "metrics.pkl")
        if os.path.exists(pkl_path):
            metrics = pickle.load(open(pkl_path, "rb"))
            metrics.append(stats)
        else:
            pickle.dump([stats], open(pkl_path, "wb"))
        logfile.close()


def validate(val_loader, model, criterion, attack):
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

    end = time.time()
    for i, samples in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if len(samples) == 2:
            images, targets = samples
            segs = None
        elif seg_only:
            images, targets, _ = samples
            segs = None
        else:
            images, segs, targets = samples
            segs = segs.cuda(args.gpu, non_blocking=True)

        # DEBUG
        if args.debug:
            save_image(COLORMAP[segs].permute(0, 3, 1, 2), "gt.png")
            save_image(images, "test.png")

        images = images.cuda(args.gpu, non_blocking=True)
        targets = targets.cuda(args.gpu, non_blocking=True)
        batch_size = images.size(0)

        # DEBUG: fixed clean segmentation masks
        if "clean" in args.experiment:
            model(images, clean=True)

        # compute output
        with torch.no_grad():
            # Run attack end-to-end using the obtained attack mask
            if attack.use_mask:
                images = attack(images, targets, segs)
            else:
                images = attack(images, targets)

            # Need to duplicate segs and targets to match images expanded by
            # image corruption attack
            if images.shape[0] != targets.shape[0]:
                ratio = images.shape[0] // targets.shape[0]
                targets = targets.repeat(
                    (ratio,) + (1,) * (len(targets.shape) - 1)
                )
                segs = segs.repeat((ratio,) + (1,) * (len(segs.shape) - 1))

            if segs is None or "normal" in args.experiment or seg_only:
                outputs = model(images)
            elif "groundtruth" in args.experiment:
                outputs = model(images, segs=segs)
                loss = criterion(outputs, targets)
            else:
                outputs, masks = model(images, return_mask=True)
                pixel_acc = pixel_accuracy(masks, segs)
                pacc.update(pixel_acc.item(), batch_size)
            loss = criterion(outputs, targets)

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
        "Part Classification", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main()
