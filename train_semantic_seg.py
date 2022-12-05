#!/usr/bin/env python
# The template of this code is generously provided by Norman Mu (@normster)
# The original version is from https://github.com/pytorch/examples/blob/master/imagenet/main.py
import argparse
import json
import math
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import wandb
from torchmetrics import IoU

from part_model.attack import (
    setup_eval_attacker,
    setup_train_attacker,
    setup_val_attacker,
)
from part_model.dataloader import load_dataset
from part_model.models import build_segmentation
from part_model.utils import (
    AverageMeter,
    ProgressMeter,
    TRADESLoss,
    adjust_learning_rate_deeplabv3,
    get_rank,
    init_distributed_mode,
    is_main_process,
    pixel_accuracy,
    save_on_master,
)
from part_model.utils.argparse import get_args_parser

best_acc = 0


def main():
    init_distributed_mode(args)

    global best_acc

    # Fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data loading code
    print("=> creating dataset")
    loaders = load_dataset(args)
    if len(loaders) == 4:
        train_loader, train_sampler, val_loader, test_loader = loaders
    else:
        train_loader, train_sampler, val_loader = loaders
        test_loader = val_loader

    # Create model
    print("=> creating model")
    model, optimizer, scaler = build_segmentation(args)
    cudnn.benchmark = True

    # Define loss function
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    compute_iou = IoU(args.seg_labels).cuda(args.gpu)
    train_criterion = criterion
    if args.adv_train == "trades":
        train_criterion = TRADESLoss(args.trades_beta).cuda(args.gpu)

    if is_main_process() and args.wandb:
        wandb_id = os.path.split(args.output_dir)[-1]
        wandb.init(
            project="part-model", id=wandb_id, config=args, resume="allow"
        )
        print("wandb step:", wandb.run.step)

    eval_attack = setup_eval_attacker(args, model, num_classes=args.seg_labels)
    no_attack = eval_attack[0][1]
    train_attack = setup_train_attacker(args, model)
    val_attack = setup_val_attacker(args, model)

    print(args)

    if not args.evaluate:
        print("=> beginning training")
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)

            # train for one epoch
            train_stats = train(
                train_loader,
                model,
                train_criterion,
                train_attack,
                optimizer,
                scaler,
                epoch,
            )
            val_stats = validate(
                val_loader,
                model,
                criterion,
                compute_iou,
                no_attack,
            )
            acc, clean_acc = val_stats["acc"], val_stats["acc"]
            if args.adv_train == "pgd":
                val_stats = validate(
                    val_loader,
                    model,
                    criterion,
                    compute_iou,
                    val_attack,
                )
                acc = val_stats["acc"]

            is_best = acc > best_acc and clean_acc >= acc
            best_acc = max(acc, best_acc)

            if is_best:
                print("=> Saving new best checkpoint")
                save_on_master(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "best_acc": best_acc,
                        "args": args,
                    },
                    args.output_dir,
                )

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in val_stats.items()},
                "epoch": epoch,
            }

            if is_main_process():
                if args.wandb:
                    wandb.log(log_stats)
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

    # Compute stats of best model
    best_path = f"{args.output_dir}/checkpoint_best.pt"
    print(f"=> loading best checkpoint {best_path}")
    if args.gpu is None:
        checkpoint = torch.load(best_path)
    else:
        # Map model to be loaded to specified single gpu.
        loc = "cuda:{}".format(args.gpu)
        checkpoint = torch.load(best_path, map_location=loc)
    model.load_state_dict(checkpoint["state_dict"])
    for attack in eval_attack:
        stats = validate(
            test_loader, model, criterion, compute_iou, attack[1], args
        )
        print(f"=> {attack[0]}: {stats}")


def train(
    train_loader,
    model,
    criterion,
    attack,
    optimizer,
    scaler,
    epoch,
):
    num_batches = len(train_loader)
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc", ":6.2f")
    mem = AverageMeter("Mem (GB)", ":6.1f")
    progress = ProgressMeter(
        num_batches,
        [batch_time, data_time, losses, top1, mem],
        prefix="Epoch: [{}]".format(epoch),
    )

    # Switch to train mode
    model.train()

    end = time.time()
    for i, (images, targets) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        adjust_learning_rate_deeplabv3(optimizer, epoch, i, num_batches, args)

        images = images.cuda(args.gpu, non_blocking=True)
        targets = targets.cuda(args.gpu, non_blocking=True)

        # Compute output
        with amp.autocast(enabled=not args.full_precision):
            images = attack(images, targets)
            outputs = model(images)
            loss = criterion(outputs, targets)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        # Measure accuracy and record loss
        acc = pixel_accuracy(outputs, targets)
        losses.update(loss.item(), images.size(0))
        top1.update(acc.item(), images.size(0))

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
                        "acc": acc.item(),
                        "loss": loss.item(),
                        "scaler": scaler.get_scale(),
                    }
                )
            progress.display(i)

    progress.synchronize()
    return {
        "acc": top1.avg,
        "loss": losses.avg,
        "lr": optimizer.param_groups[0]["lr"],
    }


def validate(val_loader, model, criterion, compute_iou, attack):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc", ":6.2f")
    iou = AverageMeter("IoU", ":6.2f")
    mem = AverageMeter("Mem (GB)", ":6.1f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses, top1, mem],
        prefix="Test: ",
    )

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (images, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.cuda(args.gpu, non_blocking=True)
        targets = targets.cuda(args.gpu, non_blocking=True)

        # compute output
        with torch.no_grad():
            images = attack(images, targets)
            outputs = model(images)
            loss = criterion(outputs, targets)

        # DEBUG
        # from torchvision.utils import save_image
        # from part_model.dataloader.pascal_part import COLORMAP
        # save_image(COLORMAP[targets].permute(0, 3, 1, 2), 'gt.png')
        # save_image(COLORMAP[outputs.argmax(1)].permute(0, 3, 1, 2), 'pred_seg.png')
        # save_image(images, 'test.png')
        # import pdb
        # pdb.set_trace()

        # measure accuracy and record loss
        acc = pixel_accuracy(outputs, targets)
        losses.update(loss.item(), images.size(0))
        top1.update(acc.item(), images.size(0))
        iou.update(compute_iou(outputs, targets).item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if i % args.print_freq == 0:
            progress.display(i)

    print(f">>> Acc {top1.avg:.3f}   IoU {iou.avg:.3f}")

    progress.synchronize()
    return {"acc": top1.avg, "loss": losses.avg}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Part segmentation model", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main()
