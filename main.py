"""Main script for both training and evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import random
import sys
import time
from pprint import pprint
from typing import Any

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import wandb
from torch.backends import cudnn
from torch.cuda import amp

# Use this for older version of torchmetrics. torchmetrics <= 0.6.0 is
# supposedly a lot faster at computing mAP scores.
# https://github.com/Lightning-AI/metrics/pull/1389
# from torchmetrics import IoU
# from torchmetrics.detection import MAP as MeanAveragePrecision
from torchmetrics.classification import MulticlassJaccardIndex as IoU
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from torchvision.ops import box_convert
from torchvision.utils import save_image

from DINO.models.dino.dino import PostProcess
from DINO.util.slconfig import SLConfig
from part_model.attack import (
    setup_eval_attacker,
    setup_train_attacker,
    setup_val_attacker,
)
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
from part_model.utils.argparse import get_args_parser
from part_model.utils.dataloader_visualizer import debug_dino_dataloader
from part_model.utils.loss import get_train_criterion


def _dummy_compute_acc(outputs, targets):
    _ = outputs, targets  # Unused
    return torch.zeros(1)


def _write_metrics(save_metrics: Any) -> None:
    if is_main_process():
        # Save metrics to pickle file if not exists else append
        pkl_path = os.path.join(args.output_dir, "metrics.pkl")
        with open(pkl_path, "wb") as file:
            pickle.dump([save_metrics], file)


def main() -> None:
    """Main function."""
    init_distributed_mode(args)
    # TODO(chawins@): Have to change this when adding new detection models
    use_det = "seg-only" in args.experiment and args.obj_det_arch == "dino"

    # Fix the seed for reproducibility
    seed: int = args.seed + get_rank()
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Data loading code
    print("=> Creating dataset...")
    loaders = load_dataset(args)
    train_loader, train_sampler, val_loader, test_loader = loaders

    # Debugging dataloader
    if args.debug:
        # TODO(nab-126@): What is this debugging code for? Is it still needed?
        debug_dino_dataloader(train_loader)
        # debug_dino_dataloader(val_loader)
        # debug_dino_dataloader(test_loader)

    # Create model
    print("=> Creating model...")
    model, optimizer, scaler = build_model(args)

    # Define loss function
    criterion, train_criterion = get_train_criterion(args)

    # Logging
    if is_main_process():
        log_path: str = os.path.join(
            os.path.expanduser(args.output_dir), "log.txt"
        )
        logfile = open(log_path, "a", encoding="utf-8")
        logfile.write(str(args) + "\n")
        logfile.flush()
        if args.wandb:
            wandb_id = os.path.split(args.output_dir)[-1]
            wandb.init(
                project="part-model", id=wandb_id, config=args, resume="allow"
            )
            print("wandb step:", wandb.run.step)

    eval_attack = setup_eval_attacker(args, model)
    no_attack = eval_attack[0][1]
    train_attack = setup_train_attacker(args, model)
    val_attack = setup_val_attacker(args, model)
    save_metrics = {"train": [], "test": []}

    print(args)

    if not args.evaluate:
        print("=> Beginning training...")
        best_acc = 0
        val_stats = {}
        for epoch in range(args.start_epoch, args.epochs):
            is_best = False
            if args.distributed:
                train_sampler.set_epoch(epoch)
            learning_rate = adjust_learning_rate(optimizer, epoch, args)
            print(f"=> lr @ epoch {epoch}: {learning_rate:.2e}")

            # Train for one epoch
            train_stats = _train(
                train_loader,
                model,
                train_criterion,
                train_attack,
                optimizer,
                scaler,
                epoch,
            )

            if (epoch + 1) % 2 == 0:
                val_stats = _validate(val_loader, model, criterion, no_attack)
                metric_name = "map" if use_det else "acc1"
                clean_acc1, acc1 = val_stats[metric_name], None
                is_best = clean_acc1 > best_acc

                if args.adv_train != "none":
                    adv_val_stats = _validate(
                        val_loader, model, criterion, val_attack
                    )
                    acc1 = adv_val_stats[metric_name]
                    val_stats["adv_acc1"] = acc1
                    val_stats["adv_loss"] = adv_val_stats["loss"]
                    is_best = clean_acc1 >= acc1 > best_acc

                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_acc1": best_acc,
                    "args": args,
                }

                if is_best:
                    print("=> Saving new best checkpoint...")
                    save_on_master(save_dict, args.output_dir, is_best=True)
                    best_acc = (
                        max(clean_acc1, best_acc)
                        if acc1 is None
                        else max(acc1, best_acc)
                    )
                save_epoch = epoch + 1 if args.save_all_epochs else None
                save_on_master(
                    save_dict, args.output_dir, is_best=False, epoch=save_epoch
                )

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                "epoch": epoch,
            }

            if is_main_process():
                save_metrics["train"].append(log_stats)
                _write_metrics(save_metrics)
                if args.wandb:
                    wandb.log(log_stats)
                logfile.write(json.dumps(log_stats) + "\n")
                logfile.flush()

        # Compute stats of best model after training
        dist_barrier()
        load_path = f"{args.output_dir}/checkpoint_best.pt"
        print(f"=> Loading checkpoint from {load_path}...")
        # Map model to be loaded to specified single gpu.
        checkpoint = torch.load(
            load_path,
            map_location=None if args.gpu is None else f"cuda:{args.gpu}",
        )
        model.load_state_dict(checkpoint["state_dict"])

    # Running evaluation
    for attack in eval_attack:
        # Use DataParallel (not distributed) model for AutoAttack.
        # Otherwise, DDP model can get timeout or c10d failure.
        stats = _validate(test_loader, model, criterion, attack[1])
        print(f"=> {attack[0]}: {stats}")
        stats["attack"] = str(attack[0])
        dist_barrier()
        if is_main_process():
            save_metrics["test"].append(stats)
            _write_metrics(save_metrics)
            if args.wandb:
                wandb.log(stats)
            logfile.write(json.dumps(stats) + "\n")

    if is_main_process():
        # Save metrics to pickle file if not exists else append
        _write_metrics(save_metrics)
        last_path = f"{args.output_dir}/checkpoint_last.pt"
        if os.path.exists(last_path):
            os.remove(last_path)
        logfile.close()


def _train(train_loader, model, criterion, attack, optimizer, scaler, epoch):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    mem = AverageMeter("Mem (GB)", ":6.1f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, mem],
        prefix=f"Epoch: [{epoch}]",
    )
    compute_acc = get_compute_acc(args)
    seg_only = "seg-only" in args.experiment
    if args.obj_det_arch == "dino" and seg_only:
        compute_acc = _dummy_compute_acc

    # Switch to train mode
    model.train()

    segs, masks, target_bbox = None, None, None
    need_tgt_for_training = True

    end = time.time()
    for i, samples in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        # TODO(nabeel@): Ideally we want to unify the data handling for both
        # DINO and segmentation, or otherwise, make a separate training script.
        if args.obj_det_arch == "dino":
            nested_tensors, target_bbox, targets = samples
            images, masks = nested_tensors.decompose()
            images = images.cuda(args.gpu, non_blocking=True)
            masks = masks.cuda(args.gpu, non_blocking=True)
            targets = torch.tensor(
                targets, device=masks.device, dtype=torch.long
            )
            target_bbox = [
                {k: v.cuda(args.gpu, non_blocking=True) for k, v in t.items()}
                for t in target_bbox
            ]
        else:
            # If training segmenter only, `targets` is segmentation mask
            images, targets = samples[:2]
            if len(samples) == 3 and not seg_only:
                images, segs, targets = samples
                segs = segs.cuda(args.gpu, non_blocking=True)
            images = images.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)

        batch_size: int = targets.size(0)
        with amp.autocast(enabled=not args.full_precision):
            if args.obj_det_arch == "dino":
                forward_args = {
                    "masks": masks,
                    "dino_targets": target_bbox,
                    "need_tgt_for_training": need_tgt_for_training,
                    "return_mask": False,
                    "return_mask_only": seg_only,
                }
                if seg_only:
                    # TODO(nab-126@): target_bbox is being passed in
                    # forward_args already. Do we still need it as arg here?
                    images = attack(images, target_bbox, **forward_args)
                else:
                    images = attack(images, targets, **forward_args)

                if args.adv_train in ("trades", "mat"):
                    masks = torch.cat([masks.detach(), masks.detach()], dim=0)
                    target_bbox = [*target_bbox, *target_bbox]

                if seg_only:
                    outputs = model(images, **forward_args)
                    loss = criterion(outputs, target_bbox)
                else:
                    # Change to true to get dino outputs for map calculation
                    forward_args["return_mask"] = True
                    outputs, dino_outputs = model(images, **forward_args)
                    loss = criterion(
                        outputs, dino_outputs, target_bbox, targets
                    )

                if args.adv_train in ("trades", "mat"):
                    outputs = outputs[batch_size:]
            else:
                images = attack(images, targets, seg_targets=segs)
                if attack.dual_losses:
                    targets = torch.cat([targets, targets], axis=0)
                    segs = torch.cat([segs, segs], axis=0)

                # TODO(chawins@): unify model interface
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
            print(f"Loss is {loss.item()}, stopping training")
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


def _validate(val_loader, model, criterion, attack):
    seg_only = "seg-only" in args.experiment
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    pacc = AverageMeter("PixelAcc", ":6.2f")
    mem = AverageMeter("Mem (GB)", ":6.1f")
    iou = AverageMeter("IoU", ":6.2f")
    mAP = AverageMeter("mAP", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses, top1, mem],
        prefix="Test: ",
    )
    compute_acc = get_compute_acc(args)
    compute_iou = IoU(args.seg_labels).cuda(args.gpu)
    if (args.obj_det_arch == "dino" and seg_only) or args.calculate_map:
        if seg_only:
            compute_acc = _dummy_compute_acc
        map_metric = MeanAveragePrecision().cuda(args.gpu)
        postprocessors = {
            "bbox": PostProcess(
                num_select=args.num_select,
                nms_iou_threshold=args.nms_iou_threshold,
            ).cuda(args.gpu)
        }

    # switch to evaluate mode
    model.eval()
    need_tgt_for_training = True
    end = time.time()

    for i, samples in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if len(samples) == 2:
            images, targets = samples
            segs = None
        elif seg_only and args.obj_det_arch != "dino":
            images, targets, _ = samples
            segs = None
        else:
            # handling dino validation
            if args.obj_det_arch == "dino":
                nested_tensors, target_bbox, targets = samples
                images, masks = nested_tensors.decompose()
                masks = masks.cuda(args.gpu, non_blocking=True)
                targets = torch.tensor(
                    targets, device=masks.device, dtype=torch.long
                )
                target_bbox = [
                    {
                        k: v.cuda(args.gpu, non_blocking=True)
                        for k, v in t.items()
                    }
                    for t in target_bbox
                ]
            else:
                images, segs, targets = samples
                segs = segs.cuda(args.gpu, non_blocking=True)

        # DEBUG
        if args.debug:
            save_image(COLORMAP[segs.cpu()].permute(0, 3, 1, 2), "gt.png")
            save_image(images, "test.png")

        images = images.cuda(args.gpu, non_blocking=True)
        targets = targets.cuda(args.gpu, non_blocking=True)
        batch_size = targets.size(0)

        # DEBUG: fixed clean segmentation masks
        if "clean" in args.experiment:
            model(images, clean=True)

        # compute output
        with torch.no_grad():
            if args.obj_det_arch == "dino":
                forward_args = {
                    "masks": masks,
                    "dino_targets": target_bbox,
                    "need_tgt_for_training": need_tgt_for_training,
                    "return_mask": False,
                    "return_mask_only": seg_only,
                }

                # TODO(chawins@): This if-else should be merged and use
                # return_mask_only to control the output
                if seg_only:
                    images = attack(images, target_bbox, **forward_args)
                    dino_outputs = model(images, **forward_args)
                    loss = criterion(dino_outputs, target_bbox)
                else:
                    images = attack(images, targets, **forward_args)
                    # Change to true to get dino outputs for map calculation
                    forward_args["return_mask"] = True
                    outputs, dino_outputs = model(images, **forward_args)
                    loss = criterion(outputs, targets)

                if seg_only or args.calculate_map:
                    orig_target_sizes = torch.stack(
                        [t["orig_size"] for t in target_bbox], dim=0
                    )
                    results = postprocessors["bbox"](
                        dino_outputs, orig_target_sizes
                    )

                    # target_bbox_copy = copy.deepcopy(targets)
                    for j, tbox in enumerate(target_bbox):
                        shape = tbox["orig_size"]
                        boxes = tbox["boxes"]
                        boxes = box_convert(
                            boxes, in_fmt="cxcywh", out_fmt="xyxy"
                        )
                        boxes[:, ::2] = boxes[:, ::2] * shape[1]
                        boxes[:, 1::2] = boxes[:, 1::2] * shape[0]
                        target_bbox[j]["boxes"] = boxes

                    map_metric.update(results, target_bbox)

            else:
                images = attack(images, targets, seg_targets=segs)

                # Need to duplicate segs and targets to match images expanded by
                # image corruption attack
                if images.shape[0] != targets.shape[0]:
                    ratio = images.shape[0] // targets.shape[0]
                    targets = targets.repeat(
                        (ratio,) + (1,) * (len(targets.shape) - 1)
                    )
                    if segs:
                        segs = segs.repeat(
                            (ratio,) + (1,) * (len(segs.shape) - 1)
                        )

                if segs is None or "normal" in args.experiment or seg_only:
                    outputs = model(images)
                elif "groundtruth" in args.experiment:
                    outputs = model(images, segs=segs)
                else:
                    outputs, masks = model(images, return_mask=True)
                    if "centroid" in args.experiment:
                        masks, _, _, _ = masks
                    pixel_acc = pixel_accuracy(masks, segs)
                    pacc.update(pixel_acc.item(), batch_size)
                loss = criterion(outputs, targets)

        # DEBUG
        if args.debug:
            save_image(
                COLORMAP[masks.argmax(1).cpu()].permute(0, 3, 1, 2),
                "pred_seg_clean.png",
            )
            print(targets == outputs.argmax(1))
            raise NotImplementedError("End of debugging. Exit.")

        # measure accuracy and record loss
        acc1 = compute_acc(outputs, targets)
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        if seg_only and args.obj_det_arch != "dino":
            iou.update(compute_iou(outputs, targets).item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if i % args.print_freq == 0:
            progress.display(i)

    progress.synchronize()
    print(f" * Acc@1 {top1.avg:.3f}")

    return_dict = {
        "acc1": top1.avg,
        "loss": losses.avg,
        "pixel-acc": pacc.avg,
    }

    if pacc.count > 0:
        pacc.synchronize()
        print(f"Pixelwise accuracy: {pacc.avg:.4f}")
    if seg_only or args.calculate_map:
        if args.obj_det_arch == "dino":
            print(" * mAP metric")
            map_dict = map_metric.compute()
            mAP.update(map_dict["map"].item(), 1)
            mAP.synchronize()
            # Average mAP across workers in DDP. This is not strictly correct,
            # but should be good enough for logging. Not sure if there is a
            # correct way: https://github.com/Lightning-AI/metrics/issues/53.
            print(f"mAP: {mAP.avg:.4f}")
            return_dict["map"] = mAP.avg
            pprint({k: v.item() for k, v in map_dict.items()})
        else:
            iou.synchronize()
            print(f"IoU: {iou.avg:.4f}")
            return_dict["iou"] = iou.avg

    return return_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Part Classification", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    # TODO(nab-126@): add to argparser?
    # handling dino args
    if args.config_file:
        cfg = SLConfig.fromfile(args.config_file)

        if args.options is not None:
            cfg.merge_from_dict(args.options)

        cfg_dict = cfg._cfg_dict.to_dict()
        args_vars = vars(args)
        for k, v in cfg_dict.items():
            if k not in args_vars:
                setattr(args, k, v)
            else:
                raise ValueError(f"Key {k} can used by args only")

    # Set logging config
    logging.basicConfig(
        stream=sys.stdout,
        format="[%(asctime)s - %(name)s - %(levelname)s]: %(message)s",
        level=logging.DEBUG if args.debug else logging.INFO,
        force=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    main()
