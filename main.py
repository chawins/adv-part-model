"""Main script for both training and evaluation."""

from __future__ import annotations

import argparse
import json
import math
import numpy as np
import os
import pickle
# from pycocotools.cocoeval import COCOeval
import random
import sys
import time
from typing import Any

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import wandb
from torch.backends import cudnn
from torch.cuda import amp

from torchmetrics import IoU as IoU   # Use this for older version of torchmetrics
# from torchmetrics.classification import MulticlassJaccardIndex as IoU
# from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.detection import MAP as MeanAveragePrecision
from torchvision.datasets import CocoDetection
from torchvision.utils import save_image

# from DINO.datasets.coco_eval import CocoEvaluator
from DINO.models.dino.dino import PostProcess
from DINO.util.slconfig import SLConfig
from part_model.attack import (
    setup_eval_attacker,
    setup_train_attacker,
    setup_val_attacker,
)
from part_model.dataloader.coco_eval import CocoEvaluator
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
from part_model.utils.image import unnormalize_bbox_targets, get_bbox_from_masks, get_masks_from_bbox, plot_img_bbox
from part_model.utils.loss import get_train_criterion

BEST_ACC = 0
BEST_LOSS = float("inf")

def _write_metrics(save_metrics: Any) -> None:
    if is_main_process():
        # Save metrics to pickle file if not exists else append
        pkl_path = os.path.join(args.output_dir, "metrics.pkl")
        with open(pkl_path, "wb") as file:
            pickle.dump([save_metrics], file)


def main() -> None:
    """Main function."""
    init_distributed_mode(args)

    global BEST_ACC
    global BEST_LOSS

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
    train_loader, train_sampler, val_loader, test_loader, train_dataset, val_dataset, test_dataset = loaders

    # Debugging dataloader
    if args.debug:
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
            # wandb_id = os.path.split(args.output_dir)[-1]
            wandb_id = '_'.join(args.output_dir.split('/'))
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

                # TODO: clean/unify
                if 'seg-only' in args.experiment and args.obj_det_arch == 'dino':
                    # clean_acc1, acc1 = val_stats["map"], None
                    # is_best = clean_acc1 > BEST_ACC
                    clean_loss1, acc1 = val_stats["loss"], None
                    adv_loss1 = None
                    is_best = clean_loss1 < BEST_LOSS
                else:
                    clean_acc1, acc1 = val_stats["acc1"], None
                    is_best = clean_acc1 > BEST_ACC

                if args.adv_train != "none":
                    adv_val_stats = _validate(
                        val_loader, model, criterion, val_attack
                    )
                    # TODO: clean/unify
                    if 'seg-only' in args.experiment and args.obj_det_arch == 'dino':
                        # acc1 = adv_val_stats["map"]
                        # val_stats["adv_acc1"] = acc1
                        # val_stats["adv_loss"] = adv_val_stats["loss"]
                        # is_best = clean_acc1 >= acc1 > BEST_ACC

                        adv_loss1 = val_stats["loss"]
                        val_stats["adv_acc1"] = acc1
                        val_stats["adv_loss"] = adv_val_stats["loss"]
                        is_best = clean_loss1 <= adv_loss1 < BEST_LOSS
                    else:
                        acc1 = adv_val_stats["acc1"]
                        val_stats["adv_acc1"] = acc1
                        val_stats["adv_loss"] = adv_val_stats["loss"]
                        is_best = clean_acc1 >= acc1 > BEST_ACC

                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_acc1": BEST_ACC,
                    "args": args,
                }

                if is_best:
                    print("=> Saving new best checkpoint...")
                    save_on_master(save_dict, args.output_dir, is_best=True)
                    if 'seg-only' in args.experiment and args.obj_det_arch == 'dino':
                        # BEST_ACC = (
                        #     max(clean_acc1, BEST_ACC)
                        #     if acc1 is None
                        #     else max(acc1, BEST_ACC)
                        # )
                        # print('best map', BEST_ACC)
                        if adv_loss1 is None:
                            BEST_LOSS = min(BEST_LOSS, clean_loss1)
                        else:
                            BEST_LOSS = min(BEST_LOSS, adv_loss1)
                        print('BEST_LOSS', BEST_LOSS)
                    else:
                        BEST_ACC = (
                            max(clean_acc1, BEST_ACC)
                            if acc1 is None
                            else max(acc1, BEST_ACC)
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
        stats = _validate(test_loader, model, criterion, attack[1], test_dataset)
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
    if args.obj_det_arch == 'dino' and seg_only:
        compute_acc = lambda x, y: torch.tensor(0) # dummy

    # Switch to train mode
    model.train()

    target_segs, masks, target_bbox = None, None, None
    need_tgt_for_training = True

    end = time.time()
    for i, samples in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        images, target_segs, target_bbox, targets = samples

        if seg_only:
            targets = target_segs
            target_segs = None
        else:
            target_segs = torch.stack(target_segs, dim=0) # convert images from list to tensor
            target_segs = target_segs.cuda(args.gpu, non_blocking=True)
    
        # load to GPU  
        images = torch.stack(images, dim=0) # convert images from list to tensor
        images = images.cuda(args.gpu, non_blocking=True)
        target_bbox = [{k: v.cuda(args.gpu, non_blocking=True) for k, v in t.items()} for t in target_bbox]
        targets = torch.tensor(targets) if isinstance(targets[0], int) else torch.stack(targets, dim=0)# convert targets from list to tensor
        targets = targets.cuda(args.gpu, non_blocking=True)

        batch_size = images.shape[0]

        with amp.autocast(enabled=not args.full_precision):
            if args.obj_det_arch == "dino":
                masks = torch.zeros(batch_size, images.shape[2], images.shape[3])
                forward_args = {
                    "masks": masks,
                    "dino_targets": target_bbox,
                    "need_tgt_for_training": need_tgt_for_training,
                    "return_mask": False,
                    "return_mask_only": seg_only,
                }
                if seg_only:
                    images = attack(images, target_bbox, **forward_args)
                else:
                    images = attack(images, targets, **forward_args)
                
                if args.adv_train in ("trades", "mat"):
                    masks = torch.cat([masks.detach(), masks.detach()], dim=0)
                    target_bbox = [*target_bbox, *target_bbox]
                

                if seg_only:
                    dino_outputs = model(images, **forward_args)        
                    loss = criterion(dino_outputs, target_bbox)
                    outputs = dino_outputs
                else:
                    forward_args[
                        "return_mask"
                    ] = True  # change to true to get dino outputs for map calculation
                    outputs, dino_outputs = model(images, **forward_args)                
                    loss = criterion(outputs, dino_outputs, target_bbox, targets)

                if args.adv_train in ("trades", "mat"):
                    outputs = outputs[batch_size:]

            else:
                images = attack(images, targets)
                if attack.dual_losses:
                    targets = torch.cat([targets, targets], axis=0)
                    target_segs = torch.cat([target_segs, target_segs], axis=0)

                # TODO(chawins@): unify model interface
                if target_segs is None or seg_only:
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                elif "groundtruth" in args.experiment:
                    outputs = model(images, segs=target_segs)
                    loss = criterion(outputs, targets)
                else:
                    outputs = model(images, return_mask=True)
                    loss = criterion(outputs, targets, target_segs)
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


def _validate(val_loader, model, criterion, attack, val_dataset=None):
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
    metrics_dict = {}
    compute_acc = get_compute_acc(args)
    compute_iou = IoU(args.seg_labels).cuda(args.gpu)

    if (args.obj_det_arch == 'dino' and seg_only) or args.calculate_map:
        map_metric = MeanAveragePrecision() # box_format='xyxy'
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

        images, target_segs, target_bbox, targets = samples

        if seg_only:
            targets = target_segs
            target_segs = None
        else:
            target_segs = torch.stack(target_segs, dim=0) # convert images from list to tensor
            target_segs = target_segs.cuda(args.gpu, non_blocking=True)

        # DEBUG
        if args.debug:
            save_image(COLORMAP[target_segs.cpu()].permute(0, 3, 1, 2), "gt.png")
            save_image(images, "test.png")

        # load to GPU  
        images = torch.stack(images, dim=0) # convert images from list to tensor
        images = images.cuda(args.gpu, non_blocking=True)
        target_bbox = [{k: v.cuda(args.gpu, non_blocking=True) for k, v in t.items()} for t in target_bbox]
        targets = torch.tensor(targets) if isinstance(targets[0], int) else torch.stack(targets, dim=0)# convert targets from list to tensor
        targets = targets.cuda(args.gpu, non_blocking=True)

        batch_size = images.shape[0]

        # DEBUG: fixed clean segmentation masks
        if "clean" in args.experiment:
            model(images, clean=True)

        # compute output
        with torch.no_grad():
            if args.obj_det_arch == "dino":
                masks = torch.zeros(batch_size, images.shape[2], images.shape[3])
                forward_args = {
                    "masks": masks,
                    "dino_targets": target_bbox,
                    "need_tgt_for_training": need_tgt_for_training,
                    "return_mask": False,
                    "return_mask_only": seg_only
                }

                if seg_only:
                    images = attack(images, target_bbox, **forward_args)
                    dino_outputs = model(images, **forward_args)        
                    loss = criterion(dino_outputs, target_bbox)
                    outputs = dino_outputs
                else:
                    images = attack(images, targets, **forward_args)
                    forward_args[
                        "return_mask"
                    ] = True  # change to true to get dino outputs for map calculation
                    outputs, dino_outputs = model(images, **forward_args)                
                    loss = criterion(outputs, targets)

                if seg_only or args.calculate_map:
                    orig_target_sizes = torch.stack([t["size"] for t in target_bbox], dim=0)
                    pred_bbox = postprocessors["bbox"](
                        dino_outputs, orig_target_sizes
                    ) 

                    outputs = get_masks_from_bbox(pred_bbox, orig_target_sizes, args.seg_labels)
                    target_bbox = unnormalize_bbox_targets(target_bbox)
                    map_metric.update(pred_bbox, target_bbox)
                    pixel_acc = pixel_accuracy(outputs, targets)
                    pacc.update(pixel_acc.item(), batch_size)
                
                    if args.debug:  
                        plot_img_bbox(images, target_bbox)
                        plot_img_bbox(images, pred_bbox)
                        save_image(COLORMAP[outputs.argmax(1).cpu()].permute(0, 3, 1, 2), "test_preds.png")
                        save_image(COLORMAP[targets.cpu()].permute(0, 3, 1, 2), "test_masks.png")
                        plot_img_bbox(COLORMAP[targets.cpu()].permute(0, 3, 1, 2), target_bbox)
                        import pdb; pdb.set_trace()

            else:
                # images = attack(images, targets, target_segs=target_segs)
                images = attack(images, targets)

                # Need to duplicate segs and targets to match images expanded by
                # image corruption attack
                if images.shape[0] != targets.shape[0]:
                    ratio = images.shape[0] // targets.shape[0]
                    targets = targets.repeat(
                        (ratio,) + (1,) * (len(targets.shape) - 1)
                    )
                    if target_segs:
                        target_segs = target_segs.repeat(
                            (ratio,) + (1,) * (len(target_segs.shape) - 1)
                        )
                if target_segs is None or "normal" in args.experiment or seg_only:
                    outputs = model(images)
                    if args.calculate_map:
                        pred_bbox = get_bbox_from_masks(outputs)
                        target_bbox = unnormalize_bbox_targets(target_bbox)
                        map_metric.update(pred_bbox, target_bbox)

                    if args.debug:
                        save_image(images, "test.png")
                        save_image(COLORMAP[targets.cpu()].permute(0, 3, 1, 2), "test_masks.png")
                        save_image((targets.unsqueeze(dim=1).cpu() * 1.0), "test_masks.png")
                        save_image(COLORMAP[outputs.argmax(1).cpu()].permute(0, 3, 1, 2), "test_preds.png")
                        save_image((outputs.argmax(dim=1).unsqueeze(dim=1).cpu() * 1.0), "test_preds.png")
                        plot_img_bbox(images, target_bbox)
                        plot_img_bbox(images, pred_bbox)
                        pred_masks = (outputs.argmax(dim=1).unsqueeze(dim=1).cpu() * 1.0)
                        pred_masks = pred_masks.repeat(1, 3, 1, 1)
                        plot_img_bbox(pred_masks, pred_bbox)
                elif "groundtruth" in args.experiment:
                    outputs = model(images, segs=target_segs)
                    loss = criterion(outputs, targets)
                else:
                    outputs, masks = model(images, return_mask=True)
                    if "centroid" in args.experiment:
                        masks, _, _, _ = masks
                    pixel_acc = pixel_accuracy(masks, target_segs)
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
        
        # deprecated: computation works but is slow
        if False and seg_only and args.obj_det_arch != "dino":
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
    
    # deprecated: computation works but is slow
    if False and seg_only and args.obj_det_arch != "dino":   
        iou.synchronize()
        print(f"IoU: {iou.avg:.4f}")

    if args.calculate_map:
        print(" * mAP metric")
        # map_dict = {'map': torch.tensor(0)}
        map_dict = map_metric.compute()
        print(map_dict)
        metrics_dict["map"] = map_dict["map"].item()
        
    metrics_dict["acc1"] = top1.avg
    metrics_dict["loss"] = losses.avg
    metrics_dict["pixel-acc"] = pacc.avg
    return metrics_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Part Classification", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    # TODO: add to argparser?
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
                raise ValueError("Key {} can used by args only".format(k))

    os.makedirs(args.output_dir, exist_ok=True)
    main()


