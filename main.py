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

from DINO.util.utils import to_device

# from torchmetrics import IoU as IoU   # Use this for older version of torchmetrics
from torchmetrics import JaccardIndex as IoU
from torchvision.utils import save_image

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


def _get_args_parser():
    parser = argparse.ArgumentParser(
        description="Part classification", add_help=False
    )
    parser.add_argument("--data", default="~/data/shared/", type=str)
    parser.add_argument("--arch", default="resnet18", type=str)
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Load pretrained model on ImageNet-1k",
    )
    parser.add_argument(
        "--output-dir", default="./", type=str, help="output dir"
    )
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
    parser.add_argument(
        "--print-freq", default=10, type=int, help="print frequency"
    )
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
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode"
    )
    # TODO
    parser.add_argument("--dataset", required=True, type=str, help="Dataset")
    parser.add_argument(
        "--num-classes", default=10, type=int, help="Number of classes"
    )
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
        type=str,
        help="Architecture of segmentation model",
    )

    parser.add_argument(
        "--obj-det-arch",
        type=str,
        help="Architecture of object detection model",
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
    parser.add_argument("--sample", action="store_true")



    ### DINO args
    # TODO: clean
    from DINO.util.slconfig import DictAction
    parser.add_argument('--config_file', '-c', type=str, required=False)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    # parser.add_argument('--coco_path', type=str, default='/comp_robot/cv_public_dataset/COCO2017/')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')


    # training parameters
    # TODO: merge output_dir and output-dir from before
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')
    

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    



    return parser


best_acc1 = 0


@record
def main(args):
    """Main function."""
    init_distributed_mode(args)


    # handling dino args                    
    if args.config_file:
        from DINO.util.slconfig import SLConfig
        cfg = SLConfig.fromfile(args.config_file)

        if args.options is not None:
            cfg.merge_from_dict(args.options)

        cfg_dict = cfg._cfg_dict.to_dict()
        args_vars = vars(args)
        for k,v in cfg_dict.items():
            if k not in args_vars:
                setattr(args, k, v)
            else:
                raise ValueError("Key {} can used by args only".format(k))

    global best_acc1

    # Fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data loading code
    print("=> creating dataset")
    loaders = load_dataset(args)
    train_loader, train_sampler, val_loader, test_loader = loaders

    debug = False
    if debug:
        # DEBUGGING DATALOADER
        for i, samples in enumerate(train_loader):
            
            import torchvision

            images, target_bbox, targets = samples
            
            images, mask = images.decompose()
            
            debug_index = 0
            torchvision.utils.save_image(images[debug_index], f'example_images/img_{debug_index}.png')
            torchvision.utils.save_image(mask[debug_index] * 1.0, f'example_images/mask_{debug_index}.png')
            img_uint8 = torchvision.io.read_image(f'example_images/img_{debug_index}.png')
            shape = target_bbox[debug_index]['size']
            print(target_bbox[debug_index])

            # xc, xy, w, h convert to xmin, ymin, xmax, ymax
            boxes = target_bbox[debug_index]['boxes']
            boxes[:, ::2] = boxes[:, ::2] * shape[1]
            boxes[:, 1::2] = boxes[:, 1::2] * shape[0]

            box_width = boxes[:, 2]
            box_height = boxes[:, 3]

            boxes[:, 0] = boxes[:, 0] - box_width/2
            boxes[:, 2] = boxes[:, 0] + box_width
            boxes[:, 1] = boxes[:, 1] - box_height/2
            boxes[:, 3] = boxes[:, 1] + box_height
            
            boxes = torch.tensor(boxes, dtype=torch.int)
            img_with_boxes = torchvision.utils.draw_bounding_boxes(img_uint8, boxes=boxes, colors='red')
            torchvision.utils.save_image(img_with_boxes/255, f'example_images/img_{debug_index}_with_bbox.png')
            import pdb
            pdb.set_trace()

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
            wandb.init(
                project="part-model", id=wandb_id, config=args, resume="allow"
            )
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
                val_stats = _validate(
                    val_loader, model, criterion, no_attack, args
                )
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


def _train(
    train_loader, model, criterion, attack, optimizer, scaler, epoch, args
):
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

    # for index, parameter in enumerate(model.parameters()):
    #     print(index, type(parameter))
    #     print(torch.isnan(parameter.data).sum())
    #     print()

    #     if torch.isnan(parameter.data).sum():
    #         import pdb
    #         pdb.set_trace()
    # qqq

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
            # handling dino training
            if args.obj_det_arch == "dino": 
                try:
                    need_tgt_for_training = args.use_dn
                except:
                    need_tgt_for_training = False

                # images, target_bbox, targets = samples
                # print(len(samples))
                # import pdb
                # pdb.set_trace()
                nested_tensors, target_bbox, targets = samples
                images, masks = nested_tensors.decompose()
                targets = torch.LongTensor(targets)
                
            else:
                images, segs, targets = samples
                segs = segs.cuda(args.gpu, non_blocking=True)


        if args.obj_det_arch == "dino": 
            device = 'cuda'
            # images = images.to(device)
            # masks = masks.to(device)
            images = images.cuda(args.gpu, non_blocking=True)
            masks = masks.cuda(args.gpu, non_blocking=True)
            target_bbox = [{k: to_device(v, device) for k, v in t.items()} for t in target_bbox]
            targets = targets.cuda(args.gpu, non_blocking=True)
            batch_size = targets.size(0)
        else:
            images = images.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)
            batch_size = images.size(0)

        # Compute output
        # outputs, dino_outputs = model(images, target_bbox, need_tgt_for_training, return_mask=need_tgt_for_training)
        # loss = criterion(outputs, dino_outputs, target_bbox, targets)
        # print('lossssss')
        # import pdb
        # pdb.set_trace()

        with amp.autocast(enabled=not args.full_precision):
            if args.obj_det_arch == "dino": 
                forward_args = {'masks':masks, 'dino_targets': target_bbox, 'need_tgt_for_training': need_tgt_for_training, 'return_mask': False}
                images = attack(images, targets, forward_args)
                outputs, dino_outputs = model(images, masks, target_bbox, need_tgt_for_training, return_mask=True)
                loss = criterion(outputs, dino_outputs, target_bbox, targets)
            else:
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
            # handling dino validation
            if args.obj_det_arch == "dino": 
                try:
                    need_tgt_for_training = args.use_dn
                except:
                    need_tgt_for_training = False

                # images, target_bbox, targets = samples
                nested_tensors, target_bbox, targets = samples
                images, masks = nested_tensors.decompose()
                targets = torch.LongTensor(targets)
                
            else:
                images, segs, targets = samples
                segs = segs.cuda(args.gpu, non_blocking=True)

                
            # # handling dino validation
            # if args.obj_det_arch == "dino": 
            #     try:
            #         need_tgt_for_training = args.use_dn
            #     except:
            #         need_tgt_for_training = False

            #     images, target_bbox, targets = samples
            #     targets = torch.Tensor(targets)
                
    
            # else:
            #     images, segs, targets = samples
            #     segs = segs.cuda(args.gpu, non_blocking=True)


        # DEBUG
        if args.debug:
            save_image(COLORMAP[segs].permute(0, 3, 1, 2), "gt.png")
            save_image(images, "test.png")

        if args.obj_det_arch == "dino": 
            device = 'cuda'
            images = images.to(device)
            masks = masks.to(device)

            target_bbox = [{k: to_device(v, device) for k, v in t.items()} for t in target_bbox]
            targets = targets.cuda(args.gpu, non_blocking=True)
            batch_size = targets.size(0)

        else:
            images = images.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)
            batch_size = images.size(0)



        # if args.obj_det_arch == "dino": 
        #     from DINO.util.utils import to_device
        #     device = 'cuda'
        #     images = images.to(device)
        #     target_bbox = [{k: to_device(v, device) for k, v in t.items()} for t in target_bbox]
        #     # images = images.cuda(args.gpu, non_blocking=True)
        #     # target_bbox = [{k: v.cuda(args.gpu, non_blocking=True) for k, v in t.items()} for t in target_bbox]
        #     targets = targets.cuda(args.gpu, non_blocking=True)
        #     batch_size = targets.size(0)
        #     # batch_size = images.size(0)
        # else:
        #     images = images.cuda(args.gpu, non_blocking=True)
        #     targets = targets.cuda(args.gpu, non_blocking=True)
        #     batch_size = images.size(0)

        # DEBUG: fixed clean segmentation masks
        if "clean" in args.experiment:
            model(images, clean=True)

        # compute output
        with torch.no_grad():
            if args.obj_det_arch == "dino": 

                forward_args = {'masks':masks, 'dino_targets': target_bbox, 'need_tgt_for_training': need_tgt_for_training, 'return_mask': False}
                images = attack(images, targets, forward_args)
                outputs = model(images, masks, target_bbox, need_tgt_for_training, return_mask=False)
                loss = criterion(outputs, targets)


                # outputs, _ = model(images, target_bbox, need_tgt_for_training, return_mask=need_tgt_for_training)
                # loss = criterion(outputs, targets)

            else:
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
                    if segs:
                        segs = segs.repeat((ratio,) + (1,) * (len(segs.shape) - 1))

                if segs is None or "normal" in args.experiment or seg_only:
                    outputs = model(images)
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
