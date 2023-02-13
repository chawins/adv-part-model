"""Utility functions for argument parsing."""

import argparse


def get_args_parser() -> argparse.ArgumentParser:
    """Create argparser for common main function."""
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
        help="Path to segmentation labels.",
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
        help=(
            'Select types of adversarial training (options: "none", "pgd", '
            '"trades", "mat", "atta". Defaults to "none" (normal training).'
        ),
    )
    parser.add_argument(
        "--atk-steps", default=10, type=int, help="Number of attack iterations."
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
        help="Beta parameter for TRADES or MAT (default: 6.0).",
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

    # Detection model args
    parser.add_argument(
        "--bbox-label-dir",
        default="",
        type=str,
        help="Path to bounding-box labels.",
    )
    parser.add_argument(
        "--obj-det-arch",
        type=str,
        help="Architecture of object detection model.",
    )
    parser.add_argument(
        "--use-imagenet-classes",
        action="store_true",
        help=(
            "If True, use ImageNet-1k classes instead of PartImageNet "
            "meta-classes."
        ),
    )
    parser.add_argument(
        "--group-parts",
        action="store_true",
        help="If True, group PartImageNet parts.",
    )
    parser.add_argument(
        "--calculate-map",
        action="store_true",
        help="If True, calculate mAP for object detection model.",
    )

    # TODO(nab-126@): clean
    from DINO.util.slconfig import DictAction

    parser.add_argument("--config_file", "-c", type=str, required=False)
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help=(
            "override some settings in the used config, the key-value pair "
            "in xxx=yyy format will be merged into config file."
        ),
    )

    # dataset parameters
    parser.add_argument("--dataset_file", default="coco")
    parser.add_argument("--coco_panoptic_path", type=str)
    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument("--fix_size", action="store_true")
    parser.add_argument("--sample", action="store_true")

    # training parameters
    parser.add_argument(
        "--note", default="", help="add some notes to the experiment"
    )
    parser.add_argument(
        "--pretrain_model_path", help="load from other checkpoint"
    )
    parser.add_argument("--finetune_ignore", type=str, nargs="+")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--find_unused_params", action="store_true")
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--save_log", action="store_true")

    return parser
