#!/bin/bash
ID=1  # Some unique job id (only matters for distributed training)
GPU=1  # Specify id's of GPU to use
NUM_GPU=1  # Specify number of GPUs to use
BS=8  # Batch size for training (per GPU)
AA_BS=32  # Batch size for AutoAttack
PORT=1000$ID  # Port for distributed training
BACKEND=nccl  # Backend for distributed training (default is nccl)
NUM_WORKERS=2  # Number of workers per GPU

# Set some params for loading data
# =============================== PASCAL-Part =============================== #
# DATASET=pascal-part
# DATAPATH=~/data/pascal_part/PartImages/aeroplane_bird_car_cat_dog/
# SEGPATH=$DATAPATH/panoptic-parts/train
# =============================== Cityscapes ================================ #
# DATASET=cityscapes
# DATAPATH=~/data/cityscapes/PartImages/square_rand_pad0.2/
# SEGPATH=$DATAPATH
# ============================== Part-ImageNet ============================== #
DATASET=part-imagenet-pseudo
DATAPATH=/data/kornrapatp/PartImageNet/
SEGPATH=$DATAPATH/PartSegmentations/All-class-specific-processed/

EPS=0.03137254901  # Set epsilon for adversarial training and evaluation
MODEL=part-pooling-4-semi  # Define model to create (see naming in README)


CUDA_VISIBLE_DEVICES=$GPU torchrun \
    --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:2940$ID --rdzv_id=$ID \
    --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
    main_pseudo.py \
    --dist-url tcp://localhost:$PORT --seed 0 --dist-backend $BACKEND \
    --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
    --data $DATAPATH --seg-label-dir $SEGPATH --seg-labels 599 --dataset $DATASET --workers $NUM_WORKERS \
    --print-freq 50 --epochs 150 --batch-size $BS --lr 1e-1 --wd 5e-4 \
    --adv-train none --epsilon $EPS --atk-norm Linf --adv-beta 0.8 \
    --seg-const-trn 0.5 --semi-label 1 \
    --output-dir /data/kornrapatp/results/158Imagenet --experiment part-seg-only --evaluate
sleep 30