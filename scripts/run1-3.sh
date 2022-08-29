#!/bin/bash
# export TORCHELASTIC_MAX_RESTARTS=0
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO
ID=3
GPU=4,5
NUM_GPU=2
BS=32
AA_BS=48
PORT=1000$ID
BACKEND=nccl
# =============================== PASCAL-Part =============================== #
# DATASET=pascal-part
# DATAPATH=~/data/pascal_part/PartImages/aeroplane_bird_car_cat_dog/
# SEGPATH=$DATAPATH/panoptic-parts/train
# =============================== Cityscapes ================================ #
# DATASET=cityscapes
# DATAPATH=~/data/cityscapes/PartImages/square_rand_pad0.2/
# SEGPATH=$DATAPATH
# ============================== Part-ImageNet ============================== #
DATASET=part-imagenet
DATAPATH=~/data/PartImageNet/
SEGPATH=$DATAPATH/PartSegmentations/All/
# SEGPATH=$DATAPATH/BoxSegmentations/All/

# 0.0156862745, 0.03137254901, 0.06274509803
EPS=0.03137254901

CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --seed 0 --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
    --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
    --print-freq 50 --batch-size $AA_BS --epsilon $EPS --atk-norm Linf --eval-attack aa \
    --resume results/1121/checkpoint_epoch44.pt \
    --output-dir results/1121 --experiment part-pooling-4-semi --evaluate
