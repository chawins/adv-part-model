#!/bin/bash
# export TORCHELASTIC_MAX_RESTARTS=0
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO
ID=1
GPU=0,1,2,3
NUM_GPU=4
BS=32
AA_BS=32
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
DATAPATH=/global/scratch/users/kornrapatp/PartImageNet/
SEGPATH=$DATAPATH/PartSegmentations/All/
# SEGPATH=$DATAPATH/BoxSegmentations/All/

# 0.0156862745, 0.03137254901, 0.06274509803
EPS=0.03137254901



for i in {1..5}; do
    CUDA_VISIBLE_DEVICES=$GPU torchrun \
        --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:2940$ID --rdzv_id=$ID \
        --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
        main.py \
        --dist-url tcp://localhost:$PORT --seed 0 --dist-backend $BACKEND \
        --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
        --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
        --print-freq 50 --epochs 50 --batch-size $BS --lr 1e-1 --wd 5e-4 \
        --adv-train none --epsilon $EPS --atk-norm Linf --adv-beta 2 \
        --seg-const-trn 0.5 --semi-label 0.3 \
        --output-dir /global/scratch/users/kornrapatp/1111 --experiment part-wbbox-norm_img-centroid-semi && break
    sleep 30
done