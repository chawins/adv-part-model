#!/bin/bash
# export TORCHELASTIC_MAX_RESTARTS=0
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO
ID=9
GPU=0
NUM_GPU=1
BS=32
AA_BS=32
PORT=1005$ID
BACKEND=nccl
NUM_WORKERS=2
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

# for i in {1..5}; do
#     torchrun \
#         --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:2940$ID --rdzv_id=$ID \
#         --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
#         main.py \
#         --dist-url tcp://localhost:$PORT --seed 0 --dist-backend $BACKEND \
#         --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
#         --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
#         --print-freq 50 --epochs 50 --batch-size $BS --lr 1e-1 --wd 5e-4 \
#         --adv-train pgd --epsilon $EPS --atk-norm Linf --adv-beta 0.8 --eval-attack pgd \
#         --seg-const-trn 0.4 --semi-label 1 \
#         --resume results/box1/checkpoint_best.pt --load-weight-only --resume-if-exist \
#         --output-dir results/box12 --experiment part-pooling-4-semi && break
#     sleep 30
# done

python -u custom_seg_attack_main.py \
    --seed 0 --dist-url tcp://localhost:$PORT --full-precision \
    --seg-backbone resnet50 --seg-arch deeplabv3plus \
    --pretrained --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
    --print-freq 10 --batch-size $AA_BS --epsilon $EPS --atk-norm Linf \
    --workers $((NUM_GPU * NUM_WORKERS)) --eval-attack seg-guide/2nd_pred_by_scores/0.0/ts \
    --experiment part-pooling-4-no_bg-semi --evaluate \
    --resume results/462/checkpoint_best.pt --output-dir results/462
