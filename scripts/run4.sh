#!/bin/bash
# export TORCHELASTIC_MAX_RESTARTS=0
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO
ID=4
GPU=3
NUM_GPU=1
BS=64
PORT=1100$ID
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

# 0.0156862745, 0.03137254901, 0.06274509803
EPS=0.06274509803

CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --seed 0 --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
    --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
    --print-freq 50 --batch-size $BS --epsilon $EPS --atk-norm Linf --eval-attack longer-pgd \
    --output-dir results/462 --experiment part-pooling-4-semi --evaluate

EPS=0.09411764705

CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --seed 0 --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
    --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
    --print-freq 50 --batch-size $BS --epsilon $EPS --atk-norm Linf --eval-attack longer-pgd \
    --output-dir results/462 --experiment part-pooling-4-semi --evaluate

EPS=0.12549019607

CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --seed 0 --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
    --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
    --print-freq 50 --batch-size $BS --epsilon $EPS --atk-norm Linf --eval-attack longer-pgd \
    --output-dir results/462 --experiment part-pooling-4-semi --evaluate

# for i in {1..5}; do
#     CUDA_VISIBLE_DEVICES=$GPU torchrun \
#         --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:2940$ID --rdzv_id=$ID \
#         --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
#         main.py \
#         --dist-url tcp://localhost:$PORT --seed 0 --dist-backend $BACKEND \
#         --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
#         --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
#         --print-freq 50 --epochs 50 --batch-size $BS --lr 1e-1 --wd 5e-4 \
#         --adv-train trades --epsilon $EPS --atk-norm Linf --adv-beta 1 --eval-attack pgd \
#         --seg-const-trn 0.5 --semi-label 1 \
#         --resume results/2098/checkpoint_best.pt --load-weight-only --resume-if-exist \
#         --output-dir results/2143 --experiment part-pooling-4-no_bg-semi && break
#     sleep 30
# done

# CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
#     --seed 0 --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
#     --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
#     --print-freq 50 --batch-size $((BS * 2)) --epsilon $EPS --atk-norm Linf --eval-attack aa \
#     --output-dir results/2143 --experiment part-pooling-4-no_bg-semi --evaluate

# for i in {1..5}; do
#     CUDA_VISIBLE_DEVICES=$GPU torchrun \
#         --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:2940$ID --rdzv_id=$ID \
#         --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
#         main.py \
#         --dist-url tcp://localhost:$PORT --seed 0 --dist-backend $BACKEND \
#         --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
#         --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
#         --print-freq 50 --epochs 50 --batch-size $BS --lr 1e-1 --wd 5e-4 \
#         --adv-train trades --epsilon $EPS --atk-norm Linf --adv-beta 2 --eval-attack pgd \
#         --seg-const-trn 0.5 --semi-label 1 \
#         --resume results/2098/checkpoint_best.pt --load-weight-only --resume-if-exist \
#         --output-dir results/2144 --experiment part-pooling-4-no_bg-semi && break
#     sleep 30
# done

# CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
#     --seed 0 --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
#     --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
#     --print-freq 50 --batch-size $((BS * 2)) --epsilon $EPS --atk-norm Linf --eval-attack aa \
#     --output-dir results/2144 --experiment part-pooling-4-no_bg-semi --evaluate
