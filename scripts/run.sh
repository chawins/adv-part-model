#!/bin/bash
# export TORCHELASTIC_MAX_RESTARTS=0
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO
# eval "$(conda shell.bash hook)"
# conda activate part
GPU=0
NUM_GPU=1
PORT=10008
BS=32
# DATASET=pascal-part
# DATAPATH=~/data/pascal_part/PartImages/bicycle_bird_v3/
# DATAPATH=~/data/pascal_part/PartImages/aeroplane_car_v3/
# DATAPATH=~/data/pascal_part/PartImages/6classes/
# SEGPATH=$DATAPATH/panoptic-parts/train
DATASET=part-imagenet
DATAPATH=~/data/PartImageNet/
# SEGPATH=$DATAPATH/PartSegmentations/Quadruped_Snake_Reptile_Car/
# SEGPATH=$DATAPATH/PartSegmentations/Biped_Bird_Fish/
SEGPATH=$DATAPATH/PartSegmentations/All/
# SEGPATH=$DATAPATH/BoxSegmentations/All/

# 0.0156862745, 0.03137254901, 0.06274509803
EPS=0.03137254901

# for i in {1..5}; do
#     CUDA_VISIBLE_DEVICES=$GPU torchrun \
#         --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
#         main.py \
#         --dist-url tcp://localhost:$PORT --seed 0 \
#         --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
#         --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
#         --print-freq 50 --epochs 50 --batch-size $BS --lr 1e-1 --wd 5e-4 \
#         --adv-train none --epsilon $EPS --atk-norm Linf --adv-beta 1 --eval-attack pgd \
#         --seg-const-trn 0.5 --semi-label 1 \
#         --resume-if-exist \
#         --output-dir results/999 --experiment part-wbbox-norm_img-semi && break
#     sleep 30
# done

# CUDA_VISIBLE_DEVICES=$GPU python \
#     main.py \
#     --dist-url tcp://localhost:$PORT --seed 0 \
#     --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
#     --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
#     --print-freq 50 --epochs 50 --batch-size $BS --lr 1e-1 --wd 5e-4 \
#     --adv-train none --epsilon $EPS --atk-norm Linf --adv-beta 1 --eval-attack pgd \
#     --seg-const-trn 0.5 --semi-label 1 \
#     --resume-if-exist \
#     --output-dir results/999 --experiment part-pooling-4-no_bg-semi --debug

# CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
#     --seed 0 --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
#     --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
#     --print-freq 50 --batch-size $AA_BS --epsilon $EPS --atk-norm Linf --eval-attack aa \
#     --output-dir results/462 --experiment part-pooling-4-semi --evaluate

# CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
#     --seed 0 --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
#     --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
#     --print-freq 50 --batch-size $BS --epsilon $EPS --atk-norm Linf --eval-attack seg-sum \
#     --output-dir results/462 --experiment part-pooling-4-semi --evaluate \
#     --seg-const-atk 0.5

# CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
#     --seed 0 --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
#     --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
#     --print-freq 50 --batch-size $BS --epsilon $EPS --atk-norm Linf --eval-attack seg-sum \
#     --output-dir results/462 --experiment part-pooling-4-semi --evaluate \
#     --seg-const-atk 0.1

# CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
#     --seed 0 --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
#     --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
#     --print-freq 50 --batch-size $BS --epsilon $EPS --atk-norm Linf --eval-attack seg-sum \
#     --output-dir results/462 --experiment part-pooling-4-semi --evaluate \
#     --seg-const-atk 0.3

# CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
#     --seed 0 --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
#     --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
#     --print-freq 50 --batch-size $BS --epsilon $EPS --atk-norm Linf --eval-attack seg-sum \
#     --output-dir results/462 --experiment part-pooling-4-semi --evaluate \
#     --seg-const-atk 0.7

CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --seed 0 --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
    --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
    --print-freq 50 --batch-size $BS --epsilon $EPS --atk-norm Linf --eval-attack seg-sum \
    --output-dir results/462 --experiment part-pooling-4-semi --evaluate \
    --seg-const-atk 0.9 --debug
