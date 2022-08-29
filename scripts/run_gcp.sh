#!/bin/bash
export TORCHELASTIC_MAX_RESTARTS=0
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
# eval "$(conda shell.bash hook)"
# conda activate part
GPU=0,1
NUM_GPU=2
PORT=10001
# DATASET=pascal-part
# NC=6
# NSEG=19
# DATAPATH=~/data/pascal_part/PartImages/bicycle_bird_v3/
# DATAPATH=~/data/pascal_part/PartImages/aeroplane_car_v3/
# DATAPATH=~/data/pascal_part/PartImages/6classes_v3/
# SEGPATH=$DATAPATH/panoptic-parts/train
DATASET=part-imagenet
DATAPATH=~/data/PartImageNet/
SEGPATH=$DATAPATH/PartSegmentations/Quadruped_Snake_Reptile_Car/

# 0.0156862745, 0.03137254901
EPS=0.0156862745

# (weird bug) Dummy run to avoid first crash
# CUDA_VISIBLE_DEVICES=$GPU torchrun \
#     --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
#     main.py --dist-url tcp://localhost:$PORT --seed 0 \
#     --arch resnet18 --full-precision \
#     --dataset $DATASET --data $DATAPATH --seg-label-dir $SEGPATH --num-classes $NC \
#     --output-dir results/999 --print-freq 50 --epochs 20 --batch-size 64 \
#     --lr 1e-2 --wd 1e-4 --num-classes $NC --pretrained \
#     --adv-train none --experiment normal --eval-attack pgd

# CUDA_VISIBLE_DEVICES=$GPU torchrun \
#     --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
#     main.py --dist-url tcp://localhost:$PORT \
#     --seed 0 --arch resnet50 --full-precision \
#     --dataset $DATASET --data $DATAPATH --seg-label-dir $SEGPATH --num-classes $NC \
#     --print-freq 50 --epochs 50 --batch-size 64 --lr 1e-2 --wd 1e-4 --pretrained \
#     --adv-train none --epsilon $EPS --atk-norm Linf --adv-beta 2 --eval-attack pgd \
#     --output-dir results/235 --experiment normal

# CUDA_VISIBLE_DEVICES=$GPU torchrun \
#     --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
#     main.py --dist-url tcp://localhost:$PORT \
#     --seed 0 --arch resnet50 --full-precision \
#     --dataset $DATASET --data $DATAPATH --seg-label-dir $SEGPATH --num-classes $NC \
#     --print-freq 50 --epochs 50 --batch-size 64 --lr 1e-2 --wd 1e-4 --pretrained \
#     --adv-train pgd --epsilon $EPS --atk-norm Linf --adv-beta 0.9 --eval-attack pgd,aa \
#     --resume results/236/checkpoint_best.pt --load-weight-only \
#     --output-dir results/241 --experiment normal 

# CUDA_VISIBLE_DEVICES=$GPU torchrun \
#     --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
#     main.py \
#     --dist-url tcp://localhost:$PORT --seed 0 \
#     --arch resnet18 --seg-backbone resnet18 --seg-arch deeplabv3plus --full-precision --pretrained \
#     --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
#     --print-freq 50 --epochs 50 --batch-size 64 --lr 1e-2 --wd 1e-4 \
#     --adv-train none --epsilon $EPS --atk-norm Linf --adv-beta 1 --eval-attack pgd \
#     --seg-loss-const 1 --semi-loss-const 1.0 --semi-label 1.0 --temperature 1.0 \
#     --output-dir results/237 --experiment part-mask-inpt-semi

# CUDA_VISIBLE_DEVICES=$GPU torchrun \
#     --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
#     main.py \
#     --dist-url tcp://localhost:$PORT --seed 0 \
#     --arch resnet18 --seg-backbone resnet18 --seg-arch deeplabv3plus --full-precision --pretrained \
#     --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
#     --print-freq 50 --epochs 50 --batch-size 64 --lr 1e-2 --wd 1e-4 \
#     --adv-train pgd --epsilon $EPS --atk-norm Linf --adv-beta 1 --eval-attack pgd,aa \
#     --seg-loss-const 1 --semi-loss-const 1.0 --semi-label 1.0 --temperature 1.0 \
#     --resume results/237/checkpoint_best.pt --load-weight-only \
#     --output-dir results/238 --experiment part-mask-inpt-semi

# ============================= EXTRA ======================================= #

CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch \
    --nnodes=1 --nproc_per_node=$NUM_GPU \
    --use_env main.py \
    --dist-url tcp://localhost:$PORT --seed 0 \
    --arch resnet50 --seg-backbone resnet18 --seg-arch deeplabv3plus --full-precision --pretrained \
    --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
    --print-freq 50 --epochs 50 --batch-size 64 --lr 1e-2 --wd 1e-4 \
    --adv-train none --epsilon $EPS --atk-norm Linf --adv-beta 1 --eval-attack pgd \
    --seg-loss-const 1 --semi-loss-const 1.0 --semi-label 1.0 --temperature 1.0 \
    --output-dir results/287 --experiment object-mask-inpt-nobg-semi

CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch \
    --nnodes=1 --nproc_per_node=$NUM_GPU \
    --use_env main.py \
    --dist-url tcp://localhost:$PORT --seed 0 --gpu $NUM_GPU \
    --arch resnet50 --seg-backbone resnet18 --seg-arch deeplabv3plus --full-precision --pretrained \
    --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
    --print-freq 50 --epochs 50 --batch-size 64 --lr 1e-2 --wd 1e-4 \
    --adv-train pgd --epsilon $EPS --atk-norm Linf --adv-beta 1 --eval-attack pgd \
    --seg-loss-const 1 --semi-loss-const 1.0 --semi-label 1.0 --temperature 1.0 \
    --resume results/287/checkpoint_best.pt --load-weight-only \
    --output-dir results/288 --experiment object-mask-inpt-nobg-semi