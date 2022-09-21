#!/bin/bash
ID=9
GPU=1
NUM_GPU=1
BS=32
PORT=1000$ID
BACKEND=nccl
# =============================== PASCAL-Part =============================== #
DATASET=pascal-part
DATAPATH=~/data/pascal_part/PartImages/aeroplane_bird_car_cat_dog/
SEGPATH=$DATAPATH/panoptic-parts/train
# =============================== Cityscapes ================================ #
# DATASET=cityscapes
# DATAPATH=~/data/cityscapes/PartImages/square_rand_pad0.2/
# SEGPATH=$DATAPATH
# ============================== Part-ImageNet ============================== #
# DATASET=part-imagenet
# DATAPATH=~/data/PartImageNet/
# SEGPATH=$DATAPATH/PartSegmentations/All/

# 0.0156862745, 0.03137254901, 0.06274509803
EPS=0.03137254901

CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
    --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET --batch-size $BS \
    --epsilon $EPS --atk-norm Linf --evaluate --debug --eval-attack pgd \
    --output-dir results/2082 --experiment object-wbbox-norm_img-semi

# CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
#     --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
#     --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET --batch-size $BS \
#     --epsilon $EPS --atk-norm Linf --evaluate \
#     --output-dir results/2065 --experiment part-pooling-4-no_bg-semi

# for i in {1..5}; do
#     CUDA_VISIBLE_DEVICES=$GPU torchrun \
#         --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:2940$ID --rdzv_id=$ID \
#         --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
#         main.py \
#         --dist-url tcp://localhost:$PORT --seed 0 --dist-backend $BACKEND \
#         --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
#         --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
#         --print-freq 50 --epochs 50 --batch-size $BS --lr 1e-1 --wd 5e-4 \
#         --adv-train none --epsilon $EPS --atk-norm Linf --adv-beta 0.8 --eval-attack pgd \
#         --seg-const-trn 0.5 --semi-label 1 \
#         --resume-if-exist \
#         --output-dir results/2098 --experiment part-pooling-4-no_bg-semi && break
#     sleep 30
# done

# CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
#     --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
#     --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET --batch-size $BS \
#     --epsilon $EPS --atk-norm Linf --evaluate --debug \
#     --output-dir results/2072 --experiment part-wbbox-norm_img-semi

# for i in {1..5}; do
#     CUDA_VISIBLE_DEVICES=$GPU torchrun \
#         --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:2940$ID --rdzv_id=$ID \
#         --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
#         main.py \
#         --dist-url tcp://localhost:$PORT --seed 0 --dist-backend $BACKEND \
#         --arch resnet50 --full-precision --pretrained \
#         --data $DATAPATH --dataset $DATASET \
#         --print-freq 50 --epochs 50 --batch-size $BS --lr 1e-1 --wd 5e-4 \
#         --adv-train none --epsilon $EPS --atk-norm Linf --adv-beta 0.8 --eval-attack pgd \
#         --resume-if-exist \
#         --output-dir results/2105 --experiment normal && break
#     sleep 30
# done
