#!/bin/bash
# export TORCHELASTIC_MAX_RESTARTS=0
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO
ID=8
GPU=0,1,2,3
NUM_GPU=4
BS=32
AA_BS=32
PORT=1000$ID
BACKEND=nccl
NUM_WORKERS=2
# =============================== PASCAL-Part =============================== #
# DATASET=pascal-part
# DATAPATH=~/data/pascal_part/PartImages/aeroplane_bird_car_cat_dog/
# SEGPATH=$DATAPATH/panoptic-parts/train
# =============================== Cityscapes ================================ #
DATASET=cityscapes
DATAPATH=~/data/cityscapes/PartImages/square_rand_pad0.2/
SEGPATH=$DATAPATH
# ============================== Part-ImageNet ============================== #
# DATASET=part-imagenet
# DATAPATH=/global/scratch/users/kornrapatp/PartImageNet/
# SEGPATH=$DATAPATH/PartSegmentations/All/
# SEGPATH=$DATAPATH/BoxSegmentations/All/

# 0.0156862745, 0.03137254901, 0.06274509803
EPS=0.03137254901

torchrun \
    --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:2940$ID --rdzv_id=$ID \
    --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
    main.py \
    --dist-url tcp://localhost:$PORT --seed 0 --dist-backend $BACKEND \
    --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
    --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET --workers $NUM_WORKERS \
    --print-freq 50 --epochs 50 --batch-size $BS --lr 1e-1 --wd 5e-4 \
    --adv-train none --epsilon $EPS --atk-norm Linf --adv-beta 0.8 --eval-attack pgd \
    --seg-const-trn 0.5 --semi-label 1 \
    --output-dir results/ct_kp_1 --experiment part-wbbox-norm_img-centroid-semi
sleep 30

# for i in {1..5}; do
#     torchrun \
#         --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:2940$ID --rdzv_id=$ID \
#         --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
#         main.py \
#         --dist-url tcp://localhost:$PORT --seed 0 --dist-backend $BACKEND \
#         --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
#         --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
#         --print-freq 50 --epochs 50 --batch-size $BS --lr 1e-1 --wd 5e-4 \
#         --adv-train pgd --epsilon $EPS --atk-norm Linf --adv-beta 0.8 \
#         --seg-const-trn 0.4 --semi-label 1 \
#         --resume results/pc_kp_1/checkpoint_best.pt --load-weight-only --resume-if-exist \
#         --output-dir results/paskey1 --experiment part-wbbox-norm_img-centroid-semi && break
#     sleep 30
# done

# for i in {1..5}; do
#     torchrun \
#         --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:2940$ID --rdzv_id=$ID \
#         --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
#         main.py \
#         --dist-url tcp://localhost:$PORT --seed 0 --dist-backend $BACKEND \
#         --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
#         --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
#         --print-freq 50 --epochs 50 --batch-size $BS --lr 1e-1 --wd 5e-4 \
#         --adv-train pgd --epsilon $EPS --atk-norm Linf --adv-beta 0.8 \
#         --seg-const-trn 0.5 --semi-label 1 \
#         --resume /global/scratch/users/kornrapatp/paskey/checkpoint_best.pt --load-weight-only --resume-if-exist \
#         --output-dir /global/scratch/users/kornrapatp/paskey2 --experiment part-wbbox-norm_img-centroid-semi && break
#     sleep 30
# done

# for i in {1..5}; do
#     torchrun \
#         --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:2940$ID --rdzv_id=$ID \
#         --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
#         main.py \
#         --dist-url tcp://localhost:$PORT --seed 0 --dist-backend $BACKEND \
#         --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
#         --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
#         --print-freq 50 --epochs 50 --batch-size $BS --lr 1e-1 --wd 5e-4 \
#         --adv-train pgd --epsilon $EPS --atk-norm Linf --adv-beta 0.8 \
#         --seg-const-trn 0.6 --semi-label 1 \
#         --resume /global/scratch/users/kornrapatp/paskey/checkpoint_best.pt --load-weight-only --resume-if-exist \
#         --output-dir /global/scratch/users/kornrapatp/paskey3 --experiment part-wbbox-norm_img-centroid-semi && break
#     sleep 30
# done

# python -u main.py \
#     --seed 0 --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
#     --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
#     --print-freq 50 --batch-size $AA_BS --epsilon $EPS --atk-norm Linf --eval-attack aa \
#     --output-dir /global/scratch/users/kornrapatp/paskey1 --experiment part-wbbox-norm_img-centroid-semi --evaluate

# python -u main.py \
#     --seed 0 --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
#     --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
#     --print-freq 50 --batch-size $AA_BS --epsilon $EPS --atk-norm Linf --eval-attack aa \
#     --output-dir /global/scratch/users/kornrapatp/paskey2 --experiment part-wbbox-norm_img-centroid-semi --evaluate

# python -u main.py \
#     --seed 0 --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
#     --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
#     --print-freq 50 --batch-size $AA_BS --epsilon $EPS --atk-norm Linf --eval-attack aa \
#     --output-dir /global/scratch/users/kornrapatp/paskey3 --experiment part-wbbox-norm_img-centroid-semi --evaluate