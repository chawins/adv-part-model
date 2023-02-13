#!/bin/bash
ID=9
GPU=1
NUM_GPU=1
BS=2
AA_BS=32
PORT=1000$ID
# =============================== PASCAL-Part =============================== #
# DATASET=pascal-part
# DATAPATH=~/data/pascal_part/PartImages/aeroplane_bird_car_cat_dog/
# SEGPATH=$DATAPATH/panoptic-parts/train
# =============================== Cityscapes ================================ #
# DATASET=cityscapes
# DATAPATH=~/data/cityscapes/PartImages/square_rand_pad0.2/
# SEGPATH=$DATAPATH
# ============================== Part-ImageNet ============================== #
# DATASET=part-imagenet
# DATAPATH=/data/shared/PartImageNet/
# SEGPATH=$DATAPATH/PartSegmentations/All/
# ============================== Part-ImageNet-BBox ============================== #
DATASET="part-imagenet-bbox"
DATAPATH="~/data/PartImageNet" # need to change
SEGPATH="$DATAPATH/PartSegmentations/All/"
BBOXDIR="$DATAPATH/PartBoxSegmentations"
# 0.0156862745, 0.03137254901, 0.06274509803
EPS=0.03137254901

EPOCHS=50


### Data Prep
mkdir $DATAPATH/PartBoxSegmentations/train/
mkdir $DATAPATH/PartBoxSegmentations/val/
mkdir $DATAPATH/PartBoxSegmentations/test/

ln -s $DATAPATH/JPEGImages/* $DATAPATH/PartBoxSegmentations/train/
ln -s $DATAPATH/JPEGImages/* $DATAPATH/PartBoxSegmentations/val/
ln -s $DATAPATH/JPEGImages/* $DATAPATH/PartBoxSegmentations/test/

python3 prepare_part_imagenet_bbox.py --label-dir $DATAPATH --split train
python3 prepare_part_imagenet_bbox.py --label-dir $DATAPATH --split val
python3 prepare_part_imagenet_bbox.py --label-dir $DATAPATH --split test


### Training
EXP_NAME="part-seq-norm_img-semi"
ADV_TRAIN="pgd"
OUTPUT_DIR="./models/part-imagenet/$EXP_NAME/$ADV_TRAIN/"  # Change as needed
ADV_BETA=0.6 # need to change
# pretrain dino bbox part model
CUDA_VISIBLE_DEVICES=$GPU torchrun \
    --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
    main.py --dist-url tcp://localhost:$PORT \
    --seg-backbone "resnet50" --obj-det-arch "dino" --full-precision --pretrained \
    --dataset $DATASET --batch-size $BS --output-dir $OUTPUT_DIR \
    --data $DATAPATH --seg-label-dir $SEGPATH --bbox-label-dir $BBOXDIR \
    --adv-train $ADV_TRAIN --epochs $EPOCHS --experiment $EXP_NAME \
    --epsilon $EPS --atk-steps 10 --adv-beta $ADV_BETA \
    --seg-const-trn 0.5 \
    --lr 0.0001 \
    --seg-labels 41 \
    --config_file "DINO/config/DINO/DINO_4scale_modified.py" \
    --options dn_scalar=100 dn_label_coef=1.0 dn_bbox_coef=1.0

# adv train (TRADES) dino bbox part model 
# torchrun \
#     --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
#     main.py --dist-url tcp://localhost:$PORT \
#     --seg-backbone resnet50 --obj-det-arch dino --full-precision --pretrained \
#     --data $DATAPATH --seg-label-dir $SEGPATH --bbox-label-dir $BBOXDIR --dataset $DATASET --batch-size $BS \
#     --adv-train none \
#     --seg-const-trn 0.5 \
#     --lr 0.0001 \
#     --epsilon $EPS --atk-norm Linf \
#     --output-dir $OUTPUT_DIR/pretrained \
#     --epochs $EPOCHS \
#     --experiment part-bbox-norm_img-semi \
#     --seg-labels 41 \
#     --config_file DINO/config/DINO/DINO_4scale_modified.py \
#     --options dn_scalar=100 dn_label_coef=1.0 dn_bbox_coef=1.0

# adv train (TRADES) dino bbox part model 
# torchrun \
#     --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
#     main.py --dist-url tcp://localhost:$PORT \
#     --seg-backbone resnet50 --obj-det-arch dino --full-precision --pretrained \
#     --data $DATAPATH --seg-label-dir $SEGPATH --bbox-label-dir $BBOXDIR --dataset $DATASET --batch-size $BS \
#     --adv-train trades \
#     --adv-beta $ADV_BETA \
#     --seg-const-trn 0.5 \
#     --lr 0.0001 \
#     --epsilon $EPS --atk-norm Linf \
#     --resume $OUTPUT_DIR/pretrained/checkpoint_best.pt \
#     --load-weight-only \
#     --resume-if-exist \
#     --output-dir $OUTPUT_DIR/advtrained/ \
#     --epochs $EPOCHS \
#     --experiment part-bbox-norm_img-semi \
#     --seg-labels 41 \
#     --config_file DINO/config/DINO/DINO_4scale_modified.py \
#     --options dn_scalar=100 dn_label_coef=1.0 dn_bbox_coef=1.0

EXP_NAME="part-seq-norm_img-semi"
ADV_TRAIN="pgd"
ADV_BETA=0.6 # need to change
OUTPUT_DIR="./models/part-imagenet/$EXP_NAME/$ADV_TRAIN/"  # Change as needed
# adv train (pgd) dino two headed bbox part model
CUDA_VISIBLE_DEVICES=$GPU torchrun \
    --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
    main.py --dist-url tcp://localhost:$PORT \
    --seg-backbone "resnet50" --obj-det-arch "dino" --full-precision --pretrained \
    --data $DATAPATH --seg-label-dir $SEGPATH --bbox-label-dir $BBOXDIR \
    --dataset $DATASET --batch-size $BS --adv-train $ADV_TRAIN \
    --output-dir $OUTPUT_DIR --experiment $EXP_NAME --epochs $EPOCHS \
    --epsilon $EPS --adv-beta $ADV_BETA \
    --resume $OUTPUT_DIR/pretrained/checkpoint_best.pt --load-weight-only \
    --resume-if-exist \
    --seg-labels 41 \
    --seg-const-trn 0.5 \
    --lr 0.0001 \
    --config_file "DINO/config/DINO/DINO_4scale_modified.py" \
    --options dn_scalar=100 dn_label_coef=1.0 dn_bbox_coef=1.0
