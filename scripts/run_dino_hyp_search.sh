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
DATASET=part-imagenet-bbox
DATAPATH=~/data/PartImageNet 
SEGPATH=$DATAPATH/PartSegmentations/All/
BBOXDIR=$DATAPATH/PartBoxSegmentations
EPS=0.03137254901

EPOCHS=50

c_seg_arr=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

for C_SEG in ${c_seg_arr[@]};
do
    OUTPUT_DIR="~/data/models/part-image-net/all/c_seg_${C_SEG}/" # need to change
    echo $OUTPUT_DIR
    # pretrain dino bbox part model
    torchrun \
        --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
        main.py --dist-url tcp://localhost:$PORT \
        --seg-backbone resnet50 --obj-det-arch dino --full-precision --pretrained \
        --data $DATAPATH --seg-label-dir $SEGPATH --bbox-label-dir $BBOXDIR --dataset $DATASET --batch-size $BS \
        --adv-train none \
        --seg-const-trn $C_SEG \
        --lr 0.01 \
        --epsilon $EPS --atk-norm Linf \
        --output-dir $OUTPUT_DIR/pretrained \
        --epochs $EPOCHS \
        --experiment part-seq-norm_img_conv1d-semi \
        --seg-labels 40 \
        --config_file DINO/config/DINO/DINO_4scale_increased_backbone_lr.py \
        --options dn_scalar=100 dn_label_coef=1.0 dn_bbox_coef=1.0

    torchrun \
        --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
        main.py --dist-url tcp://localhost:$PORT \
        --seg-backbone resnet50 --obj-det-arch dino --full-precision --pretrained \
        --data $DATAPATH --seg-label-dir $SEGPATH --bbox-label-dir $BBOXDIR --dataset $DATASET --batch-size $BS \
        --adv-train pgd \
        --seg-const-trn $C_SEG \
        --lr 0.01 \
        --epsilon $EPS --atk-norm Linf \
        --resume $OUTPUT_DIR/pretrained/checkpoint_best.pt \
        --load-weight-only \
        --resume-if-exist \
        --output-dir $OUTPUT_DIR/advtrained/ \
        --epochs $EPOCHS \
        --experiment part-seq-norm_img_conv1d-semi \
        --seg-labels 40 \
        --config_file DINO/config/DINO/DINO_4scale_increased_backbone_lr.py \
        --options dn_scalar=100 dn_label_coef=1.0 dn_bbox_coef=1.0
done
