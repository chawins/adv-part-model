#!/bin/bash
ID=9
GPU=0
NUM_GPU=1
BS=32
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
# ============================ Part-ImageNet-BBox =========================== #
DATASET=part-imagenet-bbox
DATAPATH=~/data/PartImageNet 
SEGPATH=$DATAPATH/PartSegmentations/All/
BBOXDIR=$DATAPATH/PartBoxSegmentations
EPS=0.03137254901


c_seg_arr=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

BASE_PATH="$HOME/data/adv-part-model/models/part-image-net/all"
EXP_NAME="part-seq-conv1d-semi"
EPOCHS=50

# pretrain dino bbox part model
PRETRAIN_DIR="$BASE_PATH/c_seg_0.5"
CUDA_VISIBLE_DEVICES=$GPU torchrun \
    --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
    main.py --dist-url tcp://localhost:$PORT \
    --seg-backbone "resnet50" --obj-det-arch "dino" --full-precision --pretrained \
    --data $DATAPATH --seg-label-dir $SEGPATH --bbox-label-dir $BBOXDIR \
    --dataset $DATASET --batch-size $BS --epochs $EPOCHS --experiment $EXP_NAME \
    --epsilon $EPS \
    --adv-train "none" \
    --seg-const-trn 0.5 \
    --lr 0.01 \
    --output-dir "$PRETRAIN_DIR/pretrained" \
    --seg-labels 40 \
    --config_file "DINO/config/DINO/DINO_4scale_increased_backbone_lr.py" \
    --options dn_scalar=100 dn_label_coef=1.0 dn_bbox_coef=1.0

for C_SEG in ${c_seg_arr[@]}; do
    OUTPUT_DIR="$BASE_PATH/c_seg_${C_SEG}/" # need to change
    echo "$OUTPUT_DIR"
    CUDA_VISIBLE_DEVICES=$GPU torchrun \
        --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
        main.py --dist-url tcp://localhost:$PORT \
        --seg-backbone "resnet50" --obj-det-arch "dino" --full-precision --pretrained \
        --data $DATAPATH --seg-label-dir $SEGPATH --bbox-label-dir $BBOXDIR \
        --dataset $DATASET --batch-size $BS --seg-const-trn $C_SEG \
        --epochs $EPOCHS --experiment $EXP_NAME --epsilon $EPS \
        --adv-train "pgd" \
        --lr 0.01 \
        --resume "$PRETRAIN_DIR/pretrained/checkpoint_best.pt" \
        --load-weight-only \
        --resume-if-exist \
        --output-dir "$OUTPUT_DIR/advtrained/" \
        --seg-labels 40 \
        --config_file "DINO/config/DINO/DINO_4scale_increased_backbone_lr.py" \
        --options dn_scalar=100 dn_label_coef=1.0 dn_bbox_coef=1.0
done
