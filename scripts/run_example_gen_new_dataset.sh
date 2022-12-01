#!/bin/bash
ID=1  # Some unique job id (only matters for distributed training)
GPU=0,1  # Specify id's of GPU to use
NUM_GPU=2  # Specify number of GPUs to use
BS=32  # Batch size for training (per GPU)
AA_BS=32  # Batch size for AutoAttack
PORT=1000$ID  # Port for distributed training
BACKEND=nccl  # Backend for distributed training (default is nccl)
NUM_WORKERS=2  # Number of workers per GPU

# Set some params for loading data
# =============================== PASCAL-Part =============================== #
# DATASET=pascal-part
# DATAPATH=~/data/pascal_part/PartImages/aeroplane_bird_car_cat_dog/
# SEGPATH=$DATAPATH/panoptic-parts/train
# =============================== Cityscapes ================================ #
# DATASET=cityscapes
# DATAPATH=~/data/cityscapes/PartImages/square_rand_pad0.2/
# SEGPATH=$DATAPATH
# ============================== Part-ImageNet ============================== #
DATASET=part-imagenet-pseudo
DATAPATH=/data/kornrapatp/PartImageNet
SEGPATH=$DATAPATH/PartSegmentations/All/

OLD_DATASET=$DATAPATH/PartSegmentations/All
NEW_DATASET=$DATAPATH/PartSegmentations/test2
NEW_SEGPATH=$NEW_DATASET-mask
JPEG_PATH=$DATAPATH/JPEGImages
NUM_NEW_SAMPLES=1000
PREDICTION_PATH=/data/kornrapatp/test
PREDICTION_MODEL_PATH=/data/kornrapatp/results/MetaClassAll-new

EPS=0.03137254901  # Set epsilon for adversarial training and evaluation
MODEL=part-seg-only  # Define model to create (see naming in README)

# train a segmenter using this command (use main.py and part-imagenet-pseudo as dataset)
CUDA_VISIBLE_DEVICES=$GPU torchrun \
    --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:2940$ID --rdzv_id=$ID \
    --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
    main.py \
    --dist-url tcp://localhost:$PORT --seed 0 --dist-backend $BACKEND \
    --seg-backbone resnet101 --seg-arch deeplabv3plus --full-precision --pretrained \
    --data $DATAPATH --seg-label-dir $SEGPATH --seg-labels 41 --dataset $DATASET --workers $NUM_WORKERS \
    --print-freq 50 --epochs 150 --batch-size $BS --lr 1e-1 --wd 5e-4 \
    --adv-train none --epsilon $EPS --atk-norm Linf --adv-beta 0.8 \
    --seg-const-trn 0.5 --semi-label 1 \
    --output-dir $PREDICTION_MODEL_PATH --experiment $MODEL

# preprocess the dataset
python3 pre_prediction.py --old-dataset $OLD_DATASET --new-dataset $NEW_DATASET \
    --jpeg-path $JPEG_PATH --num-new-samples $NUM_NEW_SAMPLES --prediction-path $PREDICTION_PATH

# get predicted masks
CUDA_VISIBLE_DEVICES=$GPU torchrun \
    --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:2940$ID --rdzv_id=$ID \
    --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
    main_pseudo.py \
    --dist-url tcp://localhost:$PORT --seed 0 --dist-backend $BACKEND \
    --seg-backbone resnet101 --seg-arch deeplabv3plus --full-precision --pretrained \
    --data $DATAPATH --seg-label-dir $NEW_SEGPATH --seg-labels 41 --dataset $DATASET --workers $NUM_WORKERS \
    --print-freq 50 --epochs 150 --batch-size $BS --lr 1e-1 --wd 5e-4 \
    --adv-train none --epsilon $EPS --atk-norm Linf --adv-beta 0.8 \
    --seg-const-trn 0.5 --semi-label 1 --prediction-path $PREDICTION_PATH \
    --output-dir $PREDICTION_MODEL_PATH --experiment $MODEL --evaluate

# postprocess the dataset
python3 post_prediction.py --old-dataset $OLD_DATASET --new-dataset $NEW_DATASET \
    --jpeg-path $JPEG_PATH --num-new-samples $NUM_NEW_SAMPLES --prediction-path $PREDICTION_PATH
