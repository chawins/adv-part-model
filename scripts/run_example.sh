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
DATASET=part-imagenet
DATAPATH=~/data/PartImageNet/
SEGPATH=$DATAPATH/PartSegmentations/All/

EPS=0.03137254901  # Set epsilon for adversarial training and evaluation
MODEL=part-wbbox-norm_img-semi  # Define model to create (see naming in README)

# Run normal training
torchrun \
    --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:2940$ID --rdzv_id=$ID \
    --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
    main.py \
    --dist-url tcp://localhost:$PORT --seed 0 --dist-backend $BACKEND \
    --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision \
    --pretrained --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
    --workers $NUM_WORKERS --print-freq 50 --epochs 50 --batch-size $BS \
    --lr 1e-1 --wd 5e-4 --adv-train none --epsilon $EPS --atk-norm Linf \
    --adv-beta 1.0 --seg-const-trn 0.5 --semi-label 1 \
    --eval-attack pgd --output-dir results/example_normal --experiment $MODEL
sleep 30

# Run adversarial training by tuning from the normally trained model above.
# After training finishes, it runs PGD evaluation.
# NOTE: For loop just restarts the job in case it dies before finishing
for i in {1..5}; do
    torchrun \
        --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:2940$ID --rdzv_id=$ID \
        --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
        main.py \
        --dist-url tcp://localhost:$PORT --seed 0 --dist-backend $BACKEND \
        --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision \
        --pretrained --data $DATAPATH --seg-label-dir $SEGPATH \
        --dataset $DATASET --print-freq 50 --epochs 50 --batch-size $BS \
        --lr 1e-1 --wd 5e-4 --adv-train pgd --epsilon $EPS --atk-norm Linf \
        --adv-beta 1.0 --seg-const-trn 0.5 --semi-label 1 \
        --resume results/example_normal/checkpoint_best.pt --load-weight-only \
        --resume-if-exist --output-dir results/example_adv --experiment $MODEL && break
    sleep 30
done

# Run evaluation with AutoAttack
python -u main.py \
    --seed 0 --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision \
    --pretrained --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
    --print-freq 50 --batch-size $AA_BS --epsilon $EPS --atk-norm Linf \
    --eval-attack aa --output-dir results/example_normal \
    --experiment $MODEL --evaluate