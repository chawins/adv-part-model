#!/bin/bash
ID=9
GPU=1
NUM_GPU=1
BS=2
AA_BS=32
PORT=1000$ID
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
DATAPATH=/data/shared/PartImageNet/
SEGPATH=$DATAPATH/PartSegmentations/All/

# 0.0156862745, 0.03137254901, 0.06274509803
EPS=0.03137254901

# seg-guide/2nd_gt_random/0.0/ts
# CUDA_VISIBLE_DEVICES=$GPU python -u custom_seg_attack_main.py \
#     --seed 0 --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision \
#     --pretrained --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
#     --print-freq 50 --batch-size $AA_BS --epsilon $EPS --atk-norm Linf \
#     --eval-attack seg-guide/untargeted/0.0/ts \
#     --output-dir results/468 --experiment part-wbbox-norm_img-semi --evaluate

# CUDA_VISIBLE_DEVICES=$GPU python -u custom_seg_attack_main.py \
#     --seed 0 --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision \
#     --pretrained --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
#     --print-freq 50 --batch-size $AA_BS --epsilon $EPS --atk-norm Linf \
#     --eval-attack seg-guide/2nd_gt_random/0.0/ts \
#     --output-dir results/468 --experiment part-wbbox-norm_img-semi --evaluate

# CUDA_VISIBLE_DEVICES=$GPU python -u custom_seg_attack_main.py \
#     --seed 0 --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision \
#     --pretrained --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
#     --print-freq 50 --batch-size $AA_BS --epsilon $EPS --atk-norm Linf \
#     --eval-attack seg-guide/2nd_pred_by_scores/0.0/ts \
#     --output-dir results/468 --experiment part-wbbox-norm_img-semi --evaluate

# CUDA_VISIBLE_DEVICES=$GPU python -u custom_seg_attack_main.py \
#     --seed 0 --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision \
#     --pretrained --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
#     --print-freq 50 --batch-size $AA_BS --epsilon $EPS --atk-norm Linf \
#     --eval-attack seg-guide/2nd_pred_by_scores/0.1 \
#     --output-dir results/468 --experiment part-wbbox-norm_img-semi --evaluate

# CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
#     --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
#     --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET --batch-size $BS \
#     --adv-train none \
#     --epsilon $EPS --atk-norm Linf --debug \
#     --resume-if-exist \
#     --output-dir results/2082 --experiment part-wbbox-norm_img-semi



# DATASET=part-imagenet
# BS=32

# CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
#     --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
#     --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET --batch-size $BS \
#     --adv-train none \
#     --epsilon $EPS --atk-norm Linf --debug \
#     --resume-if-exist \
#     --output-dir results/2082 --experiment part-bbox-norm_img-semi

DATASET=part-imagenet-bbox
# GPU=0
# NUM_GPU=1
# TODO: check pretrained and use it for dino
# TODO: does pretrained in this context mean pretrained resnet (backbone) only or does it mean whole dino model?
# TODO: should seg_labels be 41 or 40?

# no adversarial training
# CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
#     --seg-backbone resnet50 --obj-det-arch dino --full-precision --pretrained \
#     --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET --batch-size $BS \
#     --adv-train none \
#     --epsilon $EPS --atk-norm Linf \
#     --resume-if-exist \
#     --seg-const-trn 0.5 \
#     --lr 0.0001 \
#     --output-dir results/2082 --experiment part-bbox-norm_img-semi \
#     --seg-labels 41 \
#     --config_file DINO/config/DINO/DINO_4scale_modified.py \
#     --options dn_scalar=100 dn_label_coef=1.0 dn_bbox_coef=1.0 # all of these are for options

# with pgd adversarial training
CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --seg-backbone resnet50 --obj-det-arch dino --full-precision --pretrained \
    --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET --batch-size $BS \
    --adv-train none \
    --epsilon $EPS --atk-norm Linf \
    --resume-if-exist \
    --seg-const-trn 0.5 \
    --lr 0.0001 \
    --output-dir results/2082 --experiment part-bbox-norm_img-semi \
    --seg-labels 41 \
    --config_file DINO/config/DINO/DINO_4scale_modified.py \
    --options dn_scalar=100 dn_label_coef=1.0 dn_bbox_coef=1.0 # all of these are for options


# python -m torch.distributed.launch --nproc_per_node=2 \
#     main.py \
#     --seg-backbone resnet50 --obj-det-arch dino --full-precision --pretrained \
#     --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET --batch-size $BS \
#     --adv-train none \
#     --epsilon $EPS --atk-norm Linf --debug \
#     --resume-if-exist \
#     --output-dir results/2082 --experiment part-bbox-norm_img-semi \
#     --seg-labels 41 \
#     --config_file DINO/config/DINO/DINO_4scale_modified.py \
#     --options dn_scalar=100 dn_label_coef=1.0 dn_bbox_coef=1.0 # all of these are for options



# python -m torch.distributed.launch --nproc_per_node=8 main.py \
# 	--output_dir logs/DINO/R50-MS4 -c config/DINO/DINO_4scale.py --coco_path $coco_path \
# 	--options dn_scalar=100 embed_init_tgt=TRUE \
# 	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
# 	dn_box_noise_scale=1.0


#TODO: reduce number of queries via options arg

# --dn_box_noise_scale=1.0 in scripts but 0.4 in config file

# # TODO: do not hardcode
# args.config_file = 'DINO/config/DINO/DINO_4scale_modified.py'
# # duplicate keys that exist are [num_classes, lr]
# args.options = {'dn_scalar': 100}

# # TODO: add as args. this was in the original script by dino, DINO_eval.sh
# # args.embed_init_tgt = True 
# args.dn_label_coef=1.0 
# args.dn_bbox_coef=1.0
# # args.use_ema=False
# args.dn_box_noise_scale=1.0


# coco_path=$1
# python -m torch.distributed.launch --nproc_per_node=8 main.py \
# 	--output_dir logs/DINO/R50-MS4 -c config/DINO/DINO_4scale.py --coco_path $coco_path \
# 	--options dn_scalar=100 embed_init_tgt=TRUE \
# 	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
# 	dn_box_noise_scale=1.0









# torchrun \
#     --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
#     main.py --dist-url tcp://localhost:$PORT \
#     --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
#     --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET --batch-size $BS \
#     --adv-train none \
#     --epsilon $EPS --atk-norm Linf --debug \
#     --resume-if-exist \
#     --output-dir results/2082 --experiment part-dino-norm_img-semi

# --evaluate


# CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
#     --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
#     --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET --batch-size $BS \
#     --epsilon $EPS --atk-norm Linf --evaluate \
#     --output-dir results/2065 --experiment part-pooling-4-no_bg-semi

# CUDA_VISIBLE_DEVICES=$GPU python main.py \
#         --dist-url tcp://localhost:$PORT --seed 0 --dist-backend $BACKEND \
#         --seg-backbone resnet50 --seg-arch deeplabv3plus --full-precision --pretrained \
#         --data $DATAPATH --seg-label-dir $SEGPATH --dataset $DATASET \
#         --print-freq 50 --epochs 50 --batch-size $BS --lr 1e-1 --wd 5e-4 \
#         --adv-train none --epsilon $EPS --atk-norm Linf --adv-beta 0.8 --eval-attack pgd \
#         --seg-const-trn 0.5 --semi-label 1 \
#         --output-dir results/temp --experiment part-pooling-4-no_bg-semi

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
