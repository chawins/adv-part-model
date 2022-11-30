#!/bin/bash
#SBATCH --job-name=part-model
#SBATCH --account=fc_wagner
#SBATCH --partition=savio2_1080ti
# Number of nodes:
#SBATCH --nodes=1
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=2
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=1
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output slurm-%j-ct_kp_0.5.out
## Command(s) to run:
# source /global/home/users/$USER/.bash_profile
module purge
# module load python
source activate /global/scratch/users/$USER/env_part_based
module load cuda/10.2
module load gcc/6.4.0 

python main_debug.py

cd /global/home/users/nabeel126/adv-part-model/DINO/models/dino/ops/
bash make.sh
cd /global/home/users/nabeel126/adv-part-model/DINO/





ID=8
GPU=0
NUM_GPU=1
# BS=32
# AA_BS=32
PORT=1000$ID
# BACKEND=nccl
# NUM_WORKERS=2

coco_path=/global/scratch/users/nabeel126/coco/
checkpoint=/global/scratch/users/nabeel126/adv-part-model/DINO/DINO_pretrained_models/checkpoint0011_4scale.pth

echo "Starting Evaluation"


torchrun \
    --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
	main.py \
	--output_dir logs/DINO/R50-MS4-%j \
	-c config/DINO/DINO_4scale.py --coco_path $coco_path  \
	--eval --resume $checkpoint \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0 


# python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 1 --rank 0 ...

# python main.py \
#   --output_dir logs/DINO/R50-MS4-%j \
# 	-c config/DINO/DINO_4scale.py --coco_path $coco_path  \
# 	--eval --resume $checkpoint \
# 	--options dn_scalar=100 embed_init_tgt=TRUE \
# 	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
# 	dn_box_noise_scale=1.0 \
# 	--dist-backend $BACKEND
	
# python main.py \
#   --output_dir logs/DINO/R50-MS4-%j \
# 	-c config/DINO/DINO_4scale.py --coco_path $coco_path  \
# 	--eval --resume $checkpoint \
# 	--options dn_scalar=100 embed_init_tgt=TRUE \
# 	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
# 	dn_box_noise_scale=1.0

