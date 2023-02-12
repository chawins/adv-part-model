#!/bin/bash
#SBATCH --job-name=part-model
#SBATCH --account=fc_wagner
#SBATCH --partition=savio3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=6:00:00
#SBATCH --output slurm-prepare-data-%j.out
## Command(s) to run:
# source /global/home/users/$USER/.bash_profile
# module purge
# module load python
# source activate /global/scratch/users/$USER/env_part_based

# python prepare_cityscapes.py \
#     --seed 0 \
#     --data-dir /global/scratch/users/nabeel126/cityscapes/ \
#     --name bbox_square_rand_pad0.2 \
#     --pad 0.2 \
#     --min-area 1000 \
#     --square \
#     --rand-pad \
#     --allow-missing-parts \
#     --use-box-seg

# python prepare_pascal_part.py \
#     --data-dir /global/scratch/users/nabeel126/pascal_part/ \
#     --name aeroplane_bird_car_cat_dog_new \
#     --min-area 0.

# python -u prepare_part_imagenet.py \
#     --data-dir ~/data/PartImageNet/ \
#     --name All


DATAPATH=~/data/PartImageNet 

# ### Data Prep
# mkdir $DATAPATH/PartBoxSegmentations/train/
# mkdir $DATAPATH/PartBoxSegmentations/val/
# mkdir $DATAPATH/PartBoxSegmentations/test/

# ln -s $DATAPATH/JPEGImages/* $DATAPATH/PartBoxSegmentations/train/
# ln -s $DATAPATH/JPEGImages/* $DATAPATH/PartBoxSegmentations/val/
# ln -s $DATAPATH/JPEGImages/* $DATAPATH/PartBoxSegmentations/test/

python prepare_part_imagenet_bbox.py --label-dir $DATAPATH --split train
python prepare_part_imagenet_bbox.py --label-dir $DATAPATH --split val
python prepare_part_imagenet_bbox.py --label-dir $DATAPATH --split test

# python prepare_part_imagenet_bbox.py --label-dir $DATAPATH --split train --use-imagenet-classes
# python prepare_part_imagenet_bbox.py --label-dir $DATAPATH --split val --use-imagenet-classes
# python prepare_part_imagenet_bbox.py --label-dir $DATAPATH --split test --use-imagenet-classes

# python prepare_part_imagenet_bbox.py --label-dir $DATAPATH --split train --group-parts
# python prepare_part_imagenet_bbox.py --label-dir $DATAPATH --split val --group-parts
# python prepare_part_imagenet_bbox.py --label-dir $DATAPATH --split test --group-parts

# python prepare_part_imagenet_bbox.py --label-dir $DATAPATH --split train --use-imagenet-classes --group-parts
# python prepare_part_imagenet_bbox.py --label-dir $DATAPATH --split val --use-imagenet-classes --group-parts
# python prepare_part_imagenet_bbox.py --label-dir $DATAPATH --split test --use-imagenet-classes --group-parts