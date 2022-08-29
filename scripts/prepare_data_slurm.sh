#!/bin/bash
#SBATCH --job-name=prepare_data
#SBATCH --account=fc_wagner
#SBATCH --partition=savio3_bigmem
# Number of nodes:
#SBATCH --nodes=1
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
## Command(s) to run:
python prepare_cityscapes_v2.py \
    --data-dir /global/scratch/users/chawins/data/cityscapes/ \
    --name all_parts_sqaure_bal \
    --pad 0.1 \
    --min-area 1000 \
    --square \
    --allow-missing-parts
