#!/bin/bash
#SBATCH --job-name=part-model
#SBATCH --account=fc_wagner
#SBATCH --partition=savio3_gpu
# Number of nodes:
#SBATCH --nodes=1
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=2
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=4
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:V100:2
#SBATCH --qos=v100_gpu3_normal
#SBATCH --time=24:00:00
#SBATCH --output slurm-%j-v100-exp-1121.out
## Command(s) to run:
source /global/home/users/$USER/.bash_profile
module purge
module load python
source activate /global/scratch/users/$USER/pt

bash scripts/run1-1.sh # 1121
# bash scripts/run1-2.sh # 956-957
# bash scripts/run3-4.sh # 1112
