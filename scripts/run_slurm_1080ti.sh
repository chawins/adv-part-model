#!/bin/bash
#SBATCH --job-name=part-model
#SBATCH --account=fc_wagner
#SBATCH --partition=savio2_1080ti
# Number of nodes:
#SBATCH --nodes=1
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=4
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=2
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output slurm-%j-exp138.out
## Command(s) to run:
source /global/home/users/$USER/.bash_profile
module purge
module load python
source activate /global/scratch/users/$USER/pt

# bash scripts/run1-3.sh # 1096
# bash scripts/run1-4.sh # 1098
# bash scripts/run2-1.sh # 1100
# bash scripts/run2-2.sh # 1102
# bash scripts/run2-3.sh # 1104
# bash scripts/run2-4.sh # 138
bash scripts/run3-1.sh # 1106
# bash scripts/run3-2.sh # 1108
# bash scripts/run3-3.sh # 1110
