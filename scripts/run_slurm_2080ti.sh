#!/bin/bash
#SBATCH --job-name=part-model
#SBATCH --account=fc_wagner
#SBATCH --partition=savio3_gpu
# Number of nodes:
#SBATCH --nodes=1
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=2
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=2
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:GTX2080TI:2
#SBATCH --time=12:00:00
#SBATCH --output slurm-%j-ct_kp_0.5.out
## Command(s) to run:
source /global/home/users/$USER/.bash_profile
module purge
module load python
source activate /global/scratch/users/$USER/pt

# bash scripts/run1-1.sh #
# bash scripts/run1-2.sh # ct_kp_0.4
# bash scripts/run1-3.sh # ct_kp_0.6
bash scripts/run1-4.sh # ct_kp_0.5
# bash scripts/run2-1.sh # box10
# bash scripts/run2-2.sh # box11
# bash scripts/run2-3.sh # 2nd_gt_random/0.0/ts
# bash scripts/run2-4.sh # random/0.0/ts
# bash scripts/run3-1.sh # 2nd_pred_by_scores/0.0/ts
# bash scripts/run3-2.sh # 2nd_pred_by_scores/0.1
# bash scripts/run3-3.sh # untargeted/0.0/ts
# bash scripts/run3-4.sh # 2126

# bash scripts/run4.sh # 2020-2021
# bash scripts/run3.sh   # 2143

# bash scripts/run_reds.sh
