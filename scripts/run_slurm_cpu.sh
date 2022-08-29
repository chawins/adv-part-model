#!/bin/bash
#SBATCH --job-name=part-model
#SBATCH --account=fc_wagner
#SBATCH --partition=savio
# Number of nodes:
#SBATCH --nodes=1
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=20
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#xSBATCH --gres=gpu:GTX2080TI:2
#xSBATCH --qos=v100_gpu3_normal
#SBATCH --time=00:30:00
#SBATCH --output slurm-%j.out
## Command(s) to run:
source /global/home/users/$USER/.bash_profile
module purge
module load python
source activate /global/scratch/users/$USER/pt
sh scripts/prepare_data.sh
