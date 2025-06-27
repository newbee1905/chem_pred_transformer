#!/bin/sh
#SBATCH --qos=xbatch
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=12
#SBATCH --time=1-12:00:00 
#SBATCH --mem=16G
#SBATCH --output=finetune_gpu_job.out
#SBATCH --error=finetune_gpu_job.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s221056384@deakin.edu.au

# activate conda env
source activate $1
source .venv/bin/activate

# srun python train.py
srun python tmp.py
