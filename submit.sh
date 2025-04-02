#!/bin/sh
#SBATCH --qos=xbatch
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=6
#SBATCH --time=0-24:00:00 
#SBATCH --mem=16G
#SBATCH --output=finetune_gpu_job.out
#SBATCH --error=finetune_gpu_job.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s221056384@deakin.edu.au

# activate conda env
source activate $1
source .venv/bin/activate

srun python train.py
