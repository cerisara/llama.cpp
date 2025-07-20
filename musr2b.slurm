#!/bin/bash
#SBATCH --job-name=murder
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --time=1:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --cpus-per-task=8
#SBATCH --account=knb@a100
#SBATCH -C a100

# hack to avoid issues of very small $HOME
export HOME=$WORK"/home/"

# to get idr_torch
module load arch/a100
module load pytorch-gpu/py3/2.7.0

## launch script on every node
set -x

echo "DATEDEBUT"
date

srun python musr2b.py
 
