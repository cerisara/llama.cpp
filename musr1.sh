#!/bin/bash
#SBATCH --job-name=murder
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --time=2:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --cpus-per-task=8
#SBATCH --account=knb@a100
#SBATCH -C a100

# cmake -B build -DGGML_CUDA=ON
# cmake --build build --config Release

# hack to avoid issues of very small $HOME
export HOME=$WORK"/home/"

# to get idr_torch
module load arch/a100
module load pytorch-gpu/py3/2.7.0

## launch script on every node
set -x

echo "DATEDEBUT"
date

cd ../llamacppgerg
mod="../models/Qwen3-14B-Q4_K_M.gguf"

co=0
while IFS="" read -r p || [ -n "$p" ]
do
    ./llama-cli -m "$mod" --temp 0.7 -c 16000 -fa -ngl 100 -n 500 -p "$p" -no-cnv --no-display-prompt > outok.$co
    co=$((co+1))
done < ../llama.cpp/qmurder_ok

co=0
while IFS="" read -r p || [ -n "$p" ]
do
    ./llama-cli -m "$mod" --temp 0.7 -c 16000 -fa -ngl 100 -n 500 -p "$p" -no-cnv --no-display-prompt > outko.$co
    co=$((co+1))
done < ../llama.cpp/qmurder_ko
 
