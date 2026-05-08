#!/bin/bash
#SBATCH --job-name=train_final_dora
#SBATCH --output=logs/train_final_dora_%j.out
#SBATCH --error=logs/train_final_dora_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=coc-gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

cd ~/scratch/peft_llm_benchmark || exit 1
mkdir -p logs outputs reports

source ~/.bashrc
conda activate ~/scratch/conda_envs/llm_peft

export PYTHONPATH=src
export HF_HOME=~/scratch/hf_cache
export TOKENIZERS_PARALLELISM=false

rm -rf outputs/final_dora_all_r12

python scripts/train.py \
  --config configs/final_dora_all_r12.yaml
