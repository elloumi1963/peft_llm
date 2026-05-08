#!/bin/bash
#SBATCH --job-name=train_close_lora
#SBATCH --output=logs/train_close_lora_%j.out
#SBATCH --error=logs/train_close_lora_%j.err
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

python scripts/train.py \
  --config configs/close_lora_attn_r12.yaml
