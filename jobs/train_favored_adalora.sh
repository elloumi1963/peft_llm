#!/bin/bash
#SBATCH --job-name=train_favored_ada
#SBATCH --output=logs/train_favored_ada_%j.out
#SBATCH --error=logs/train_favored_ada_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=coc-gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

cd ~/scratch/peft_llm_benchmark || exit 1
mkdir -p logs outputs reports

export PYTHONPATH=src
export HF_HOME=~/scratch/hf_cache
export TOKENIZERS_PARALLELISM=false

rm -rf outputs/favored_adalora_all

~/scratch/conda_envs/llm_peft/bin/python scripts/train.py \
  --config configs/favored_adalora_all.yaml
