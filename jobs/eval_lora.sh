#!/bin/bash
#SBATCH --job-name=eval_lora
#SBATCH --output=logs/eval_lora_%j.out
#SBATCH --error=logs/eval_lora_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=coc-gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00

cd ~/scratch/peft_llm_benchmark || exit 1

mkdir -p logs outputs reports

source ~/.bashrc
conda activate ~/scratch/conda_envs/llm_peft

export PYTHONPATH=src
export HF_HOME=~/scratch/hf_cache
export TRANSFORMERS_CACHE=~/scratch/hf_cache
export HUGGINGFACE_HUB_CACHE=~/scratch/hf_cache

python scripts/eval.py \
  --config configs/lora.yaml \
  --checkpoint outputs/lora/adapter
