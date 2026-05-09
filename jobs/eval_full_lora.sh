#!/bin/bash
#SBATCH --job-name=eval_full_lora
#SBATCH --output=logs/eval_full_lora_%j.out
#SBATCH --error=logs/eval_full_lora_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=coc-gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00

cd ~/scratch/peft_llm_benchmark || exit 1
mkdir -p logs reports

export PYTHONPATH=src
export HF_HOME=~/scratch/hf_cache
export TOKENIZERS_PARALLELISM=false

~/scratch/conda_envs/llm_peft/bin/python scripts/eval.py \
  --config configs/full_lora_all_r12.yaml \
  --checkpoint outputs/full_lora_all_r12/adapter
