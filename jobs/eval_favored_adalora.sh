#!/bin/bash
#SBATCH --job-name=eval_favored_ada
#SBATCH --output=logs/eval_favored_ada_%j.out
#SBATCH --error=logs/eval_favored_ada_%j.err
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
  --config configs/favored_adalora_all.yaml \
  --checkpoint outputs/favored_adalora_all/adapter
