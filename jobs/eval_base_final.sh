#!/bin/bash
#SBATCH --job-name=eval_base_final
#SBATCH --output=logs/eval_base_final_%j.out
#SBATCH --error=logs/eval_base_final_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=coc-gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00

cd ~/scratch/peft_llm_benchmark || exit 1

mkdir -p logs reports outputs/base_final

source ~/.bashrc
conda activate ~/scratch/conda_envs/llm_peft

export PYTHONPATH=src
export HF_HOME=~/scratch/hf_cache
export TOKENIZERS_PARALLELISM=false

python scripts/eval.py \
  --config configs/base_final.yaml \
  --base_only
