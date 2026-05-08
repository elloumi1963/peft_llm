#!/bin/bash
#SBATCH --job-name=eval_fullft
#SBATCH --output=logs/eval_fullft_%j.out
#SBATCH --error=logs/eval_fullft_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=coc-gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00

cd ~/scratch/peft_llm_benchmark || exit 1

mkdir -p logs outputs reports

source ~/.bashrc
conda activate ~/scratch/conda_envs/llm_peft

export PYTHONPATH=src
export HF_HOME=~/scratch/hf_cache
export TOKENIZERS_PARALLELISM=false

python scripts/eval.py \
  --config configs/full_ft.yaml \
  --checkpoint outputs/full_ft/full_model \
  --full_model
