#!/bin/bash
#SBATCH --job-name=eval_dora
#SBATCH --output=logs/eval_dora_%j.out
#SBATCH --error=logs/eval_dora_%j.err
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
export TRANSFORMERS_CACHE=~/scratch/hf_cache
export HUGGINGFACE_HUB_CACHE=~/scratch/hf_cache
export TOKENIZERS_PARALLELISM=false

python scripts/eval.py \
  --config configs/dora.yaml \
  --checkpoint outputs/dora/adapter
