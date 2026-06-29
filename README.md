# Parameter-Efficient Fine-Tuning of Large Language Models for Dialogue Summarization

## Project Overview

Large Language Models achieve strong performance on text generation tasks but are expensive to fine-tune because updating billions of parameters requires substantial computational resources and GPU memory.

Parameter-Efficient Fine-Tuning (PEFT) addresses this challenge by updating only a small subset of parameters while keeping the pretrained model frozen.

This project investigates whether PEFT methods can effectively adapt a **1.5B-parameter instruction-tuned language model** to the **dialogue summarization** task while maintaining high performance and significantly reducing training cost.

Using **Qwen2.5-1.5B-Instruct** and the **SAMSum** dataset, we compare three popular PEFT approaches:

- **LoRA (Low-Rank Adaptation)**
- **DoRA (Weight-Decomposed Low-Rank Adaptation)**
- **AdaLoRA (Adaptive Low-Rank Adaptation)**

All methods are trained under the same experimental setting and evaluated using standard summarization metrics including **ROUGE**, **BLEU**, and **BERTScore**.

The objective is to identify which PEFT strategy provides the best trade-off between summarization quality, parameter efficiency, and computational cost.

## Repository Structure

```
peft_llm/
│
├── configs/                 # YAML configuration files for each experiment
│
├── jobs/                    # SLURM scripts for GPU cluster execution
│
├── outputs/
│   ├── base_final/          # Baseline evaluation
│   ├── final_lora_all_r12/  # Final LoRA experiment
│   ├── final_dora_all_r12/  # Final DoRA experiment
│   └── final_adalora_all/   # Final AdaLoRA experiment
│
├── scripts/
│   ├── train.py             # Fine-tuning entry point
│   ├── evaluate.py          # Model evaluation
│   ├── generate.py          # Generate summaries
│   └── ...
│
├── src/
│   ├── dataset.py           # Dataset preprocessing
│   ├── model.py             # PEFT model creation
│   ├── metrics.py           # ROUGE/BLEU/BERTScore computation
│   └── ...
│
├── requirements.txt
└── README.md
```
## Installation

Clone the repository

```bash
git clone https://github.com/elloumi1963/peft_llm.git

cd peft_llm
```

Install dependencies

```bash
pip install -r requirements.txt
```

## Running Experiments

### Baseline Evaluation

```bash
python scripts/evaluate.py \
    --config configs/base.yaml
```

---

### Train LoRA

```bash
python scripts/train.py \
    --config configs/lora_r12.yaml
```

---

### Train DoRA

```bash
python scripts/train.py \
    --config configs/dora_r12.yaml
```

---

### Train AdaLoRA

```bash
python scripts/train.py \
    --config configs/adalora.yaml
```

---

### Evaluate a Trained Model

```bash
python scripts/evaluate.py \
    --checkpoint outputs/final_lora_all_r12/checkpoint-best
```

# Experimental Results

## Training Configuration

| Setting | Value |
|----------|-------|
| Base Model | Qwen2.5-1.5B-Instruct |
| Dataset | SAMSum |
| Epochs | 3 |
| LoRA Rank | 12 |
| Precision | BF16 |
| Optimizer | AdamW |
| Framework | HuggingFace PEFT |

---

## Final Performance

| Model | Trainable Parameters | Trainable % | ROUGE-L ↑ | ROUGE-2 ↑ | BLEU ↑ | BERTScore F1 ↑ |
|--------|---------------------:|------------:|----------:|----------:|-------:|---------------:|
| Base Model | 0 | 0% | 0.196 | 0.074 | 2.99 | 0.877 |
| LoRA | 13.85 M | ~0.92% | **0.365** | 0.200 | 13.68 | 0.912 |
| DoRA | 14.49 M | ~0.96% | 0.364 | **0.207** | **13.85** | **0.912** |
| AdaLoRA | 13.85 M | ~0.92% | 0.345 | 0.184 | 12.03 | 0.908 |

---

## Performance Improvement over the Base Model

| Method | ROUGE-L Gain | BLEU Gain |
|---------|-------------:|----------:|
| LoRA | +86% | +357% |
| DoRA | +86% | +363% |
| AdaLoRA | +76% | +302% |

---

# Discussion

## Base Model

Without task-specific fine-tuning, the instruction-tuned model generates verbose responses rather than concise summaries. This results in poor lexical overlap with the reference summaries and establishes the baseline for comparison.

---

## LoRA

LoRA provides the strongest balance between efficiency and performance.

Only **13.85 million parameters** (less than **1%** of the full model) are updated, yet the model nearly doubles its ROUGE-L score and increases BLEU by more than four times compared to the frozen base model.

This demonstrates that low-rank adaptation is sufficient to specialize a 1.5B parameter language model for dialogue summarization.

---

## DoRA

DoRA updates **14.49 million trainable parameters**, slightly more than LoRA due to its magnitude decomposition.

It achieves nearly identical overall performance while obtaining the highest BLEU and ROUGE-2 scores, indicating slightly better phrase-level generation.

However, these gains remain marginal considering the additional computational complexity introduced by DoRA.

---

## AdaLoRA

AdaLoRA also fine-tunes only **13.85 million parameters**, but unlike LoRA it dynamically reallocates the rank budget across layers during training.

Although AdaLoRA substantially outperforms the frozen base model, it consistently underperforms both LoRA and DoRA in our experiments.

Several factors likely explain this behavior.

- **Limited dataset size.** SAMSum contains approximately 15,000 training dialogues. Adaptive rank allocation generally benefits from larger and more diverse datasets where different layers exhibit significantly different importance.

- **Additional optimization complexity.** AdaLoRA must simultaneously learn the summarization task while continuously redistributing its rank budget. This introduces a more challenging optimization problem than fixed-rank LoRA.

- **Insufficient adaptation time.** Dynamic rank allocation usually requires enough training iterations for the importance scores to stabilize. On a relatively small dataset with only a few training epochs, the adaptive allocation may not fully converge.

As a result, the flexibility offered by AdaLoRA does not translate into higher summarization quality in this setting. A fixed-rank strategy (LoRA) proves sufficient and more stable for medium-scale dialogue summarization.

---

# Key Findings

- PEFT methods dramatically outperform the frozen base model while updating less than **1%** of the model parameters.
- LoRA provides the best trade-off between performance, simplicity, and computational efficiency.
- DoRA achieves slightly better lexical metrics but with additional computational overhead.
- AdaLoRA demonstrates that adaptive rank allocation is **not universally beneficial**. On a medium-sized dataset such as SAMSum, the added optimization complexity outweighs its theoretical advantages, making fixed-rank LoRA the more effective choice.
