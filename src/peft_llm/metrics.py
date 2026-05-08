import math
from collections import Counter

import evaluate
import numpy as np


def safe_perplexity(loss):
    try:
        return float(math.exp(loss))
    except OverflowError:
        return float("inf")


def distinct_n(texts, n):
    total = 0
    unique = set()

    for text in texts:
        tokens = text.split()
        grams = list(zip(*[tokens[i:] for i in range(n)]))
        total += len(grams)
        unique.update(grams)

    if total == 0:
        return 0.0

    return len(unique) / total


def compression_ratio(predictions, references):
    pred_lens = [len(x.split()) for x in predictions]
    ref_lens = [len(x.split()) for x in references]

    return float(np.mean(pred_lens) / max(np.mean(ref_lens), 1e-9))


def compute_metrics(rows):
    predictions = [r["prediction"] for r in rows]
    references = [r["reference"] for r in rows]

    rouge = evaluate.load("rouge")
    bleu = evaluate.load("sacrebleu")
    bertscore = evaluate.load("bertscore")

    rouge_scores = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )

    bleu_scores = bleu.compute(
        predictions=predictions,
        references=[[r] for r in references],
    )

    bert_scores = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
    )

    pred_lengths = [len(x.split()) for x in predictions]
    ref_lengths = [len(x.split()) for x in references]
    gen_times = [r["generation_time_s"] for r in rows]
    output_tokens = [r["output_tokens"] for r in rows]

    return {
        "rouge1": float(rouge_scores["rouge1"]),
        "rouge2": float(rouge_scores["rouge2"]),
        "rougeL": float(rouge_scores["rougeL"]),
        "rougeLsum": float(rouge_scores["rougeLsum"]),
        "sacrebleu": float(bleu_scores["score"]),

        "bertscore_precision": float(np.mean(bert_scores["precision"])),
        "bertscore_recall": float(np.mean(bert_scores["recall"])),
        "bertscore_f1": float(np.mean(bert_scores["f1"])),

        "distinct_1": float(distinct_n(predictions, 1)),
        "distinct_2": float(distinct_n(predictions, 2)),

        "prediction_words_mean": float(np.mean(pred_lengths)),
        "prediction_words_std": float(np.std(pred_lengths)),
        "reference_words_mean": float(np.mean(ref_lengths)),
        "length_ratio_pred_ref": compression_ratio(predictions, references),

        "empty_prediction_count": int(sum(len(x.strip()) == 0 for x in predictions)),

        "generation_time_total_s": float(np.sum(gen_times)),
        "generation_time_mean_s": float(np.mean(gen_times)),
        "tokens_per_second_mean": float(
            np.mean([tok / max(t, 1e-9) for tok, t in zip(output_tokens, gen_times)])
        ),
    }
