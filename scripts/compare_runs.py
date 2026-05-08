import argparse
import json
from pathlib import Path

import pandas as pd


def load_json(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--out", default="reports/final_metrics.csv")
    args = parser.parse_args()

    rows = []

    for run_dir in Path(args.outputs_dir).iterdir():
        if not run_dir.is_dir():
            continue

        config = load_json(run_dir / "config.json")
        params = load_json(run_dir / "parameter_stats.json")
        train = load_json(run_dir / "train_metrics.json")
        eval_loss = load_json(run_dir / "eval_loss_metrics.json")
        gen = load_json(run_dir / "generation_metrics.json")
        gpu = load_json(run_dir / "gpu_stats.json")

        row = {
            "method": config.get("method", run_dir.name),
            "model": config.get("model_name"),
            **params,
            **{f"train_{k}": v for k, v in train.items()},
            **{f"eval_{k}": v for k, v in eval_loss.items()},
            **gen,
            **{f"gpu_{k}": v for k, v in gpu.items()},
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    preferred_cols = [
        "method",
        "trainable_params",
        "trainable_percent",
        "eval_eval_loss",
        "eval_perplexity_proxy",
        "rouge1",
        "rouge2",
        "rougeL",
        "rougeLsum",
        "sacrebleu",
        "bertscore_f1",
        "distinct_1",
        "distinct_2",
        "length_ratio_pred_ref",
        "generation_time_mean_s",
        "tokens_per_second_mean",
        "gpu_max_memory_allocated_mb",
        "gpu_max_memory_reserved_mb",
        "train_train_runtime",
        "train_train_samples_per_second",
        "train_train_steps_per_second",
    ]

    cols = [c for c in preferred_cols if c in df.columns]
    cols += [c for c in df.columns if c not in cols]

    df = df[cols]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
