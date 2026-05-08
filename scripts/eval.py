import argparse
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM

from peft_llm.data import load_samsum
from peft_llm.generation import generate_predictions
from peft_llm.metrics import compute_metrics
from peft_llm.model import load_base_model, load_tokenizer
from peft_llm.utils import gpu_stats, load_config, save_json, save_jsonl, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--base_only", action="store_true")
    parser.add_argument("--full_model", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config["seed"])

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config["model_name"])
    _, _, test_dataset = load_samsum(config)

    if args.base_only:
        model = load_base_model(config)

    elif args.full_model:
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            device_map="auto",
            trust_remote_code=True,
        )

    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required unless --base_only or --full_model is used")

        base_model = load_base_model(config)
        model = PeftModel.from_pretrained(base_model, args.checkpoint)

    model.eval()

    rows = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        dataset=test_dataset,
        config=config,
    )

    metrics = compute_metrics(rows)

    save_jsonl(rows, output_dir / "generations.jsonl")
    save_json(metrics, output_dir / "generation_metrics.json")
    save_json(gpu_stats(), output_dir / "eval_gpu_stats.json")

    print("Evaluation complete.")
    print(metrics)


if __name__ == "__main__":
    main()
