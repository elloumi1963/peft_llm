import argparse
from pathlib import Path

from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from peft_llm.data import load_samsum, tokenize_example
from peft_llm.metrics import safe_perplexity
from peft_llm.model import apply_peft, load_base_model, load_tokenizer
from peft_llm.utils import count_parameters, gpu_stats, load_config, save_json, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config["seed"])

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(config, output_dir / "config.json")

    tokenizer = load_tokenizer(config["model_name"])
    train_raw, val_raw, _ = load_samsum(config)

    train_dataset = train_raw.map(
        lambda x: tokenize_example(x, tokenizer, config),
        remove_columns=train_raw.column_names,
    )

    val_dataset = val_raw.map(
        lambda x: tokenize_example(x, tokenizer, config),
        remove_columns=val_raw.column_names,
    )

    model = load_base_model(config)

    if config["method"] != "full_ft":
        model = apply_peft(model, config)

    parameter_stats = count_parameters(model)
    save_json(parameter_stats, output_dir / "parameter_stats.json")

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],

        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type=config["lr_scheduler_type"],

        logging_steps=config["logging_steps"],
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        save_strategy="steps",
        save_steps=config["save_steps"],
        save_total_limit=1,

        bf16=config["bf16"],
        fp16=config["fp16"],

        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=config["gradient_checkpointing"],
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    train_result = trainer.train()
    eval_result = trainer.evaluate()

    train_metrics = train_result.metrics
    eval_metrics = eval_result

    if "eval_loss" in eval_metrics:
        eval_metrics["perplexity_proxy"] = safe_perplexity(eval_metrics["eval_loss"])

    save_json(train_metrics, output_dir / "train_metrics.json")
    save_json(eval_metrics, output_dir / "eval_loss_metrics.json")

    if config["method"] == "full_ft":
        model_dir = output_dir / "full_model"
        trainer.model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
    else:
        adapter_dir = output_dir / "adapter"

        if config["method"] == "adalora":
            trainer.model.save_pretrained(
                adapter_dir,
                safe_serialization=False,
            )
        else:
            trainer.model.save_pretrained(adapter_dir)

        tokenizer.save_pretrained(adapter_dir)

    save_json(gpu_stats(), output_dir / "gpu_stats.json")

    print("Training complete.")
    print(parameter_stats)
    print(eval_metrics)


if __name__ == "__main__":
    main()
