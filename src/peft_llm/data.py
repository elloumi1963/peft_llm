from datasets import load_dataset


def build_train_prompt(dialogue, summary, tokenizer):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant specialized in dialogue summarization.",
        },
        {
            "role": "user",
            "content": f"Summarize the following dialogue in one concise paragraph:\n\n{dialogue}",
        },
        {
            "role": "assistant",
            "content": summary,
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def build_eval_prompt(dialogue, tokenizer):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant specialized in dialogue summarization.",
        },
        {
            "role": "user",
            "content": f"Summarize the following dialogue in one concise paragraph:\n\n{dialogue}",
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_samsum(config):
    dataset = load_dataset(config["dataset_name"])

    train = dataset["train"].shuffle(seed=config["seed"])
    val = dataset["validation"].shuffle(seed=config["seed"])
    test = dataset["test"].shuffle(seed=config["seed"])

    train = train.select(range(min(config["max_train_samples"], len(train))))
    val = val.select(range(min(config["max_eval_samples"], len(val))))
    test = test.select(range(min(config["max_test_samples"], len(test))))

    return train, val, test


def tokenize_example(example, tokenizer, config):
    text = build_train_prompt(
        dialogue=example["dialogue"],
        summary=example["summary"],
        tokenizer=tokenizer,
    )

    tokens = tokenizer(
        text,
        truncation=True,
        max_length=config["max_length"],
        padding=False,
    )

    tokens["labels"] = tokens["input_ids"].copy()
    return tokens
