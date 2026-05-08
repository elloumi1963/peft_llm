import time

import torch
from tqdm import tqdm

from peft_llm.data import build_eval_prompt
from peft_llm.utils import clean_text


def generate_predictions(model, tokenizer, dataset, config):
    model.eval()
    rows = []

    for idx, example in enumerate(tqdm(dataset, desc="Generating")):
        prompt = build_eval_prompt(example["dialogue"], tokenizer)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config["max_input_length"],
        ).to(model.device)

        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config["max_new_tokens"],
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        end = time.perf_counter()

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompt_decoded = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

        prediction = decoded.replace(prompt_decoded, "").strip()

        rows.append(
            {
                "id": str(example.get("id", idx)),
                "dialogue": example["dialogue"],
                "reference": clean_text(example["summary"]),
                "prediction": clean_text(prediction),
                "input_tokens": int(inputs["input_ids"].shape[-1]),
                "output_tokens": int(outputs.shape[-1] - inputs["input_ids"].shape[-1]),
                "generation_time_s": float(end - start),
            }
        )

    return rows

