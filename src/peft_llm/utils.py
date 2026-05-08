import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import set_seed


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def count_parameters(model):
    total = 0
    trainable = 0

    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n

    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "trainable_percent": float(100 * trainable / total),
    }


def gpu_stats():
    if not torch.cuda.is_available():
        return {"cuda_available": False}

    return {
        "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "max_memory_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "max_memory_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
    }


def clean_text(text):
    return " ".join(str(text).replace("\n", " ").split()).strip()
