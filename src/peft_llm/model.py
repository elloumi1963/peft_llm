import torch
from peft import (
    AdaLoraConfig,
    LoraConfig,
    TaskType,
    get_peft_model,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(config):
    if config["method"] == "full_ft":
        dtype = torch.float32
    elif config.get("bf16", False):
        dtype = torch.bfloat16
    elif config.get("fp16", False):
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False

    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    return model


def build_lora_config(config, use_dora=False):
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        bias="none",
        use_dora=use_dora,
    )


def build_adalora_config(config):
    return AdaLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        init_r=config["adalora_init_r"],
        target_r=config["adalora_target_r"],
        beta1=config["adalora_beta1"],
        beta2=config["adalora_beta2"],
        tinit=config["adalora_tinit"],
        tfinal=config["adalora_tfinal"],
        deltaT=config["adalora_deltaT"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
	total_step=config["adalora_total_step"],
        bias="none",
    )


def apply_peft(model, config):
    method = config["method"]

    if method == "lora":
        peft_config = build_lora_config(config, use_dora=False)

    elif method == "dora":
        peft_config = build_lora_config(config, use_dora=True)

    elif method == "adalora":
        peft_config = build_adalora_config(config)

    else:
        raise ValueError(f"Unknown method: {method}")

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model
