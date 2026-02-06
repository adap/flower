import math

import torch
from omegaconf import DictConfig
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft.utils import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""

    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_model(model_cfg: DictConfig):
    """Load model with appropriate quantization config and other optimizations.

    Please refer to this example for `peft + BitsAndBytes`:
    https://github.com/huggingface/peft/blob/main/examples/fp4_finetuning/finetune_fp4_opt_bnb_peft.py
    """

    #if model_cfg.quantization == 4:
    #    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    #elif model_cfg.quantization == 8:
    #    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    #else:
    #    raise ValueError(
    #        f"Use 4-bit or 8-bit quantization. You passed: {model_cfg.quantization}/"
    #    )

    dtype = getattr(model_cfg, "dtype", "bfloat16")
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "float32":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.bfloat16

    load_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if hasattr(model_cfg, "device_map") and model_cfg.device_map:
        load_kwargs["device_map"] = model_cfg.device_map

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        **load_kwargs,
    )

    if getattr(model_cfg, "device_map", "") == "cpu":
        model = model.to("cpu")

    #model = prepare_model_for_kbit_training(
    #    model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
    #)

    #peft_config = LoraConfig(
    #    r=model_cfg.lora.peft_lora_r,
    #    lora_alpha=model_cfg.lora.peft_lora_alpha,
    #    lora_dropout=0.075,
    #    task_type="CAUSAL_LM",
    #)

    #return get_peft_model(model, peft_config)
    return model
