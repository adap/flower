import math

import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM


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
    """Load model with appropriate dtype and device map settings."""

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

    return model
