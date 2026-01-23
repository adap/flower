"""dasha: A Flower Baseline."""

import random

import numpy as np
import torch

from flwr.common.typing import NDArrays


def _get_dataset_input_shape(dataset):
    assert len(dataset) > 0
    sample_features, _ = dataset[0]
    return list(sample_features.shape)


def get_parameters(net) -> NDArrays:
    """Return the parameters of the current model."""
    parameters = [
        val.detach().cpu().numpy().flatten()
        for _, val in net.named_parameters()
    ]
    return [np.concatenate(parameters)]


def _generate_seed(generator):
    return generator.integers(2**32 - 1)


def set_seed(seed: int):
    """Fix randomness."""
    generator = np.random.default_rng(seed=42)
    seed = int(_generate_seed(generator))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed


def reformat_config(config):
    """Reformat config into nested structure."""
    formatted_config = {}
    for key, value in config.items():
        if "." in key:
            category, key = key.split(".")
            if category not in formatted_config:
                formatted_config[category] = {key: value}
            else:
                formatted_config[category][key] = value
        else:
            formatted_config[key] = value

    return formatted_config
