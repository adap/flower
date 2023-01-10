from pathlib import Path
from argparse import ArgumentTypeError
from typing import List, Tuple
from functools import reduce
import torch


def aggregate_pytorch_tensor(
    results: List[Tuple[List[torch.Tensor], int]]
) -> List[torch.Tensor]:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: List[torch.Tensor] = [
        reduce(torch.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


def valid_folder(path_str: str) -> Path:
    """Tests if a path is a valid FL partition folder

    Args:
                path_str (str): Path to directory containing train and test folder.

    Returns:
                bool: result of checks
    """
    tmp_path = Path(path_str)
    test = True
    for sub_folder in ["train"]:
        test = test and (tmp_path / sub_folder).exists()
    if not test:
        raise ArgumentTypeError
    return tmp_path
