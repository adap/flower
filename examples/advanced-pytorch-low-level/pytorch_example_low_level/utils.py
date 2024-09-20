"""pytorch-example-low-level: A low-level Flower / PyTorch app."""

from collections import OrderedDict
from typing import Any

import torch

from flwr.common import ParametersRecord, array_from_numpy


def state_dict_to_parameters_record(state_dict: OrderedDict[str, Any]):
    """Express parameters of a PyTorch state_dict as a ParametersRecord."""

    # Convert the state_dict into a dictionary of flwr.common.Array
    state_arrays = {}
    for k, v in state_dict.items():
        state_arrays[k] = array_from_numpy(v.cpu().numpy())

    return ParametersRecord(state_arrays)


def parameters_record_to_state_dict(p_record: ParametersRecord):
    """Convert a ParametersRecord into its PyTorch state_dict representation."""

    state_dict = {}
    for k, v in p_record.items():
        state_dict[k] = torch.from_numpy(v.numpy())

    return state_dict
