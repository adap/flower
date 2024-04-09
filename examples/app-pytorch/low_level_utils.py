import torch
import numpy as np
from flwr.common.typing import NDArray
from flwr.common.record import RecordSet, ParametersRecord, Array


def _ndarray_to_array(ndarray: NDArray) -> Array:
    """Represent NumPy ndarray as Array."""
    return Array(
        data=ndarray.tobytes(),
        dtype=str(ndarray.dtype),
        stype="numpy.ndarray.tobytes",
        shape=list(ndarray.shape),
    )


def _basic_array_deserialisation(array: Array) -> NDArray:
    return np.frombuffer(buffer=array.data, dtype=array.dtype).reshape(array.shape)


def pytorch_to_parameter_record(pytorch_module: torch.nn.Module):
    """Serialise your PyTorch model."""
    state_dict = pytorch_module.state_dict()

    for k, v in state_dict.items():
        state_dict[k] = _ndarray_to_array(v.numpy())

    return ParametersRecord(state_dict)


def parameters_to_pytorch_state_dict(params_record: ParametersRecord):
    """Reconstruct PyTorch state_dict from its serialised representation."""
    state_dict = {}
    for k, v in params_record.items():
        state_dict[k] = torch.tensor(_basic_array_deserialisation(v))

    return state_dict
