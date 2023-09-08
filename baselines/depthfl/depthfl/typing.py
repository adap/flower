from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Union

Scalar = Union[bool, bytes, float, int, str]


class Code(Enum):
    """Client status codes."""

    OK = 0
    GET_PROPERTIES_NOT_IMPLEMENTED = 1
    GET_PARAMETERS_NOT_IMPLEMENTED = 2
    FIT_NOT_IMPLEMENTED = 3
    EVALUATE_NOT_IMPLEMENTED = 4


@dataclass
class Status:
    """Client status."""

    code: Code
    message: str


@dataclass
class Parameters:
    """Model parameters."""

    tensors: List[bytes]
    tensor_type: str


@dataclass
class FitIns:
    """Fit instructions for a client."""

    parameters: Parameters
    prev_grads: Dict
    config: Dict[str, Scalar]


@dataclass
class FitRes:
    """Fit response from a client."""

    status: Status
    parameters: Parameters
    prev_grads: Dict
    num_examples: int
    cid: int
