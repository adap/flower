# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower utilities shared between server and client."""


from .parameter import bytes_to_ndarray as bytes_to_ndarray
from .parameter import ndarray_to_bytes as ndarray_to_bytes
from .parameter import parameters_to_weights as parameters_to_weights
from .parameter import weights_to_parameters as weights_to_parameters
from .typing import Config as Config
from .typing import Disconnect as Disconnect
from .typing import EvaluateIns as EvaluateIns
from .typing import EvaluateRes as EvaluateRes
from .typing import FitIns as FitIns
from .typing import FitRes as FitRes
from .typing import Metrics as Metrics
from .typing import Parameters as Parameters
from .typing import ParametersRes as ParametersRes
from .typing import Properties as Properties
from .typing import PropertiesIns as PropertiesIns
from .typing import PropertiesRes as PropertiesRes
from .typing import Reconnect as Reconnect
from .typing import Scalar as Scalar
from .typing import Weights as Weights

GRPC_MAX_MESSAGE_LENGTH: int = 536_870_912  # == 512 * 1024 * 1024

__all__ = [
    "bytes_to_ndarray",
    "Config",
    "Disconnect",
    "EvaluateIns",
    "EvaluateRes",
    "FitIns",
    "FitRes",
    "GRPC_MAX_MESSAGE_LENGTH",
    "Metrics",
    "ndarray_to_bytes",
    "Parameters",
    "parameters_to_weights",
    "ParametersRes",
    "Properties",
    "PropertiesIns",
    "PropertiesRes",
    "Reconnect",
    "Scalar",
    "Weights",
    "weights_to_parameters",
]
