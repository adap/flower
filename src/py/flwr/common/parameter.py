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
"""Parameter conversion."""


from io import BytesIO
from typing import cast

import numpy as np
from numpy.core.fromnumeric import shape

from .typing import Parameters, Weights
###
import os
###

def weights_to_parameters(weights: Weights, name: str, epoch: int) -> Parameters:
    """Convert NumPy weights to parameters object."""
    ###
    # Wipe previous data, create shapes file, and ignore initialization parameters
    if(name == "ignore"):
        for f in os.listdir("data"):
            os.remove(os.path.join("data", f))
        # Create shape file
        shape_file = open(os.path.join("data", "shapes.txt"), "w")
        for weight in weights:
            shape = tuple(weight.shape)
            shape_str = ""
            for num in shape:
                shape_str += str(num) + ","
            shape_str = shape_str[:len(shape_str) - 1] + "\n"
            shape_file.write(shape_str)
        shape_file.close()
    elif(name == "testing"):
        pass
    else:
        # Flatten parameters into 1D array of 32-bit floats
        out_array = weights[0]
        out_array = out_array.flatten("C")
        for i in range(1, len(weights)):
            nparray = weights[i]
            nparray = nparray.flatten("C")
            out_array = np.concatenate((out_array, nparray))
        out_array.astype("float32")
        # Save array to .f32 file
        files = [f for f in os.listdir("data")]
        output = name + "Epoch" + str(epoch) + ".f32"
        while output in files:
            epoch += 1
            output = name + "Epoch" + str(epoch) + ".f32"
        output = os.path.join("data", output)
        out_array.tofile(output)

    ###
    tensors = [ndarray_to_bytes(ndarray) for ndarray in weights]
    return Parameters(tensors=tensors, tensor_type="numpy.ndarray")


def parameters_to_weights(parameters: Parameters) -> Weights:
    """Convert parameters object to NumPy weights."""
    return [bytes_to_ndarray(tensor) for tensor in parameters.tensors]


def ndarray_to_bytes(ndarray: np.ndarray) -> bytes:
    """Serialize NumPy ndarray to bytes."""
    bytes_io = BytesIO()
    np.save(bytes_io, ndarray, allow_pickle=False)
    ## Try saving here
    return bytes_io.getvalue()


def bytes_to_ndarray(tensor: bytes) -> np.ndarray:
    """Deserialize NumPy ndarray from bytes."""
    bytes_io = BytesIO(tensor)
    ndarray_deserialized = np.load(bytes_io, allow_pickle=False)
    return cast(np.ndarray, ndarray_deserialized)
