# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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
"""Utilities for Actors in the Virtual Client Engine."""

import traceback
import warnings
from logging import ERROR

from flwr.client import Client
from flwr.common.logger import log

try:
    import tensorflow as TF
except ModuleNotFoundError:
    TF = None

# Display Deprecation warning once
warnings.filterwarnings("once", category=DeprecationWarning)


def enable_tf_gpu_growth() -> None:
    """Enable GPU memory growth to prevent premature OOM."""
    # By default, TF maps all GPU memory to the process.
    # We don't want this behavior in simulation, since it prevents us
    # from having multiple Actors (and therefore Flower clients) sharing
    # the same GPU.
    # Luckily we can disable this behavior by enabling memory growth
    # on the GPU. In this way, VRAM allocated to the processes grows based
    # on the needs for the workload. (this is for instance the default
    # behavior in PyTorch)
    # While this behavior is critical for Actors, you'll likely need it
    # as well in your main process (where the server runs and might evaluate
    # the state of the global model using GPU.)

    if TF is None:
        raise ImportError("TensorFlow is not found.")

    # This bit of code follows the guidelines for GPU usage
    # in https://www.tensorflow.org/guide/gpu
    gpus = TF.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                TF.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as ex:
            # Memory growth must be set before GPUs have been initialized
            log(ERROR, traceback.format_exc())
            log(ERROR, ex)
            raise ex


def check_clientfn_returns_client(client: Client) -> Client:
    """Warn once that clients returned in `clinet_fn` should be of type Client.

    This is here for backwards compatibility. If a ClientFn is provided returning
    a different type of client (e.g. NumPyClient) we'll warn the user but convert
    the client internally to `Client` by calling `.to_client()`.
    """
    if not isinstance(client, Client):
        mssg = (
            " Ensure your client is of type `Client`. Please convert it"
            " using the `.to_client()` method before returning it"
            " in the `client_fn` you pass to `start_simulation`."
            " We have applied this conversion on your behalf."
            " Not returning a `Client` might trigger an error in future"
            " versions of Flower."
        )

        warnings.warn(mssg, DeprecationWarning, stacklevel=2)
        client = client.to_client()
    return client
