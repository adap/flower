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
"""Flower callable."""


import importlib
from typing import List, Optional, cast

from flwr.client.message_handler.message_handler import handle
from flwr.client.middleware.typing import Layer
from flwr.client.middleware.utils import make_app
from flwr.client.typing import Bwd, ClientFn, Fwd
from flwr.client.workload_state import WorkloadState


class Flower:
    """Flower callable.

    Examples
    --------
    Assuming a typical client implementation in `FlowerClient`, you can wrap it in a
    Flower callable as follows:

    >>> class FlowerClient(NumPyClient):
    >>>     # ...
    >>>
    >>> def client_fn(cid):
    >>>    return FlowerClient().to_client()
    >>>
    >>> flower = Flower(client_fn)

    If the above code is in a Python module called `client`, it can be started as
    follows:

    >>> flower-client --callable client:flower

    In this `client:flower` example, `client` refers to the Python module in which the
    previous code lives in. `flower` refers to the global attribute `flower` that points
    to an object of type `Flower` (a Flower callable).
    """

    def __init__(
        self,
        client_fn: ClientFn,  # Only for backward compatibility
        middleware: Optional[List[Layer]] = None,
    ) -> None:
        self.client_fn = client_fn
        self.mw_list = middleware if middleware is not None else []

    def __call__(self, fwd: Fwd) -> Bwd:
        """."""

        # Create wrapper function for `handle`
        def handle_app(_fwd: Fwd) -> Bwd:
            task_res = handle(
                client_fn=self.client_fn,
                task_ins=_fwd.task_ins,
            )
            return Bwd(task_res=task_res, state=WorkloadState(state={}))

        # Wrap middleware layers around handle_app
        app = make_app(handle_app, self.mw_list)

        # Execute the task
        bwd = app(fwd)

        return bwd


class LoadCallableError(Exception):
    """."""


def load_callable(module_attribute_str: str) -> Flower:
    """Load the `Flower` object specified in a module attribute string.

    The module/attribute string should have the form <module>:<attribute>. Valid
    examples include `client:flower` and `project.package.module:wrapper.flower`. It
    must refer to a module on the PYTHONPATH, the module needs to have the specified
    attribute, and the attribute must be of type `Flower`.
    """
    module_str, _, attributes_str = module_attribute_str.partition(":")
    if not module_str:
        raise LoadCallableError(
            f"Missing module in {module_attribute_str}",
        ) from None
    if not attributes_str:
        raise LoadCallableError(
            f"Missing attribute in {module_attribute_str}",
        ) from None

    # Load module
    try:
        module = importlib.import_module(module_str)
    except ModuleNotFoundError:
        raise LoadCallableError(
            f"Unable to load module {module_str}",
        ) from None

    # Recursively load attribute
    attribute = module
    try:
        for attribute_str in attributes_str.split("."):
            attribute = getattr(attribute, attribute_str)
    except AttributeError:
        raise LoadCallableError(
            f"Unable to load attribute {attributes_str} from module {module_str}",
        ) from None

    # Check type
    if not isinstance(attribute, Flower):
        raise LoadCallableError(
            f"Attribute {attributes_str} is not of type {Flower}",
        ) from None

    return cast(Flower, attribute)
