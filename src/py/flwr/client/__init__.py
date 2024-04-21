# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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
"""Flower client."""


from .app import run_client_app as run_client_app
from .app import start_client as start_client
from .app import start_numpy_client as start_numpy_client
from .client import Client as Client
from .client_app import ClientApp as ClientApp
from .numpy_client import NumPyClient as NumPyClient
from .supernode import run_supernode as run_supernode
from .typing import ClientFn as ClientFn

__all__ = [
    "Client",
    "ClientApp",
    "ClientFn",
    "NumPyClient",
    "run_client_app",
    "run_supernode",
    "start_client",
    "start_numpy_client",
]
