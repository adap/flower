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
"""Flower client."""


from .app import start_client as start_client
from .app import start_keras_client as start_keras_client
from .app import start_numpy_client as start_numpy_client
from .client import Client as Client
from .keras_client import KerasClient as KerasClient
from .numpy_client import NumPyClient as NumPyClient

__all__ = [
    "start_client",
    "start_keras_client",
    "start_numpy_client",
    "Client",
    "KerasClient",
    "NumPyClient",
]
