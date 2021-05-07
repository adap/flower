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
"""Networked Flower client implementation."""

import threading
from typing import Optional

from flwr import common
from flwr.client import Client
from flwr.server.client_proxy import ClientProxy


class InMemoryClientProxy(ClientProxy):
    """Flower client proxy which delegates over the network using gRPC."""

    def __init__(self, cid: str, client: Client, lock: Optional[threading.Lock] = None):
        super().__init__(cid)
        self.client = client
        self._cv = lock

    def get_parameters(self) -> common.ParametersRes:
        """Return the current local model parameters."""
        if self._cv:
            with self._cv:
                return self.client.get_parameters()

        return self.client.get_parameters()

    def fit(self, ins: common.FitIns) -> common.FitRes:
        """Refine the provided weights using the locally held dataset."""
        if self._cv:
            with self._cv:
                return self.client.fit(ins=ins)

        return self.client.fit(ins=ins)

    def evaluate(self, ins: common.EvaluateIns) -> common.EvaluateRes:
        """Evaluate the provided weights using the locally held dataset."""
        if self._cv:
            with self._cv:
                return self.client.evaluate(ins=ins)

        return self.client.evaluate(ins=ins)

    def reconnect(self, reconnect: common.Reconnect) -> common.Disconnect:
        """Disconnect and (optionally) reconnect later."""
        if self._cv:
            with self._cv:
                return common.Disconnect(reason="unknown")

        return common.Disconnect(reason="unknown")
