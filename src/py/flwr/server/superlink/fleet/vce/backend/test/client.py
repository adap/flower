# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""A ClientApp for Backend tests."""

from math import pi
from typing import Dict

from flwr.client import Client, NumPyClient
from flwr.client.clientapp import ClientApp
from flwr.common import Config, ConfigsRecord, Scalar


class DummyClient(NumPyClient):
    """A dummy NumPyClient for tests."""

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        """Return properties by doing a simple calculation."""
        result = float(config["factor"]) * pi

        # store something in context
        self.context.state.configs_records["result"] = ConfigsRecord({"result": result})
        return {"result": result}


def get_dummy_client(cid: str) -> Client:  # pylint: disable=unused-argument
    """Return a DummyClient converted to Client type."""
    return DummyClient().to_client()


def _load_app() -> ClientApp:
    return ClientApp(client_fn=get_dummy_client)


client_app = ClientApp(
    client_fn=get_dummy_client,
)
