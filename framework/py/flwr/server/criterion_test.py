# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Tests for criterion sampling."""


from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion


class TestClientProxy(ClientProxy):
    """Minimal ClientProxy test double."""

    def get_properties(self, ins, timeout, group_id):
        """Not used in this test module."""
        raise NotImplementedError

    def get_parameters(self, ins, timeout, group_id):
        """Not used in this test module."""
        raise NotImplementedError

    def fit(self, ins, timeout, group_id):
        """Not used in this test module."""
        raise NotImplementedError

    def evaluate(self, ins, timeout, group_id):
        """Not used in this test module."""
        raise NotImplementedError

    def reconnect(self, ins, timeout, group_id):
        """Not used in this test module."""
        raise NotImplementedError


def test_criterion_applied() -> None:
    """Test sampling w/ criterion."""
    # Prepare
    client1 = TestClientProxy(cid="train_client_1")
    client2 = TestClientProxy(cid="train_client_2")
    client3 = TestClientProxy(cid="test_client_1")
    client4 = TestClientProxy(cid="test_client_2")

    client_manager = SimpleClientManager()
    client_manager.register(client1)
    client_manager.register(client2)
    client_manager.register(client3)
    client_manager.register(client4)

    class TestCriterion(Criterion):
        """Criterion to select only test clients."""

        def select(self, client: ClientProxy) -> bool:
            return client.cid.startswith("test_")

    # Execute
    sampled_clients = client_manager.sample(2, criterion=TestCriterion())

    # Assert
    assert client3 in sampled_clients
    assert client4 in sampled_clients


def test_criterion_not_applied() -> None:
    """Test sampling w/o criterion."""
    # Prepare
    client1 = TestClientProxy(cid="train_client_1")
    client2 = TestClientProxy(cid="train_client_2")
    client3 = TestClientProxy(cid="test_client_1")
    client4 = TestClientProxy(cid="test_client_2")

    client_manager = SimpleClientManager()
    client_manager.register(client1)
    client_manager.register(client2)
    client_manager.register(client3)
    client_manager.register(client4)

    # Execute
    sampled_clients = client_manager.sample(4)

    # Assert
    assert client1 in sampled_clients
    assert client2 in sampled_clients
    assert client3 in sampled_clients
    assert client4 in sampled_clients
