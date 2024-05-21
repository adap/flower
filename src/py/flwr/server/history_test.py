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
"""Tests for class History."""


from flwr.server.history import History


def test_add_loss_distributed() -> None:
    """Test add_loss_distributed."""
    # Prepare
    history = History()

    # Execute
    history.add_loss_distributed(server_round=0, loss=2.0)

    # Assert
    assert len(history.losses_distributed) == 1
    assert len(history.losses_centralized) == 0
    assert len(history.metrics_distributed) == 0
    assert len(history.metrics_centralized) == 0
    assert (0, 2.0) == history.losses_distributed[0]


def test_add_loss_centralized() -> None:
    """Test add_loss_centralized."""
    # Prepare
    history = History()

    # Execute
    history.add_loss_centralized(server_round=0, loss=2.0)

    # Assert
    assert len(history.losses_distributed) == 0
    assert len(history.losses_centralized) == 1
    assert len(history.metrics_distributed) == 0
    assert len(history.metrics_centralized) == 0
    assert (0, 2.0) == history.losses_centralized[0]


def test_add_metrics_distributed() -> None:
    """Test add_metrics_distributed."""
    # Prepare
    history = History()

    # Execute
    history.add_metrics_distributed(server_round=0, metrics={"acc": 0.9})

    # Assert
    assert len(history.losses_distributed) == 0
    assert len(history.losses_centralized) == 0
    assert len(history.metrics_distributed) == 1
    assert len(history.metrics_centralized) == 0
    assert (0, 0.9) == history.metrics_distributed["acc"][0]


def test_add_metrics_centralized() -> None:
    """Test add_metrics_centralized."""
    # Prepare
    history = History()

    # Execute
    history.add_metrics_centralized(server_round=0, metrics={"acc": 0.9})

    # Assert
    assert len(history.losses_distributed) == 0
    assert len(history.losses_centralized) == 0
    assert len(history.metrics_distributed) == 0
    assert len(history.metrics_centralized) == 1
    assert (0, 0.9) == history.metrics_centralized["acc"][0]
