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
"""Strategy tests"""


from .strategy import DefaultStrategy


def test_default_strategy_evaluate_every_round() -> None:
    """Test evaluate function."""

    # Prepare
    strategy = DefaultStrategy()

    # Execute & assert
    for _ in range(5):
        assert strategy.evaluate()


def test_default_strategy_num_fit_clients_40_available() -> None:
    """Test num_fit_clients function."""

    # Prepare
    strategy = DefaultStrategy()
    expected = 4

    # Execute
    actual = strategy.num_fit_clients(num_available_clients=40)

    # Assert
    assert expected == actual


def test_default_strategy_num_fit_clients_39_available() -> None:
    """Test num_fit_clients function."""

    # Prepare
    strategy = DefaultStrategy()
    expected = 3

    # Execute
    actual = strategy.num_fit_clients(num_available_clients=39)

    # Assert
    assert expected == actual


def test_default_strategy_num_fit_clients_30_available() -> None:
    """Test num_fit_clients function."""

    # Prepare
    strategy = DefaultStrategy()
    expected = 3

    # Execute
    actual = strategy.num_fit_clients(num_available_clients=30)

    # Assert
    assert expected == actual


def test_default_strategy_num_fit_clients_minimum() -> None:
    """Test num_fit_clients function."""

    # Prepare
    strategy = DefaultStrategy()
    expected = 3

    # Execute
    actual = strategy.num_fit_clients(num_available_clients=29)

    # Assert
    assert expected == actual


def test_default_strategy_num_evaluation_clients_80_available() -> None:
    """Test num_evaluation_clients function."""

    # Prepare
    strategy = DefaultStrategy()
    expected = 4

    # Execute
    actual = strategy.num_evaluation_clients(num_available_clients=80)

    # Assert
    assert expected == actual


def test_default_strategy_num_evaluation_clients_79_available() -> None:
    """Test num_evaluation_clients function."""

    # Prepare
    strategy = DefaultStrategy()
    expected = 3

    # Execute
    actual = strategy.num_evaluation_clients(num_available_clients=79)

    # Assert
    assert expected == actual


def test_default_strategy_num_evaluation_clients_60_available() -> None:
    """Test num_evaluation_clients function."""

    # Prepare
    strategy = DefaultStrategy()
    expected = 3

    # Execute
    actual = strategy.num_evaluation_clients(num_available_clients=60)

    # Assert
    assert expected == actual


def test_default_strategy_num_evaluation_clients_minimum() -> None:
    """Test num_evaluation_clients function."""

    # Prepare
    strategy = DefaultStrategy()
    expected = 3

    # Execute
    actual = strategy.num_evaluation_clients(num_available_clients=59)

    # Assert
    assert expected == actual
