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
"""Implements tests for config module."""

from .config import sample_real_delay_factors


def test_sample_real_delay_factors_100() -> None:
    """Test delay factors."""
    # Prepare
    num_clients = 100

    # Execute
    factors = sample_real_delay_factors(num_clients=num_clients)

    # Assert
    assert len(factors) == num_clients


def test_sample_real_delay_factors_10() -> None:
    """Test delay factors."""
    # Prepare
    num_clients = 10

    # Execute
    factors = sample_real_delay_factors(num_clients=num_clients)

    # Assert
    assert len(factors) == num_clients


def test_sample_real_delay_factors_seed() -> None:
    """Test delay factors."""
    # Prepare
    num_clients = 100

    # Execute
    factors_a = sample_real_delay_factors(num_clients=num_clients, seed=0)
    factors_b = sample_real_delay_factors(num_clients=num_clients, seed=0)
    factors_c = sample_real_delay_factors(num_clients=num_clients, seed=1)

    # Assert
    assert len(factors_a) == num_clients
    assert len(factors_b) == num_clients
    assert len(factors_c) == num_clients

    # pylint: disable=invalid-name
    all_same_in_a_and_b = True
    all_same_in_a_and_c = True

    for a, b, c in zip(factors_a, factors_b, factors_c):
        all_same_in_a_and_b = all_same_in_a_and_b and (a == b)
        all_same_in_a_and_c = all_same_in_a_and_c and (a == c)

    assert all_same_in_a_and_b
    assert not all_same_in_a_and_c
