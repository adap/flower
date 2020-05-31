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


def test_sample_real_delay_factors_100():
    """Test delay factors."""
    # Prepare
    num_clients = 100

    # Execute
    factors = sample_real_delay_factors(num_clients=num_clients)
    print()
    print(factors[:50])
    print(factors[50:])

    # Assert
    assert len(factors) == num_clients


def test_sample_real_delay_factors_10():
    """Test delay factors."""
    # Prepare
    num_clients = 10

    # Execute
    factors = sample_real_delay_factors(num_clients=num_clients)
    print()
    print(factors[:50])
    print(factors[50:])

    # Assert
    assert len(factors) == num_clients
