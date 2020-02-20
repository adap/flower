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
"""Flower client tests"""

import pytest

from .client import Client


def test_method_get_weights_not_implemented() -> None:
    """Test method get_weights raises NotImplementedError"""
    # Prepare
    client = Client(cid="1")

    # Execute & assert
    with pytest.raises(NotImplementedError):
        client.get_weights()


def test_method_fit_not_implemented() -> None:
    """Test method fit raises NotImplementedError"""
    # Prepare
    client = Client(cid="1")

    # Execute & assert
    with pytest.raises(NotImplementedError):
        client.fit(weights=[])


def test_method_evaluate_not_implemented() -> None:
    """Test method fit raises NotImplementedError"""
    # Prepare
    client = Client(cid="1")

    # Execute & assert
    with pytest.raises(NotImplementedError):
        client.evaluate(weights=[])
