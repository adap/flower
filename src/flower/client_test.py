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
"""Flower client (abstract base class)"""

from flower.client import NetworkClient


def test_get_weights():
    """This test is currently quite simple and should be improved"""
    # Prepare
    client = NetworkClient(cid="1")

    # Execute
    value = client.get_weights()

    # Assert
    assert [] == value


def test_fit():
    """This test is currently quite simple and should be improved"""
    # Prepare
    client = NetworkClient(cid="1")

    # Execute
    value = client.fit([])

    # Assert
    assert ([], 1) == value


def test_evaluate():
    """This test is currently quite simple and should be improved"""
    # Prepare
    client = NetworkClient(cid="1")

    # Execute
    value = client.evaluate([])

    # Assert
    assert (1, 1.0) == value
