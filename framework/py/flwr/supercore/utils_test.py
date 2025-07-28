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
"""Tests for utility functions for the infrastructure."""


from .utils import mask_string


def test_mask_string() -> None:
    """Test the `mask_string` function."""
    assert mask_string("abcdefghi") == "abcd...fghi"
    assert mask_string("abcdefghijklm") == "abcd...jklm"
    assert mask_string("abc") == "abc"
    assert mask_string("a") == "a"
    assert mask_string("") == ""
    assert mask_string("1234567890", head=2, tail=3) == "12...890"
    assert mask_string("1234567890", head=5, tail=4) == "12345...7890"
