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
"""This module contains functions related to proto compilation."""


from os import path

from .protoc import IN_PATH, OUT_PATH, PROTO_FILES


def test_directories() -> None:
    """Test if all directories exist."""
    assert path.isdir(IN_PATH)
    assert path.isdir(OUT_PATH)


def test_proto_file_count() -> None:
    """Test if the correct number of proto files were captured by the glob."""
    assert len(PROTO_FILES) == 9
