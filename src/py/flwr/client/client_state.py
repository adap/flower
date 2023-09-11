# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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
"""Client state."""

from dataclasses import dataclass


@dataclass
class ClientState:
    """Client state definition as a standard Python dataclass."""

    cid: str

    def __repr__(self) -> str:
        """Return a string representation of a ClientState."""
        return f"{self.__class__.__name__}(cid={self.cid}): {self.__dict__}"
