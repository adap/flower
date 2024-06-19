# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Flower ServerConfig."""


from dataclasses import dataclass
from typing import Optional


@dataclass
class ServerConfig:
    """Flower server config.

    All attributes have default values which allows users to configure just the ones
    they care about.
    """

    num_rounds: int = 1
    round_timeout: Optional[float] = None

    def __repr__(self) -> str:
        """Return the string representation of the ServerConfig."""
        timeout_string = (
            "no round_timeout"
            if self.round_timeout is None
            else f"round_timeout={self.round_timeout}s"
        )
        return f"num_rounds={self.num_rounds}, {timeout_string}"