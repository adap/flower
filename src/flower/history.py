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
"""Training history."""

from functools import reduce
from typing import List, Tuple


class History:
    """History class for training and/or evaluation metrics collection."""

    def __init__(self) -> None:
        self.losses: List[Tuple[int, float]] = []

    def add_loss(self, rnd: int, loss: float) -> None:
        """Add one loss entry."""
        self.losses.append((rnd, loss))

    def __repr__(self) -> str:
        return "History:\n" + reduce(
            lambda a, b: a + b,
            [f"\tround {rnd}: {loss}\n" for rnd, loss in self.losses],
        )
