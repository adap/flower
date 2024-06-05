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
"""Executes and monitor a Flower run."""

from abc import ABC, abstractmethod
from typing import Optional


class Executor(ABC):
    """Executes and monitor a Flower run."""

    @abstractmethod
    def start_run(
        self,
        fab_path: str,
        ttl: Optional[float] = None,
    ) -> int:
        """Start a run using the given Flower App ID and version.

        This method creates a new run on the SuperLink and returns its run_id
        and also starts the run execution.

        Parameters
        ----------
        fab_path : str
            The path to the Flower App Bundle file.
        ttl : Optional[float] (default: None)
            Time-to-live for the round trip of this message, i.e., the time from sending
            this message to receiving a reply. It specifies in seconds the duration for
            which the message and its potential reply are considered valid. If unset,
            the default TTL (i.e., `common.DEFAULT_TTL`) will be used.

        Returns
        -------
        run_id : int
            The run_id of the run created by the SuperLink.
        """
