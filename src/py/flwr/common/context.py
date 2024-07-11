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
"""Context."""


from dataclasses import dataclass
from typing import Dict, Optional

from .record import RecordSet
from .typing import Value


@dataclass
class Context:
    """Context of your run.

    Parameters
    ----------
    state : RecordSet
        Holds records added by the entity in a given run and that will stay local.
        This means that the data it holds will never leave the system it's running from.
        This can be used as an intermediate storage or scratchpad when
        executing mods. It can also be used as a memory to access
        at different points during the lifecycle of this entity (e.g. across
        multiple rounds)
    run_config : Dict[str, Value]
        A config (key/value mapping) held by the entity in a given run and that will
        stay local. It can be used at any point during the lifecycle of this entity
        (e.g. across multiple rounds)
    partition_id : Optional[int] (default: None)
        An index that specifies the data partition that the ClientApp using this Context
        object should make use of. Setting this attribute is better suited for
        simulation or proto typing setups.
    """

    state: RecordSet
    partition_id: Optional[int]
    run_config: Dict[str, Value]

    def __init__(
        self,
        state: RecordSet,
        run_config: Dict[str, Value],
        partition_id: Optional[int] = None,
    ) -> None:
        self.state = state
        self.run_config = run_config
        self.partition_id = partition_id
