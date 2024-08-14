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
"""."""


from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import Message
from flwr.common.typing import Run


class Connection(ABC):

    @abstractmethod
    def create_node() -> Optional[int]:
        ...
    
    @abstractmethod
    def delete_node() -> None:
        ...
        
    @abstractmethod
    def receive() -> Optional[Message]:
        ...
    
    @abstractmethod
    def send(message: Message) -> None:
        ...
            
    @abstractmethod
    def get_run(run_id: int) -> Run:
        ...