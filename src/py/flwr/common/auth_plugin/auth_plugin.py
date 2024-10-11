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
"""Abstract classes for Flower User Auth Plugin."""

from abc import ABC, abstractmethod
from flwr.proto.exec_pb2 import LoginRequest, LoginResponse
from typing import Sequence, Tuple, Union, Any, List, Dict

Metadata = List[Any]


class SuperExecAuthPlugin(ABC):
    """Abstract Flower SuperExec Auth Plugin class."""
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        pass
            
    @abstractmethod
    def send_auth_endpoint(self) -> LoginResponse:
        """
        Send relevant auth url as a LoginResponse.
        """
        pass

    @abstractmethod
    def authenticate(self, metadata: Sequence[Tuple[str, Union[str, bytes]]]):
        """
        Authenticate auth tokens in the provided metadata.
        """
        pass


class UserAuthPlugin(ABC):
    """Abstract Flower User Auth Plugin class."""
    @staticmethod
    @abstractmethod
    def login(auth_url: str, config: Dict[str, Any], federation: str):
        """
        Read relevant auth details from federation config.
        """
        pass

    @abstractmethod
    def __init__(self, config: Dict[str, Any], federation: str):
        """Abstract constructor (init)"""
        pass

    @abstractmethod
    def provide_auth_details(self, metadata) -> Metadata:
        """
        Provide relevant auth tokens in the metadata.
        """
        pass
