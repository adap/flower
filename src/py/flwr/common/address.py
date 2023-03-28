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
"""Flower IP address utils."""


from ipaddress import ip_address
from typing import Tuple, Union


def parse_address(address: str) -> Union[Tuple[str, int, bool], None]:
    try:
        raw_host, _, raw_port = address.rpartition(":")
        host, port = raw_host.translate({ord(i): None for i in "[]"}), int(raw_port)
        if port > 65535:
            raise ValueError("Port number is too high.")
        return host, port, ip_address(host).version == 6
    except ValueError as _:
        return None
