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
        host, port = (lambda h, _, p: (h.translate({ord(i): None for i in "[]"}), int(p)))(*(address.rpartition(':')))
        return host, port, ip_address(host).version == 4
    except ValueError as error:
        return None
