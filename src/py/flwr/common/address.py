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


import re
from ipaddress import ip_address
from typing import Optional, Tuple

IPV6: int = 6

DOMAIN_PATTERN: re.Pattern = re.compile(
    r"^(localhost)|(([a-zA-Z]{1})|([a-zA-Z]{1}[a-zA-Z]{1})|"
    r"([a-zA-Z]{1}[0-9]{1})|([0-9]{1}[a-zA-Z]{1})|"
    r"([a-zA-Z0-9][-_.a-zA-Z0-9]{0,61}[a-zA-Z0-9]))\."
    r"([a-zA-Z]{2,13}|[a-zA-Z0-9-]{2,30}.[a-zA-Z]{2,3})$"
)


def parse_address(address: str) -> Optional[Tuple[str, int, bool]]:
    """Parses an IP address into a host, port, and version.

    Parameters
    ----------
    address : str
        The string representation of a domain, an IPv4, or an IPV6 address
        with the port number.

        For example, '127.0.0.1:8080', or [::1]:8080.

    Returns
    -------
    Optional[Tuple[str, int, bool]]
        If the string provided is not a correct IPv6 or IPv4,
        the function will return None, otherwise it will return the host,
        as a string, the port number, as an int, and
        a bool that is True if the address is IPv6 and False otherwise.
    """
    try:
        raw_host, _, raw_port = address.rpartition(":")

        if DOMAIN_PATTERN.match(raw_host):
            host = raw_host
            version = False
        else:
            host = raw_host.translate({ord(i): None for i in "[]"})
            version = ip_address(host).version == IPV6

        port = int(raw_port)

        if port > 65535 or port < 0:
            raise ValueError("Port number is invalid.")

        return host, port, version

    except ValueError:
        return None
