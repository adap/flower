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


from .address import parse_address


def test_ipv4_correct() -> None:
    """Test if a correct IPv4 address is correctly parsed."""

    # Prepare
    addresses = [
        ("127.0.0.1:8080", ("127.0.0.1", 8080, False)),
        ("0.0.0.0:12", ("0.0.0.0", 12, False)),
        ("flower.dev:123", ("flower.dev", 123, False)),
        ("flower.dev:123", ("flower.dev", 123, False)),
        ("sub.flower.dev:123", ("sub.flower.dev", 123, False)),
        ("sub2.sub1.flower.dev:123", ("sub2.sub1.flower.dev", 123, False)),
        ("s5.s4.s3.s2.s1.flower.dev:123", ("s5.s4.s3.s2.s1.flower.dev", 123, False)),
        ("localhost:123", ("localhost", 123, False)),
    ]

    for address, expected in addresses:
        # Execute
        actual = parse_address(address)

        # Assert
        assert actual == expected


def test_ipv4_incorrect() -> None:
    """Test if an incorrect IPv4 address returns None."""

    # Prepare
    addresses = [
        "127.0.0.1::8080",
        "0.0.0.0.0:12",
        "1112.0.0.0:50",
        "127.0.0.1",
        "42.1.1.0:9988898",
    ]

    for address in addresses:
        # Execute
        actual = parse_address(address)

        # Assert
        assert actual is None


def test_ipv6_correct() -> None:
    """Test if a correct IPv6 address is correctly parsed."""

    # Prepare
    addresses = [
        ("[::1]:8080", ("::1", 8080, True)),
        ("[::]:12", ("::", 12, True)),
        (
            "2001:db8:3333:4444:5555:6666:7777:8888:12",
            ("2001:db8:3333:4444:5555:6666:7777:8888", 12, True),
        ),
        (
            "[0000:0000:0000:0000:0000:0000:0000:0001]:443",
            ("0000:0000:0000:0000:0000:0000:0000:0001", 443, True),
        ),
        ("[::]:123", ("::", 123, True)),
        ("[0:0:0:0:0:0:0:1]:80", ("0:0:0:0:0:0:0:1", 80, True)),
        ("[::1]:80", ("::1", 80, True)),
    ]

    for address, expected in addresses:
        # Execute
        actual = parse_address(address)

        # Assert
        assert actual == expected


def test_ipv6_incorrect() -> None:
    """Test if an incorrect IPv6 address returns None."""

    # Prepare
    addresses = [
        "2001:db8:3333:4444:5555:6666:7777:8888::8080",
        "999999:db8:3333:4444:5555:6666:7777:8888:50",
        "2001:db8:3333:4444:5555:6666:7777:8888",
        "2001:db8:3333:4444:5555:6666:7777:8888:9988898",
    ]

    for address in addresses:
        # Execute
        actual = parse_address(address)

        # Assert
        assert actual is None
