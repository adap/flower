# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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
        ("0.0.0.0:65535", ("0.0.0.0", 65535, False)),
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
        "127.0.0.1",
        "42.1.1.0:9988898",
        "0.0.0.0:-999999",
        "0.0.0.0:-1",
        "0.0.0.0:0",
        "0.0.0.0:65536",
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
            "2001:db8:3333:4444:5555:6666:7777:8888:65535",
            ("2001:db8:3333:4444:5555:6666:7777:8888", 65535, True),
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
        "2001:db8:3333:4444:5555:6666:7777:8888:9988898",
        "2001:db8:3333:4444:5555:6666:7777:8888:-9988898",
        "2001:db8:3333:4444:5555:6666:7777:8888:-1",
        "2001:db8:3333:4444:5555:6666:7777:8888:0",
        "2001:db8:3333:4444:5555:6666:7777:8888:65536",
    ]

    for address in addresses:
        # Execute
        actual = parse_address(address)

        # Assert
        assert actual is None


def test_domain_correct() -> None:
    """Test if a correct domain address is correctly parsed."""
    # Prepare
    addresses = [
        ("flower.ai:123", ("flower.ai", 123, None)),
        ("sub.flower.ai:123", ("sub.flower.ai", 123, None)),
        ("sub2.sub1.flower.ai:123", ("sub2.sub1.flower.ai", 123, None)),
        ("s5.s4.s3.s2.s1.flower.ai:123", ("s5.s4.s3.s2.s1.flower.ai", 123, None)),
        ("localhost:123", ("localhost", 123, None)),
        ("https://localhost:123", ("https://localhost", 123, None)),
        ("http://localhost:123", ("http://localhost", 123, None)),
    ]

    for address, expected in addresses:
        # Execute
        actual = parse_address(address)

        # Assert
        assert actual == expected


def test_domain_incorrect() -> None:
    """Test if an incorrect domain address returns None."""
    # Prepare
    addresses = [
        "flower.ai",
        "flower.ai:65536",
    ]

    for address in addresses:
        # Execute
        actual = parse_address(address)

        # Assert
        assert actual is None
