# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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


import pytest

from .address import get_ip_address_from_servicer_context, parse_address


@pytest.mark.parametrize(
    "address, expected",
    [
        ("127.0.0.1:8080", ("127.0.0.1", 8080, False)),
        ("0.0.0.0:12", ("0.0.0.0", 12, False)),
        ("0.0.0.0:65535", ("0.0.0.0", 65535, False)),
    ],
)
def test_ipv4_correct(address: str, expected: tuple[str, int, bool]) -> None:
    """Test if a correct IPv4 address is correctly parsed."""
    # Execute
    actual = parse_address(address)

    # Assert
    assert actual == expected


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",  # Missing port
        "42.1.1.0:9988898",  # Port number out of range
        "0.0.0.0:-999999",  # Negative port number
        "0.0.0.0:-1",  # Negative port number
        "0.0.0.0:0",  # Port number zero
        "0.0.0.0:65536",  # Port number out of range
    ],
)
def test_ipv4_incorrect(address: str) -> None:
    """Test if an incorrect IPv4 address returns None."""
    # Execute
    actual = parse_address(address)

    # Assert
    assert actual is None


@pytest.mark.parametrize(
    "address, expected",
    [
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
    ],
)
def test_ipv6_correct(address: str, expected: tuple[str, int, bool]) -> None:
    """Test if a correct IPv6 address is correctly parsed."""
    # Execute
    actual = parse_address(address)

    # Assert
    assert actual == expected


@pytest.mark.parametrize(
    "address",
    [
        "[2001:db8:3333:4444:5555:6666:7777:8888]:9988898",  # Port number out of range
        "[2001:db8:3333:4444:5555:6666:7777:8888]:-9988898",  # Negative port number
        "[2001:db8:3333:4444:5555:6666:7777:8888]:-1",  # Negative port number
        "[2001:db8:3333:4444:5555:6666:7777:8888]:0",  # Port number zero
        "[2001:db8:3333:4444:5555:6666:7777:8888]:65536",  # Port number out of range
    ],
)
def test_ipv6_incorrect(address: str) -> None:
    """Test if an incorrect IPv6 address returns None."""
    # Execute
    actual = parse_address(address)

    # Assert
    assert actual is None


@pytest.mark.parametrize(
    "address, expected",
    [
        ("flower.ai:123", ("flower.ai", 123, None)),
        ("sub.flower.ai:123", ("sub.flower.ai", 123, None)),
        ("sub2.sub1.flower.ai:123", ("sub2.sub1.flower.ai", 123, None)),
        ("s5.s4.s3.s2.s1.flower.ai:123", ("s5.s4.s3.s2.s1.flower.ai", 123, None)),
        ("localhost:123", ("localhost", 123, None)),
        ("https://localhost:123", ("https://localhost", 123, None)),
        ("http://localhost:123", ("http://localhost", 123, None)),
    ],
)
def test_domain_correct(address: str, expected: tuple[str, int, bool]) -> None:
    """Test if a correct domain address is correctly parsed."""
    # Execute
    actual = parse_address(address)

    # Assert
    assert actual == expected


@pytest.mark.parametrize(
    "address",
    [
        "flower.ai",  # Missing port
        "flower.ai:65536",  # Port number out of range
    ],
)
def test_domain_incorrect(address: str) -> None:
    """Test if an incorrect domain address returns None."""
    # Execute
    actual = parse_address(address)

    # Assert
    assert actual is None


class DummyContext:
    """Dummy context to mimic grpc.ServicerContext for testing purposes."""

    def __init__(self, peer_str: str) -> None:
        self._peer = peer_str

    def peer(self) -> str:
        """."""
        return self._peer


@pytest.mark.parametrize(
    "peer_str, expected",
    [
        ("ipv4:127.0.0.1:56789", "127.0.0.1"),
        ("ipv4:0.0.0.0:8080", "0.0.0.0"),
        ("ipv4:192.168.1.1:12345", "192.168.1.1"),
    ],
)
def test_servicer_ipv4_correct(peer_str: str, expected: str) -> None:
    """Test if a correct IPv4 address is correctly parsed from grpc.ServicerContext."""
    # Prepare dummy context with the given peer string.
    context = DummyContext(peer_str)

    # Execute
    actual = get_ip_address_from_servicer_context(context)

    # Assert
    assert actual == expected


@pytest.mark.parametrize(
    "peer_str, expected",
    [
        ("ipv6:[2001:db8::1]:54321", "2001:db8::1"),
        ("ipv6:[::1]:8080", "::1"),
        ("ipv6:[fe80::1ff:fe23:4567:890a]:9999", "fe80::1ff:fe23:4567:890a"),
    ],
)
def test_servicer_ipv6_correct(peer_str: str, expected: str) -> None:
    """Test if a correct IPv6 address is correctly parsed from grpc.ServicerContext."""
    # Prepare dummy context with the given peer string.
    context = DummyContext(peer_str)

    # Execute
    actual = get_ip_address_from_servicer_context(context)

    # Assert
    assert actual == expected


@pytest.mark.parametrize(
    "peer_str",
    [
        "invalid_string",
        "ipv4:127.0.0.1",  # missing port
        "ipv6:2001:db8::1",  # missing brackets and port
        "ipv6:[2001:db8::1]56789",  # missing colon after the bracket
        "ipv6:2001:db8::1:54321",  # missing brackets
        "unix:/tmp/grpc.sock",  # unix domain socket
        "",
    ],
)
def test_servicer_incorrect_format(peer_str: str) -> None:
    """Test if an invalid grpc.ServicerContext.peer() string returns a ValueError."""
    # Prepare dummy context with the given peer string.
    context = DummyContext(peer_str)

    # Execute and Assert
    with pytest.raises(ValueError):
        get_ip_address_from_servicer_context(context)
