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
"""Tests for utility functions for the infrastructure."""


import json
from typing import Any

import pytest
import requests
from parameterized import parameterized

from . import utils
from .constant import (
    FLWR_DISABLE_UPDATE_CHECK,
    FLWR_UPDATE_CHECK_CONNECT_TIMEOUT_SECONDS,
    FLWR_UPDATE_CHECK_READ_TIMEOUT_SECONDS,
    FLWR_UPDATE_CHECK_URL,
)
from .utils import (
    get_flwr_update_check_payload,
    humanize_bytes,
    humanize_duration,
    int64_to_uint64,
    mask_string,
    parse_app_spec,
    request_download_link,
    uint64_to_int64,
    warn_if_flwr_update_available,
)


def test_mask_string() -> None:
    """Test the `mask_string` function."""
    assert mask_string("abcdefghi") == "abcd...fghi"
    assert mask_string("abcdefghijklm") == "abcd...jklm"
    assert mask_string("abc") == "abc"
    assert mask_string("a") == "a"
    assert mask_string("") == ""
    assert mask_string("1234567890", head=2, tail=3) == "12...890"
    assert mask_string("1234567890", head=5, tail=4) == "12345...7890"


@parameterized.expand(  # type: ignore
    [
        # Test values within the positive range of sint64 (below 2^63)
        (0, 0),  # Minimum positive value
        (1, 1),  # 1 remains 1 in both uint64 and sint64
        (2**62, 2**62),  # Mid-range positive value
        (2**63 - 1, 2**63 - 1),  # Maximum positive value for sint64
        # Test values at or above 2^63 (become negative in sint64)
        (2**63, -(2**63)),  # Minimum negative value for sint64
        (2**63 + 1, -(2**63) + 1),  # Slightly above the boundary
        (9223372036854775811, -9223372036854775805),  # Some value > sint64 max
        (2**64 - 1, -1),  # Maximum uint64 value becomes -1 in sint64
    ]
)
def test_convert_uint64_to_sint64(before: int, after: int) -> None:
    """Test conversion from uint64 to sint64."""
    assert uint64_to_int64(before) == after


@parameterized.expand(  # type: ignore
    [
        # Test values within the negative range of sint64
        (-(2**63), 2**63),  # Minimum sint64 value becomes 2^63 in uint64
        (-(2**63) + 1, 2**63 + 1),  # Slightly above the minimum
        (-9223372036854775805, 9223372036854775811),  # Some value > sint64 max
        # Test zero-adjacent inputs
        (-1, 2**64 - 1),  # -1 in sint64 becomes 2^64 - 1 in uint64
        (0, 0),  # 0 remains 0 in both sint64 and uint64
        (1, 1),  # 1 remains 1 in both sint64 and uint64
        # Test values within the positive range of sint64
        (2**63 - 1, 2**63 - 1),  # Maximum positive value in sint64
        # Test boundary and maximum uint64 value
        (2**63, 2**63),  # Exact boundary value for sint64
        (2**64 - 1, 2**64 - 1),  # Maximum uint64 value, stays the same
    ]
)
def test_sint64_to_uint64(before: int, after: int) -> None:
    """Test conversion from sint64 to uint64."""
    assert int64_to_uint64(before) == after


@parameterized.expand(  # type: ignore
    [
        (0),
        (1),
        (2**62),
        (2**63 - 1),
        (2**63),
        (2**63 + 1),
        (9223372036854775811),
        (2**64 - 1),
    ]
)
def test_uint64_to_sint64_to_uint64(expected: int) -> None:
    """Test conversion from sint64 to uint64."""
    actual = int64_to_uint64(uint64_to_int64(expected))
    assert actual == expected


@pytest.mark.parametrize(
    "value",
    [
        "user/app==1.2.3",  # missing '@'
        "@accountapp==1.2.3",  # missing slash
        "@account/app==1.2",  # bad version
        "@account/app==1.2.3.4",  # bad version
        "@account*/app==1.2.3",  # bad user id chars
        "@account/app*==1.2.3",  # bad app id chars
    ],
)
def test_parse_app_spec_rejects_invalid_formats(value: str) -> None:
    """For an invalid string, the function should fail fast with ValueError."""
    with pytest.raises(ValueError):
        parse_app_spec(value)


def test_request_download_link_all_scenarios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single table-driven test covering all major outcomes for
    request_download_link."""
    app_id: str = "@user/app"
    app_version: str | None = "1.0.0"
    in_url = "https://example.ai/hub/fetch-fab"
    out_url = "fab_url"

    # Table of scenarios
    scenarios: list[dict[str, Any]] = [
        {
            "name": "success_with_verifications",
            "fake_resp": {
                "ok": True,
                "status": 200,
                "json": {
                    "fab_url": "https://example.ai/fab.fab",
                    "verifications": [
                        {"public_key_id": "key1", "sig": "abc", "algo": "ed25519"}
                    ],
                },
            },
            "assert": lambda out: (
                out[0] == "https://example.ai/fab.fab"
                and isinstance(out[1], list)
                and out[1][0]["public_key_id"] == "key1"
            ),
        },
        {
            "name": "success_without_verifications",
            "fake_resp": {
                "ok": True,
                "status": 200,
                "json": {"fab_url": "https://example.ai/fab.fab"},
            },
            "assert": lambda out: out[0] == "https://example.ai/fab.fab"
            and out[1] is None,
        },
        {
            "name": "http_404_not_found",
            "fake_resp": {
                "ok": False,
                "status": 404,
                "text": "not found",
                # No JSON body -> json() returns {}
            },
            "raises": "not found in Platform API",
        },
        {
            "name": "http_503_unavailable",
            "fake_resp": {
                "ok": False,
                "status": 503,
                "text": "service unavailable",
            },
            "raises": "status 503",
        },
        {
            "name": "network_error",
            "fake_exc": requests.RequestException("network down"),
            "raises": "Unable to connect to Platform API",
        },
        {
            "name": "missing_fab_url",
            "fake_resp": {
                "ok": True,
                "status": 200,
                "json": {"verifications": []},
            },
            "raises": "Invalid response from Platform API",
        },
    ]

    current_case: dict[str, Any] = {"data": None}

    class _FakeResp:
        ok: bool
        status_code: int
        _json: dict[str, Any]
        text: str

        def __init__(
            self,
            ok: bool,
            status: int,
            json_data: dict[str, Any] | None = None,
            text: str = "",
        ) -> None:
            self.ok = ok
            self.status_code = status
            self._json = json_data or {}
            self.text = text

        def json(self) -> dict[str, Any]:
            """Return JSON data."""
            return self._json

    def fake_post(url: str, data: str | None = None, **_: Any) -> _FakeResp:
        case_data: dict[str, Any] | None = current_case.get("data")

        # Basic payload sanity check for the success-like cases
        if isinstance(case_data, dict) and "fake_resp" in case_data:
            assert url == in_url
            assert data is not None
            payload: dict[str, Any] = json.loads(data)
            assert payload["app_id"] == app_id
            assert payload["app_version"] == app_version
            assert "flwr_version" in payload

        # Simulate network error if requested
        if isinstance(case_data, dict) and "fake_exc" in case_data:
            raise case_data["fake_exc"]

        fr: dict[str, Any] = case_data["fake_resp"]  # type: ignore[index]
        return _FakeResp(
            ok=fr["ok"],
            status=fr["status"],
            json_data=fr.get("json"),
            text=fr.get("text", ""),
        )

    monkeypatch.setattr(requests, "post", fake_post)

    for case in scenarios:
        current_case["data"] = case

        if "raises" in case:
            with pytest.raises(ValueError) as exc:
                _ = request_download_link(app_id, app_version, in_url, out_url)
            msg: str = str(exc.value)
            assert case["raises"] in msg, f"Expected '{case['raises']}' in '{msg}'"
            if case["name"] == "http_404_not_found":
                assert app_id in msg
        else:
            # Expect a (fab_url, verifications) tuple
            result: tuple[str, list[dict[str, str]] | None] = request_download_link(
                app_id, app_version, in_url, out_url
            )
            assert case["assert"](result), f"Assertion failed for {case['name']}"


class _Response:
    def __init__(self, ok: bool, body: dict[str, Any]) -> None:
        self.ok = ok
        self._body = body

    def json(self) -> dict[str, Any]:
        """Return the mocked JSON response body."""
        return self._body


def test_get_flwr_update_check_payload(monkeypatch) -> None:
    """The payload should include the expected runtime metadata."""
    monkeypatch.setattr(utils.platform, "python_version", lambda: "3.11.11")
    monkeypatch.setattr(utils.platform, "system", lambda: "Linux")
    monkeypatch.setattr(utils.platform, "release", lambda: "6.8.0")
    monkeypatch.setattr(utils, "flwr_version", "1.28.0")

    payload = get_flwr_update_check_payload(process_name="flower-superlink")

    assert payload == {
        "flwr_version": "1.28.0",
        "python_version": "3.11.11",
        "os": "linux",
        "os_version": "6.8.0",
        "process_name": "flower-superlink",
    }


def test_warn_if_flwr_update_available_disabled(monkeypatch) -> None:
    """The update check should not run when disabled via environment variable."""
    monkeypatch.setenv(FLWR_DISABLE_UPDATE_CHECK, "1")
    called = False

    def _post(*_args: Any, **_kwargs: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(utils.requests, "post", _post)

    warn_if_flwr_update_available(process_name="flower-superlink")

    assert called is False


def test_warn_if_flwr_update_available_prints_message(monkeypatch, capsys) -> None:
    """The update message should be printed to stderr when outdated."""
    response = _Response(
        ok=True,
        body={
            "update_available": True,
            "message": "A newer Flower version is available: 1.0.0 -> 1.1.0",
        },
    )

    def _post(*_args: Any, **_kwargs: Any) -> _Response:
        return response

    monkeypatch.delenv(FLWR_DISABLE_UPDATE_CHECK, raising=False)
    monkeypatch.setattr(utils.requests, "post", _post)

    warn_if_flwr_update_available(process_name="flower-superlink")

    captured = capsys.readouterr()
    assert captured.err == "A newer Flower version is available: 1.0.0 -> 1.1.0\n"


def test_warn_if_flwr_update_available_sends_expected_request(monkeypatch) -> None:
    """The update check should call the correct endpoint with the runtime payload."""
    captured: dict[str, Any] = {}

    def _post(url: str, **kwargs: Any) -> _Response:
        captured["url"] = url
        captured["json"] = kwargs["json"]
        captured["timeout"] = kwargs["timeout"]
        return _Response(ok=True, body={"update_available": False})

    monkeypatch.setattr(utils.requests, "post", _post)
    monkeypatch.setattr(
        utils,
        "get_flwr_update_check_payload",
        lambda process_name=None: {
            "flwr_version": "1.28.0",
            "python_version": "3.11.11",
            "os": "linux",
            "os_version": "6.8.0",
            "process_name": process_name or "",
        },
    )

    warn_if_flwr_update_available(process_name="flower-superlink")

    assert captured == {
        "url": FLWR_UPDATE_CHECK_URL,
        "json": {
            "flwr_version": "1.28.0",
            "python_version": "3.11.11",
            "os": "linux",
            "os_version": "6.8.0",
            "process_name": "flower-superlink",
        },
        "timeout": (
            FLWR_UPDATE_CHECK_CONNECT_TIMEOUT_SECONDS,
            FLWR_UPDATE_CHECK_READ_TIMEOUT_SECONDS,
        ),
    }


def test_warn_if_flwr_update_available_swallows_request_errors(monkeypatch) -> None:
    """The update check should fail open when the endpoint is unreachable."""

    def _post(*args: Any, **kwargs: Any) -> _Response:
        raise requests.RequestException("offline")

    monkeypatch.setattr(utils.requests, "post", _post)

    warn_if_flwr_update_available(process_name="flower-superlink")


@parameterized.expand(  # type: ignore
    [
        (24, "24s"),  # seconds
        (90, "1m 30s"),  # min + sec
        (3723, "1h 2m"),  # hour + min
        (90000, "1d 1h"),  # day + hour
    ]
)
def test_humanize_duration(seconds, expected) -> None:
    """Test the humanize_duration function."""
    assert humanize_duration(seconds) == expected


@parameterized.expand(  # type: ignore
    [
        (800, "800 B"),  # bytes
        (2048, "2.0 KB"),  # KB < 10 → 1 decimal
        (10 * 1024, "10 KB"),  # KB >= 10 → no decimal
        (5 * 1024**2, "5.0 MB"),  # MB < 10
        (3 * 1024**3, "3.0 GB"),  # GB < 10
    ]
)
def test_humanize_bytes(num_bytes, expected) -> None:
    """Test the humanize_bytes function."""
    assert humanize_bytes(num_bytes) == expected
