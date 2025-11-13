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
"""Test for Flower command line interface `app review` command."""


from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

import pytest
import requests
import typer
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from .review import _download_fab, _load_private_key, _sign_fab, _submit_review


def test_download_fab_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test _download_fab() returns bytes on success and exits on network failure."""

    # --- success ---
    class FakeResp:
        """A minimal stand-in for `requests.Response` used in tests."""
        def __init__(self, content: bytes):
            self._content = content
            self.status_code = 200
            self.ok = True

        def raise_for_status(self) -> None:
            """Mimic a successful HTTP 200 response."""
            return None

        @property
        def content(self) -> bytes:
            """Return the stored byte content for the fake response."""
            return self._content

    payload = b"FABDATA"
    monkeypatch.setattr(
        requests, "get", lambda url, timeout=60: FakeResp(payload), raising=True
    )
    out = _download_fab("https://example.ai/fab")
    assert out == payload

    # --- failure (network error) ---
    class Boom(requests.RequestException):
        """Custom RequestException subclass used to simulate network errors in tests."""
        pass

    def boom() -> None:
        """Raise a Boom(RequestException) to simulate a network failure."""
        raise Boom("net down")

    monkeypatch.setattr(requests, "get", boom, raising=True)
    with pytest.raises(typer.Exit) as exc:
        _download_fab("https://example.ai/fab")
    assert exc.value.exit_code == 1


def test_load_private_key_valid_and_invalid(tmp_path: Path) -> None:
    """Test _load_private_key() correctly loads a valid Ed25519 key and exits on invalid
    data."""
    # Generate a valid ed25519 private key in OpenSSH format
    key = ed25519.Ed25519PrivateKey.generate()
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.OpenSSH,
        encryption_algorithm=serialization.NoEncryption(),
    )
    key_path = tmp_path / "id_ed25519"
    key_path.write_bytes(pem)

    loaded = _load_private_key(key_path)
    assert isinstance(loaded, ed25519.Ed25519PrivateKey)

    # Invalid content should raise Exit
    bad_path = tmp_path / "bad_key"
    bad_path.write_bytes(b"not a key")
    with pytest.raises(typer.Exit) as exc:
        _load_private_key(bad_path)
    assert exc.value.exit_code == 1


def test_sign_fab_returns_signature_and_timestamp() -> None:
    """Test _sign_fab() returns valid signature bytes and a current timestamp."""
    key = ed25519.Ed25519PrivateKey.generate()
    before = int(time.time())
    sig, ts = _sign_fab(b"hello fab", key)
    after = int(time.time())
    assert isinstance(sig, (bytes, bytearray))
    assert before <= ts <= after


def test_submit_review_success_and_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test _submit_review() on success, HTTP error, and network exception."""
    captured: dict[str, Any] = {}

    class FakeResp:
        """Lightweight mock of `requests.Response` used for testing `_submit_review`."""
        def __init__(self, ok: bool, status: int = 200, text: str = ""):
            self.ok = ok
            self.status_code = status
            self.text = text

    # --- success ---
    def fake_post(url: str, headers: dict[str, str], json: dict[str, Any]) -> FakeResp:
        """Simulate a successful POST request that captures its arguments."""
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return FakeResp(True)

    monkeypatch.setattr(requests, "post", fake_post, raising=True)
    _submit_review("@u/a", "1.2.3", b"sigbytes", 123, "TKN")
    assert captured["url"].endswith("/hub/apps/signature")
    assert captured["headers"]["Authorization"] == "Bearer TKN"
    assert captured["json"]["app_id"] == "@u/a"
    assert captured["json"]["version"] == "1.2.3"
    assert captured["json"]["signature_b64"] == base64.urlsafe_b64encode(
        b"sigbytes"
    ).rstrip(b"=").decode("ascii")
    assert captured["json"]["signed_at"] == 123

    # --- non-OK HTTP response ---
    def fake_post_fail() -> FakeResp:
        """Simulate a non-OK HTTP response from the review submission endpoint."""
        return FakeResp(False, 503, "oops")

    monkeypatch.setattr(requests, "post", fake_post_fail, raising=True)
    with pytest.raises(typer.Exit) as exc:
        _submit_review("@u/a", "1.2.3", b"sigbytes", 123, "TKN")
    assert exc.value.exit_code == 1

    # --- network exception ---
    def fake_post_raise(*_a: Any, **_k: Any) -> None:
        """Raise a RequestException (simulated network error)."""
        raise requests.RequestException("boom")

    monkeypatch.setattr(requests, "post", fake_post_raise, raising=True)
    with pytest.raises(typer.Exit) as exc:
        _submit_review("@u/a", "1.2.3", b"sigbytes", 123, "TKN")
    assert exc.value.exit_code == 1
