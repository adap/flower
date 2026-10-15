# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
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
"""Tests for shared OAuth token handling."""

from unittest.mock import Mock, patch

from .notion.definition import PROVIDER
from .oauth import OAuthFlow


def test_exchange_records_absolute_expiry() -> None:
    """Record expiry on initial authorization so the first task can reuse its token."""
    flow = OAuthFlow(
        PROVIDER,
        client_id="client",
        client_secret="secret",
        redirect_uri="https://example.com/callback",
    )
    response = Mock(status_code=200)
    response.json.return_value = {
        "access_token": "token",
        "refresh_token": "refresh",
        "expires_in": 3600,
    }
    with (
        patch("flwr.supercore.task_process.connector.oauth.time", return_value=1000),
        patch(
            "flwr.supercore.task_process.connector.oauth.requests.post",
            return_value=response,
        ),
    ):
        credentials, _ = flow.exchange_code(
            code="code", redirect_uri="https://example.com/callback", pkce_verifier=None
        )

    assert credentials == {
        "access_token": "token",
        "refresh_token": "refresh",
        "expires_in": 3600,
        "expires_at": 4600,
    }
