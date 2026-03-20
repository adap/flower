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
"""Utility functions for the infrastructure."""


import json
import os
import platform
import re
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests

from flwr.common.constant import FLWR_DIR, FLWR_HOME
from flwr.supercore.date import now as utcnow
from flwr.supercore.version import package_name as flwr_package_name
from flwr.supercore.version import package_version as flwr_version

from .constant import (
    APP_ID_PATTERN,
    APP_VERSION_PATTERN,
    FLWR_DISABLE_UPDATE_CHECK,
    FLWR_UPDATE_CHECK_CACHE_DIR,
    FLWR_UPDATE_CHECK_CACHE_FILENAME,
    FLWR_UPDATE_CHECK_CONNECT_TIMEOUT_SECONDS,
    FLWR_UPDATE_CHECK_READ_TIMEOUT_SECONDS,
    FLWR_UPDATE_CHECK_SHOW_INTERVAL_SECONDS,
    FLWR_UPDATE_CHECK_URL,
)


def mask_string(value: str, head: int = 4, tail: int = 4) -> str:
    """Mask a string by preserving only the head and tail characters.

    Mask a string for safe display by preserving the head and tail characters,
    and replacing the middle with '...'. Useful for logging tokens, secrets,
    or IDs without exposing sensitive data.

    Notes
    -----
    If the string is shorter than the combined length of `head` and `tail`,
    the original string is returned unchanged.
    """
    if len(value) <= head + tail:
        return value
    return f"{value[:head]}...{value[-tail:]}"


def uint64_to_int64(unsigned: int) -> int:
    """Convert a uint64 integer to a sint64 with the same bit pattern.

    For values >= 2^63, wraps around by subtracting 2^64.
    """
    if unsigned >= (1 << 63):
        return unsigned - (1 << 64)
    return unsigned


def int64_to_uint64(signed: int) -> int:
    """Convert a sint64 integer to a uint64 with the same bit pattern.

    For negative values, wraps around by adding 2^64.
    """
    if signed < 0:
        return signed + (1 << 64)
    return signed


def get_flwr_home() -> Path:
    """Get the Flower home directory path.

    Returns FLWR_HOME environment variable if set, otherwise returns a default
    subdirectory in the user's home directory.
    """
    if flwr_home := os.getenv(FLWR_HOME):
        return Path(flwr_home)
    return Path.home() / FLWR_DIR


def parse_app_spec(app_spec: str) -> tuple[str, str | None]:
    """Parse app specification string into app ID and version.

    Parameters
    ----------
    app_spec : str
        The app specification string in the format '@account/app' or
        '@account/app==x.y.z' (digits only).

    Returns
    -------
    tuple[str, str | None]
        A tuple containing the app ID and optional version.

    Raises
    ------
    ValueError
        If the app specification format is invalid.
    """
    if "==" in app_spec:
        app_id, app_version = app_spec.split("==", 1)

        if not re.match(APP_VERSION_PATTERN, app_version):
            raise ValueError(
                "Invalid app version. Expected format: x.y.z (digits only)."
            )
    else:
        app_id = app_spec
        app_version = None

    if not re.match(APP_ID_PATTERN, app_id):
        raise ValueError(
            "Invalid remote app ID. Expected format: '@account_name/app_name'."
        )

    return app_id, app_version


def request_download_link(
    app_id: str, app_version: str | None, in_url: str, out_url: str
) -> tuple[str, list[dict[str, str]] | None]:
    """Request a download link for the given app from Flower Hub.

    Parameters
    ----------
    app_id : str
        The application identifier in the format '@account/app'.
    app_version : str | None
        The application version (e.g., '1.2.3'), or None to request the latest version.
    in_url : str
        The Platform API endpoint URL to query.
    out_url : str
        The key name in the response that contains the download URL.

    Returns
    -------
    tuple[str, list[dict[str, str]] | None]
        A tuple containing:
        - The download URL for the application.
        - A list of verification dictionaries if provided by the API, otherwise None.

    Raises
    ------
    ValueError
        If the API connection fails, the application or version is not found,
        the API returns a non-200 response, or the response format is invalid.
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    body = {
        "app_id": app_id,  # send raw string of app_id
        "app_version": app_version,
        "flwr_version": flwr_version,
    }

    try:
        resp = requests.post(in_url, headers=headers, data=json.dumps(body), timeout=20)
    except requests.RequestException as e:
        raise ValueError(f"Unable to connect to Platform API: {e}") from e

    if resp.status_code == 404:
        # Expecting a JSON body with a "detail" field
        try:
            error_message = resp.json().get("detail")
        except ValueError:
            # JSON parsing failed
            raise ValueError(f"{app_id} not found in Platform API.") from None

        if isinstance(error_message, dict):
            available_app_versions = error_message.get("available_app_versions", [])
            available_versions_str = (
                ", ".join(map(str, available_app_versions))
                if available_app_versions
                else "None"
            )
            raise ValueError(
                f"{app_id}=={app_version} not found in Platform API. "
                f"Available app versions for {app_id}: {available_versions_str}"
            )

        raise ValueError(f"{app_id} not found in Platform API.")

    if not resp.ok:
        raise ValueError(
            f"Platform API request failed with status {resp.status_code}. "
            f"Details: {resp.text}"
        )

    data = resp.json()
    if out_url not in data:
        raise ValueError("Invalid response from Platform API")

    verifications = data["verifications"] if "verifications" in data else None

    return str(data[out_url]), verifications


def get_flwr_update_check_payload(process_name: str | None = None) -> dict[str, str]:
    """Return the runtime payload sent to the update-check endpoint."""
    payload = {
        "package_name": flwr_package_name,
        "flwr_version": flwr_version,
        "python_version": platform.python_version(),
        "os": platform.system().lower(),
        "os_version": platform.release(),
    }
    if process_name:
        payload["process_name"] = process_name
    return payload


def _get_flwr_update_check_cache_path() -> Path:
    """Return the local cache file path for update-check state."""
    return (
        get_flwr_home() / FLWR_UPDATE_CHECK_CACHE_DIR / FLWR_UPDATE_CHECK_CACHE_FILENAME
    )


def _parse_flwr_update_check_timestamp(value: Any) -> datetime | None:
    """Parse a timestamp from cache state."""
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _read_flwr_update_check_cache() -> dict[str, Any] | None:
    """Read cached update-check state from disk."""
    cache_path = _get_flwr_update_check_cache_path()
    try:
        body = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    return body if isinstance(body, dict) else None


def _write_flwr_update_check_cache(cache: dict[str, Any]) -> None:
    """Atomically persist cached update-check state to disk."""
    cache_path = _get_flwr_update_check_cache_path()
    tmp_path = cache_path.parent / f".{cache_path.name}.tmp"

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(cache), encoding="utf-8")
        tmp_path.replace(cache_path)
    except OSError:
        pass


def _cache_matches_current_install(cache: dict[str, Any]) -> bool:
    """Return True if the cached state belongs to the current install."""
    cached_package_name = cache.get("package_name") or "flwr"
    cached_flwr_version = cache.get("flwr_version")

    return (
        cached_package_name == flwr_package_name and cached_flwr_version == flwr_version
    )


def _should_show_cached_flwr_update_message(cache: dict[str, Any]) -> bool:
    """Return True if the cached update message should be shown now."""
    if not _cache_matches_current_install(cache):
        return False

    if not cache.get("update_available"):
        return False

    message = cache.get("message")
    if not isinstance(message, str) or not message.strip():
        return False

    last_shown_at = _parse_flwr_update_check_timestamp(cache.get("last_shown_at"))
    if last_shown_at is None:
        return True

    return utcnow() - last_shown_at >= timedelta(
        seconds=FLWR_UPDATE_CHECK_SHOW_INTERVAL_SECONDS
    )


def _should_refresh_flwr_update_check_cache(cache: dict[str, Any] | None) -> bool:
    """Return True if the cache is missing, invalid, or not from today (UTC)."""
    if cache is None:
        return True

    last_checked_at = _parse_flwr_update_check_timestamp(cache.get("last_checked_at"))
    if last_checked_at is None:
        return True

    return last_checked_at.date() < utcnow().date()


def _mark_cached_flwr_update_message_shown(cache: dict[str, Any]) -> None:
    """Update and persist the last time the cached message was shown."""
    cache["last_shown_at"] = utcnow().isoformat()
    _write_flwr_update_check_cache(cache)


def _request_flwr_update_check(
    process_name: str | None = None,
) -> dict[str, Any] | None:
    """Perform the update-check request and return a parsed JSON object."""
    try:
        response = requests.post(
            FLWR_UPDATE_CHECK_URL,
            json=get_flwr_update_check_payload(process_name=process_name),
            timeout=(
                FLWR_UPDATE_CHECK_CONNECT_TIMEOUT_SECONDS,
                FLWR_UPDATE_CHECK_READ_TIMEOUT_SECONDS,
            ),
        )
    except requests.RequestException:
        return None

    if not response.ok:
        return None

    try:
        body: Any = response.json()
    except ValueError:
        return None

    return body if isinstance(body, dict) else None


def _refresh_flwr_update_check_cache(process_name: str | None = None) -> None:
    """Refresh cached update-check state from the platform API."""
    body = _request_flwr_update_check(process_name=process_name)
    if body is None:
        return

    previous_cache = _read_flwr_update_check_cache() or {}
    cache: dict[str, Any] = {
        "package_name": flwr_package_name,
        "flwr_version": flwr_version,
        "update_available": body.get("update_available") is True,
        "last_checked_at": utcnow().isoformat(),
    }

    if cache["update_available"]:
        message = body.get("message")
        latest_version = body.get("latest_version")
        upgrade_hint = body.get("upgrade_hint")

        if isinstance(message, str) and message.strip():
            cache["message"] = message
        if isinstance(latest_version, str) and latest_version.strip():
            cache["latest_version"] = latest_version
        if isinstance(upgrade_hint, str) and upgrade_hint.strip():
            cache["upgrade_hint"] = upgrade_hint

        if _cache_matches_current_install(previous_cache):
            last_shown_at = previous_cache.get("last_shown_at")
            if isinstance(last_shown_at, str):
                cache["last_shown_at"] = last_shown_at

    _write_flwr_update_check_cache(cache)


def _start_flwr_update_check_refresh_thread(process_name: str | None = None) -> None:
    """Refresh cached update-check state in the background."""
    thread = threading.Thread(
        target=_refresh_flwr_update_check_cache,
        args=(process_name,),
        daemon=False,
    )
    thread.start()


def warn_if_flwr_update_available(process_name: str | None = None) -> None:
    """Print the cached update message and refresh state in the background."""
    if os.getenv(FLWR_DISABLE_UPDATE_CHECK) == "1":
        return

    cache = _read_flwr_update_check_cache()
    if cache is not None and _should_show_cached_flwr_update_message(cache):
        message = cache.get("message")
        if isinstance(message, str):
            print(message, file=sys.stderr)
            _mark_cached_flwr_update_message_shown(cache)

    if _should_refresh_flwr_update_check_cache(cache):
        _start_flwr_update_check_refresh_thread(process_name)


def humanize_duration(seconds: float) -> str:
    """Convert a duration in seconds to a human-friendly string.

    Rules:
      - < 90 seconds: show seconds
      - < 1 hour: show minutes + seconds
      - < 1 day: show hours + minutes
      - >= 1 day: show days + hours
    """
    seconds = int(seconds)

    # Under 90 seconds → Seconds only
    if seconds < 90:
        return f"{seconds}s"

    # Under 1 hour → Minutes and seconds
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"

    # Under 1 day → Hours and minutes
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m"

    # 1+ days → Days and hours
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h"


def humanize_bytes(num_bytes: int) -> str:
    """Convert a number of bytes to a human-friendly string.

    Uses 1024-based units and 0-1 decimal precision.
    Rules:
      - < 1 KB: bytes
      - < 1 MB: KB
      - < 1 GB: MB
      - < 1 TB: GB
    """
    value = float(num_bytes)

    for suffix in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or suffix == "TB":
            # Bytes → no decimals
            if suffix == "B":
                return f"{int(value)} B"

            # Decide precision: 1 decimal for <10, otherwise no decimal
            if value < 10:
                formatted = f"{value:.1f}"
            else:
                formatted = f"{int(value)}"

            return f"{formatted} {suffix}"

        value /= 1024

    raise RuntimeError("Unreachable code")  # Make mypy happy


def check_federation_format(federation: str) -> None:
    """Check if the federation string is valid.

    Parameters
    ----------
    federation : str
        The federation string to check.

    Raises
    ------
    ValueError
        If the federation string is not valid. The expected
        format is '@<account-name>/<federation-name>'.
    """
    if not re.match(r"^@[a-zA-Z0-9\-_]+/[a-zA-Z0-9\-_]+$", federation):
        raise ValueError(
            f"Invalid federation format: {federation}. "
            f"Expected format: '@<account-name>/<federation-name>'."
        )
