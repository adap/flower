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
"""Flower date utils."""


import datetime


def now() -> datetime.datetime:
    """Construct a datetime from time.time() with time zone set to UTC."""
    return datetime.datetime.now(tz=datetime.timezone.utc)


def format_timedelta(td: datetime.timedelta) -> str:
    """Format a timedelta as a string."""
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if days > 0:
        return f"{days}d {hours:02}:{minutes:02}:{seconds:02}"
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def isoformat8601_utc(dt: datetime.datetime) -> str:
    """Return the datetime formatted as an ISO 8601 string with a trailing 'Z'."""
    if dt.tzinfo != datetime.timezone.utc:
        raise ValueError("Expected datetime with timezone set to UTC")
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")
