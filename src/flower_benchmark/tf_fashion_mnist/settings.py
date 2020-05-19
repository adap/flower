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
"""Provides a variaty of benchmark settings."""

from dataclasses import dataclass
from typing import List


@dataclass
class ServerSetting:
    """Settings for the server."""

    rounds: int = 100
    min_num_clients: int = 100
    sample_fraction: float = 1.0
    min_sample_size: int = 100
    training_round_timeout: int = 3600
    lr_initial: float = 0.1
    dry_run: bool = False


@dataclass
class ClientSetting:
    """Settings for the client."""

    num_clients: int = 100
    dry_run: bool = False


@dataclass
class Setting:
    """One specific training setting."""

    server: ServerSetting
    clients: List[ClientSetting]


def get_setting(name: str) -> Setting:
    """Return appropriate setting."""
    if name not in SETTINGS:
        raise Exception(
            "Setting does not exist. Valid settings are: %s" % list(SETTINGS.keys())
        )

    return SETTINGS[name]


SETTINGS = {
    "dry": Setting(
        server=ServerSetting(
            rounds=1,
            min_num_clients=1,
            sample_fraction=1.0,
            min_sample_size=1,
            training_round_timeout=600,
            dry_run=True,
        ),
        clients=[ClientSetting(num_clients=4, dry_run=True)],
    ),
    "minimal": Setting(
        server=ServerSetting(
            rounds=2,
            min_num_clients=4,
            sample_fraction=1.0,
            min_sample_size=3,
            training_round_timeout=3600,
        ),
        clients=[ClientSetting(num_clients=4)],
    ),
}
