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


@dataclass
class Setting:
    """One specific training setting."""

    rounds: int
    num_clients: int
    sample_fraction: float
    min_sample_size: int
    min_num_clients: int
    training_round_timeout: int


def get_setting(name: str) -> Setting:
    """Return appropriate setting."""
    if name not in SETTINGS:
        raise Exception(
            "Setting does not exist. Valid settings are: %s" % list(SETTINGS.keys())
        )

    return SETTINGS[name]


SETTINGS = {
    "fixed_delay_variable_timeout_60": Setting(
        rounds=5,
        num_clients=10,
        sample_fraction=1.0,
        min_sample_size=10,
        min_num_clients=10,
        training_round_timeout=60,
    )
}
