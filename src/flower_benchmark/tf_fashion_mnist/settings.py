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
"""Provides a variaty of benchmark settings for Fashion-MNIST."""

from typing import List

from flower_benchmark.setting import ClientSetting, ServerSetting, Setting


def get_setting(name: str) -> Setting:
    """Return appropriate setting."""
    if name not in SETTINGS:
        raise Exception(
            f"Setting {name} does not exist. Valid settings are: {list(SETTINGS.keys())}"
        )
    return SETTINGS[name]


def configure_uniform_clients(
    iid_fraction: float, num_clients: int, dry_run: bool,
) -> List[ClientSetting]:
    """Configure `num_clients`, all using the same delay factor."""
    clients = []
    for i in range(num_clients):
        client = ClientSetting(
            # Individual
            cid=str(i),
            partition=i,
            delay_factor=0.0,
            # Shared
            iid_fraction=iid_fraction,
            num_clients=num_clients,
            dry_run=dry_run,
        )
        clients.append(client)

    return clients


def configure_clients(
    iid_fraction: float,
    num_clients: int,
    dry_run: bool,
    delay_factor_fast: float,
    delay_factor_slow: float,
) -> List[ClientSetting]:
    """Configure `num_clients` with different delay factors."""
    clients = []
    for i in range(num_clients):
        client = ClientSetting(
            # Individual
            cid=str(i),
            partition=i,
            # Indices 0 to 49 fast, 50 to 99 slow
            delay_factor=delay_factor_fast
            if i < int(num_clients / 2)
            else delay_factor_slow,
            # Shared
            iid_fraction=iid_fraction,
            num_clients=num_clients,
            dry_run=dry_run,
        )
        clients.append(client)

    return clients


SETTINGS = {
    "dry": Setting(
        server=ServerSetting(
            strategy="fedavg",
            rounds=1,
            min_num_clients=1,
            sample_fraction=1.0,
            min_sample_size=1,
            training_round_timeout=600,
            lr_initial=0.1,
            partial_updates=False,
            dry_run=True,
        ),
        clients=configure_uniform_clients(
            iid_fraction=0.0, num_clients=4, dry_run=True
        ),
    ),
    "minimal": Setting(
        server=ServerSetting(
            strategy="fedavg",
            rounds=2,
            min_num_clients=4,
            sample_fraction=1.0,
            min_sample_size=3,
            training_round_timeout=3600,
            lr_initial=0.1,
            partial_updates=False,
            dry_run=False,
        ),
        clients=configure_uniform_clients(
            iid_fraction=0.0, num_clients=4, dry_run=False
        ),
    ),
    "fedavg-sync": Setting(
        server=ServerSetting(
            strategy="fedavg",
            rounds=25,
            min_num_clients=80,
            sample_fraction=0.1,
            min_sample_size=10,
            training_round_timeout=None,
            lr_initial=0.1,
            partial_updates=False,
            dry_run=False,
        ),
        clients=configure_clients(
            iid_fraction=0.0,
            num_clients=100,
            dry_run=False,
            delay_factor_fast=0.0,
            delay_factor_slow=3.0,
        ),
    ),
    "fedavg-async": Setting(
        server=ServerSetting(
            strategy="fedavg",
            rounds=25,
            min_num_clients=80,
            sample_fraction=0.1,
            min_sample_size=10,
            training_round_timeout=20,
            lr_initial=0.1,
            partial_updates=False,
            dry_run=False,
        ),
        clients=configure_clients(
            iid_fraction=0.0,
            num_clients=100,
            dry_run=False,
            delay_factor_fast=0.0,
            delay_factor_slow=3.0,
        ),
    ),
    "fast-and-slow-only-partial-updates": Setting(
        server=ServerSetting(
            strategy="fast-and-slow",
            rounds=25,
            min_num_clients=80,
            sample_fraction=0.1,
            min_sample_size=10,
            training_round_timeout=20,
            lr_initial=0.1,
            partial_updates=True,
            dry_run=False,
        ),
        clients=configure_clients(
            iid_fraction=0.0,
            num_clients=100,
            dry_run=False,
            delay_factor_fast=0.0,
            delay_factor_slow=3.0,
        ),
    ),
    # "fast-and-slow-only-dynamic-timeouts": Setting(
    #     server=ServerSetting(
    #         strategy="fast-and-slow",
    #         rounds=25,
    #         min_num_clients=80,
    #         sample_fraction=0.1,
    #         min_sample_size=10,
    #         training_round_timeout=20,
    #         lr_initial=0.1,
    #         partial_updates=True,
    #         dry_run=False,
    #     ),
    #     clients=configure_clients(
    #         iid_fraction=0.0,
    #         num_clients=100,
    #         dry_run=False,
    #         delay_factor_fast=0.0,
    #         delay_factor_slow=3.0,
    #     ),
    # ),
    # "fast-and-slow-only-importance-sampling": Setting(
    #     server=ServerSetting(
    #         strategy="fast-and-slow",
    #         rounds=25,
    #         min_num_clients=80,
    #         sample_fraction=0.1,
    #         min_sample_size=10,
    #         training_round_timeout=20,
    #         lr_initial=0.1,
    #         partial_updates=True,
    #         dry_run=False,
    #     ),
    #     clients=configure_clients(
    #         iid_fraction=0.0,
    #         num_clients=100,
    #         dry_run=False,
    #         delay_factor_fast=0.0,
    #         delay_factor_slow=3.0,
    #     ),
    # ),
    "fast-and-slow": Setting(
        server=ServerSetting(
            strategy="fast-and-slow",
            rounds=25,
            min_num_clients=80,
            sample_fraction=0.1,
            min_sample_size=10,
            training_round_timeout=20,
            lr_initial=0.1,
            partial_updates=True,
            dry_run=False,
        ),
        clients=configure_clients(
            iid_fraction=0.0,
            num_clients=100,
            dry_run=False,
            delay_factor_fast=0.0,
            delay_factor_slow=3.0,
        ),
    ),
}
