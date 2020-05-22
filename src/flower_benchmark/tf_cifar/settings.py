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
"""Provides a variaty of benchmark settings for CIFAR."""

from typing import List

from flower_benchmark.setting import ClientSetting, ServerSetting, Setting
from flower_ops.cluster import Instance


def get_setting(name: str) -> Setting:
    """Return appropriate setting."""
    if name not in SETTINGS:
        raise Exception(
            f"Setting {name} does not exist. Valid settings are: {list(SETTINGS.keys())}"
        )
    return SETTINGS[name]


def get_instance_name(
    instance_names: List[str], num_clients: int, client_index: int
) -> str:
    """Return instance_name."""
    idx = client_index // (num_clients // len(instance_names))
    idx = min([idx, len(instance_names) - 1])
    return instance_names[min(idx, len(instance_names))]


def configure_uniform_clients(
    iid_fraction: float, instance_names: List[str], num_clients: int, dry_run: bool,
) -> List[ClientSetting]:
    """Configure `num_clients`, all using the same delay factor."""
    clients = []
    for i in range(num_clients):
        client = ClientSetting(
            # Set instance on which to run
            instance_name=get_instance_name(instance_names, num_clients, i),
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


# pylint: disable=too-many-arguments
def configure_clients(
    iid_fraction: float,
    instance_names: List[str],
    num_clients: int,
    dry_run: bool,
    delay_factor_fast: float,
    delay_factor_slow: float,
) -> List[ClientSetting]:
    """Configure `num_clients` with different delay factors."""
    clients = []
    for i in range(num_clients):
        client = ClientSetting(
            # Set instance on which to run
            instance_name=get_instance_name(instance_names, num_clients, i),
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
    "dry-run": Setting(
        instances=[
            Instance(name="server", group="server", num_cpu=2, num_ram=8),
            Instance(name="client", group="clients", num_cpu=2, num_ram=8),
        ],
        server=ServerSetting(
            instance_name="server",
            strategy="fedavg",
            rounds=1,
            min_num_clients=1,
            sample_fraction=1.0,
            min_sample_size=1,
            training_round_timeout=600,
            lr_initial=0.1,
            partial_updates=False,
            importance_sampling=False,
            dynamic_timeout=False,
            dry_run=True,
        ),
        clients=configure_uniform_clients(
            iid_fraction=0.0, instance_names=["client"], num_clients=4, dry_run=True
        ),
    ),
    "minimal": Setting(
        instances=[
            Instance(name="server", group="server", num_cpu=2, num_ram=8),
            Instance(name="client_0", group="clients", num_cpu=2, num_ram=8),
            Instance(name="client_1", group="clients", num_cpu=2, num_ram=8),
        ],
        server=ServerSetting(
            instance_name="server",
            strategy="fedavg",
            rounds=2,
            min_num_clients=4,
            sample_fraction=1.0,
            min_sample_size=3,
            training_round_timeout=3600,
            lr_initial=0.1,
            partial_updates=False,
            importance_sampling=False,
            dynamic_timeout=False,
            dry_run=False,
        ),
        clients=configure_uniform_clients(
            iid_fraction=0.0,
            instance_names=["client_0", "client_1"],
            num_clients=4,
            dry_run=False,
        ),
    ),
    "fedavg-sync": Setting(
        instances=[
            Instance(name="server", group="server", num_cpu=4, num_ram=16),
            Instance(name="client_0", group="clients", num_cpu=48, num_ram=192),
            Instance(name="client_1", group="clients", num_cpu=48, num_ram=192),
        ],
        server=ServerSetting(
            instance_name="server",
            strategy="fedavg",
            rounds=10,
            min_num_clients=80,
            sample_fraction=0.1,
            min_sample_size=10,
            training_round_timeout=None,
            lr_initial=0.1,
            partial_updates=False,
            importance_sampling=False,
            dynamic_timeout=False,
            dry_run=False,
        ),
        clients=configure_clients(
            iid_fraction=0.5,
            instance_names=["client_0", "client_1"],
            num_clients=100,
            dry_run=False,
            delay_factor_fast=0.0,
            delay_factor_slow=0.0,
        ),
    ),
    "fedavg-async": Setting(
        instances=[
            Instance(name="server", group="server", num_cpu=4, num_ram=16),
            Instance(name="client_0", group="clients", num_cpu=48, num_ram=192),
            Instance(name="client_1", group="clients", num_cpu=48, num_ram=192),
        ],
        server=ServerSetting(
            instance_name="server",
            strategy="fedavg",
            rounds=10,
            min_num_clients=80,
            sample_fraction=0.1,
            min_sample_size=10,
            training_round_timeout=60,
            lr_initial=0.1,
            partial_updates=False,
            importance_sampling=False,
            dynamic_timeout=False,
            dry_run=False,
        ),
        clients=configure_clients(
            iid_fraction=0.5,
            instance_names=["client_0", "client_1"],
            num_clients=100,
            dry_run=False,
            delay_factor_fast=0.0,
            delay_factor_slow=0.0,
        ),
    ),
}
