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
"""Provides a variaty of benchmark settings for Spoken Keyword classification."""


from typing import List

from flower_benchmark.common import (
    configure_client_instances,
    sample_delay_factors,
    sample_real_delay_factors,
)
from flower_benchmark.setting import ClientSetting, ServerSetting, Setting
from flower_ops.cluster import Instance

ROUNDS = 20
MIN_NUM_CLIENTS = 80
SAMPLE_FRACTION = 0.5
MIN_SAMPLE_SIZE = 50

LR_INITIAL = 0.01

IID_FRACTION = 0.1
MAX_DELAY_FACTOR = 4.0


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


client_instances_100, client_names_100 = configure_client_instances(
    num_clients=100, num_cpu=2, num_ram=8
)

client_instances_10, client_names_10 = configure_client_instances(
    num_clients=10, num_cpu=2, num_ram=8
)

# pylint: disable=too-many-arguments
def configure_clients(
    iid_fraction: float,
    instance_names: List[str],
    num_clients: int,
    dry_run: bool,
    delay_factor_fast: float,
    delay_factor_slow: float,
    sample_delays: bool = True,
    real_delays: bool = False,
) -> List[ClientSetting]:
    """Configure `num_clients` with different delay factors."""
    if sample_delays:
        # Configure clients with sampled delay factors
        if real_delays:
            delay_factors = sample_real_delay_factors(
                num_clients=num_clients, seed=2020
            )
        else:
            delay_factors = sample_delay_factors(
                num_clients=num_clients, max_delay=delay_factor_slow, seed=2020
            )
        return [
            ClientSetting(
                # Set instance on which to run
                instance_name=get_instance_name(instance_names, num_clients, i),
                # Individual
                cid=str(i),
                partition=i,
                delay_factor=delay_factors[i],
                # Shared
                iid_fraction=iid_fraction,
                num_clients=num_clients,
                dry_run=dry_run,
            )
            for i in range(num_clients)
        ]
    # Configure clients with fixed delay factors
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
            Instance(name="server", group="server", num_cpu=4, num_ram=16),
            Instance(name="client", group="clients", num_cpu=4, num_ram=16),
        ],
        server=ServerSetting(
            instance_name="server",
            strategy="fedavg",
            rounds=1,
            min_num_clients=1,
            sample_fraction=1.0,
            min_sample_size=1,
            training_round_timeout=600,
            lr_initial=LR_INITIAL,
            partial_updates=False,
            importance_sampling=False,
            dynamic_timeout=False,
            dry_run=True,
        ),
        clients=configure_uniform_clients(
            iid_fraction=IID_FRACTION,
            instance_names=["client"],
            num_clients=4,
            dry_run=True,
        ),
    ),
    "minimal": Setting(
        instances=[Instance(name="server", group="server", num_cpu=4, num_ram=16)]
        + client_instances_10,
        server=ServerSetting(
            instance_name="server",
            strategy="fedavg",
            rounds=2,
            min_num_clients=4,
            sample_fraction=1.0,
            min_sample_size=3,
            training_round_timeout=3600,
            lr_initial=LR_INITIAL,
            partial_updates=False,
            importance_sampling=False,
            dynamic_timeout=False,
            dry_run=False,
        ),
        clients=configure_clients(
            iid_fraction=IID_FRACTION,
            instance_names=client_names_10,
            num_clients=10,
            dry_run=False,
            delay_factor_fast=0.0,
            delay_factor_slow=MAX_DELAY_FACTOR,
            real_delays=True,
        ),
    ),
    "fedavg-sync": Setting(
        instances=[Instance(name="server", group="server", num_cpu=8, num_ram=32)]
        + client_instances_100,
        server=ServerSetting(
            instance_name="server",
            strategy="fedavg",
            rounds=ROUNDS,
            min_num_clients=MIN_NUM_CLIENTS,
            sample_fraction=SAMPLE_FRACTION,
            min_sample_size=MIN_SAMPLE_SIZE,
            training_round_timeout=None,
            lr_initial=LR_INITIAL,
            partial_updates=False,
            importance_sampling=False,
            dynamic_timeout=False,
            dry_run=False,
        ),
        clients=configure_clients(
            iid_fraction=IID_FRACTION,
            instance_names=client_names_100,
            num_clients=100,
            dry_run=False,
            delay_factor_fast=0.0,
            delay_factor_slow=MAX_DELAY_FACTOR,
            real_delays=True,
        ),
    ),
    "fedavg-async": Setting(
        instances=[Instance(name="server", group="server", num_cpu=8, num_ram=32)]
        + client_instances_100,
        server=ServerSetting(
            instance_name="server",
            strategy="fedavg",
            rounds=ROUNDS,
            min_num_clients=MIN_NUM_CLIENTS,
            sample_fraction=SAMPLE_FRACTION,
            min_sample_size=MIN_SAMPLE_SIZE,
            training_round_timeout=20,
            lr_initial=LR_INITIAL,
            partial_updates=False,
            importance_sampling=False,
            dynamic_timeout=False,
            dry_run=False,
        ),
        clients=configure_clients(
            iid_fraction=IID_FRACTION,
            instance_names=client_names_100,
            num_clients=100,
            dry_run=False,
            delay_factor_fast=0.0,
            delay_factor_slow=MAX_DELAY_FACTOR,
            real_delays=True,
        ),
    ),
    "fast-and-slow-only-partial-updates": Setting(
        instances=[Instance(name="server", group="server", num_cpu=8, num_ram=32)]
        + client_instances_100,
        server=ServerSetting(
            instance_name="server",
            strategy="fast-and-slow",
            rounds=ROUNDS,
            min_num_clients=MIN_NUM_CLIENTS,
            sample_fraction=SAMPLE_FRACTION,
            min_sample_size=MIN_SAMPLE_SIZE,
            training_round_timeout=20,
            lr_initial=LR_INITIAL,
            partial_updates=True,
            importance_sampling=False,
            dynamic_timeout=False,
            dry_run=False,
        ),
        clients=configure_clients(
            iid_fraction=IID_FRACTION,
            instance_names=client_names_100,
            num_clients=100,
            dry_run=False,
            delay_factor_fast=0.0,
            delay_factor_slow=MAX_DELAY_FACTOR,
            real_delays=True,
        ),
    ),
    "fast-and-slow-only-dynamic-timeouts": Setting(
        instances=[Instance(name="server", group="server", num_cpu=8, num_ram=32)]
        + client_instances_100,
        server=ServerSetting(
            instance_name="server",
            strategy="fast-and-slow",
            rounds=ROUNDS,
            min_num_clients=MIN_NUM_CLIENTS,
            sample_fraction=SAMPLE_FRACTION,
            min_sample_size=MIN_SAMPLE_SIZE,
            training_round_timeout=20,
            lr_initial=LR_INITIAL,
            partial_updates=False,
            importance_sampling=False,
            dynamic_timeout=True,
            dry_run=False,
        ),
        clients=configure_clients(
            iid_fraction=IID_FRACTION,
            instance_names=client_names_100,
            num_clients=100,
            dry_run=False,
            delay_factor_fast=0.0,
            delay_factor_slow=MAX_DELAY_FACTOR,
            real_delays=True,
        ),
    ),
    "fast-and-slow-only-importance-sampling": Setting(
        instances=[Instance(name="server", group="server", num_cpu=8, num_ram=32)]
        + client_instances_100,
        server=ServerSetting(
            instance_name="server",
            strategy="fast-and-slow",
            rounds=ROUNDS,
            min_num_clients=MIN_NUM_CLIENTS,
            sample_fraction=SAMPLE_FRACTION,
            min_sample_size=MIN_SAMPLE_SIZE,
            training_round_timeout=20,
            lr_initial=LR_INITIAL,
            partial_updates=False,
            importance_sampling=True,
            dynamic_timeout=False,
            dry_run=False,
        ),
        clients=configure_clients(
            iid_fraction=IID_FRACTION,
            instance_names=client_names_100,
            num_clients=100,
            dry_run=False,
            delay_factor_fast=0.0,
            delay_factor_slow=MAX_DELAY_FACTOR,
            real_delays=True,
        ),
    ),
    "fast-and-slow": Setting(
        instances=[Instance(name="server", group="server", num_cpu=8, num_ram=32)]
        + client_instances_100,
        server=ServerSetting(
            instance_name="server",
            strategy="fast-and-slow",
            rounds=ROUNDS,
            min_num_clients=MIN_NUM_CLIENTS,
            sample_fraction=SAMPLE_FRACTION,
            min_sample_size=MIN_SAMPLE_SIZE,
            training_round_timeout=60,
            lr_initial=LR_INITIAL,
            partial_updates=True,
            importance_sampling=True,
            dynamic_timeout=True,
            dry_run=False,
        ),
        clients=configure_clients(
            iid_fraction=IID_FRACTION,
            instance_names=client_names_100,
            num_clients=100,
            dry_run=False,
            delay_factor_fast=0.0,
            delay_factor_slow=MAX_DELAY_FACTOR,
            real_delays=True,
        ),
    ),
    "qffedavg": Setting(
        instances=[Instance(name="server", group="server", num_cpu=8, num_ram=32)]
        + client_instances_100,
        server=ServerSetting(
            instance_name="server",
            strategy="qffedavg",
            rounds=ROUNDS,
            min_num_clients=MIN_NUM_CLIENTS,
            sample_fraction=SAMPLE_FRACTION,
            min_sample_size=MIN_SAMPLE_SIZE,
            training_round_timeout=None,
            lr_initial=LR_INITIAL,
            partial_updates=False,
            importance_sampling=False,
            dynamic_timeout=False,
            dry_run=False,
        ),
        clients=configure_clients(
            iid_fraction=IID_FRACTION,
            instance_names=client_names_100,
            num_clients=100,
            dry_run=False,
            delay_factor_fast=0.0,
            delay_factor_slow=MAX_DELAY_FACTOR,
            real_delays=True,
        ),
    ),
}
