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
"""Baseline settings."""


from typing import List

from flower_benchmark.common import configure_client_instances
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


client_instances_100, client_names_100 = configure_client_instances(
    num_clients=100, num_cpu=2, num_ram=4
)

client_instances_10, client_names_10 = configure_client_instances(
    num_clients=10, num_cpu=2, num_ram=4
)

SETTINGS = {
    "cifar-dry-run": Setting(
        instances=[
            Instance(name="server", group="server", num_cpu=2, num_ram=8),
            Instance(name="client", group="clients", num_cpu=2, num_ram=4),
        ],
        server=ServerSetting(
            instance_name="server",
            strategy="fedavg",
            rounds=1,
            min_num_clients=2,
            sample_fraction=1.0,
            min_sample_size=2,
            training_round_timeout=600,
            lr_initial=0.001,
            partial_updates=False,
            importance_sampling=False,
            dynamic_timeout=False,
            dry_run=True,
        ),
        clients=configure_uniform_clients(
            iid_fraction=1.0, instance_names=["client"], num_clients=2, dry_run=True,
        ),
    ),
    "cifar-fedavg-10-10": Setting(
        instances=[Instance(name="server", group="server", num_cpu=2, num_ram=8)]
        + client_instances_10,
        server=ServerSetting(
            instance_name="server",
            strategy="fedavg",
            rounds=5,
            min_num_clients=10,
            sample_fraction=1.0,
            min_sample_size=10,
            training_round_timeout=3600,
            lr_initial=0.001,
            partial_updates=False,
            importance_sampling=False,
            dynamic_timeout=False,
            dry_run=False,
        ),
        clients=configure_uniform_clients(
            iid_fraction=1.0,
            instance_names=client_names_10,
            num_clients=10,
            dry_run=False,
        ),
    ),
    "cifar-fedavg-100-10": Setting(
        instances=[Instance(name="server", group="server", num_cpu=2, num_ram=8)]
        + client_instances_100,
        server=ServerSetting(
            instance_name="server",
            strategy="fedavg",
            rounds=5,
            min_num_clients=100,
            sample_fraction=0.1,
            min_sample_size=10,
            training_round_timeout=3600,
            lr_initial=0.001,
            partial_updates=False,
            importance_sampling=False,
            dynamic_timeout=False,
            dry_run=False,
        ),
        clients=configure_uniform_clients(
            iid_fraction=1.0,
            instance_names=client_names_100,
            num_clients=100,
            dry_run=False,
        ),
    ),
}
