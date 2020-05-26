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


from typing import List, Optional, Tuple

import numpy as np

from flower_ops.cluster import Instance


def sample_delay_factors(
    num_clients: int, max_delay: float, seed: Optional[int]
) -> List[float]:
    """Sample delay factors."""
    np.random.seed(seed)
    # pylint: disable-msg=invalid-name
    ps = [float(p) for p in np.random.rand(num_clients)]
    step_size = max_delay / num_clients
    ds = [(i + 1) * step_size for i in range(num_clients)]
    return [p * d for p, d in zip(ps, ds)]


def configure_client_instances(
    num_clients: int, num_cpu: int, num_ram: float
) -> Tuple[List[Instance], List[str]]:
    """Return list of client instances and a list of instance names."""
    instance_names = [f"client_{i}" for i in range(num_clients)]

    instances = [
        Instance(name=instance_name, group="clients", num_cpu=num_cpu, num_ram=num_ram)
        for instance_name in instance_names
    ]

    return instances, instance_names
