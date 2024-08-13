# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Backend config tests."""


import contextlib
from typing import Optional

import pytest

from .backendconfig import BackendConfig, ClientAppResources


@pytest.mark.parametrize(
    "num_cpus, num_gpus, warn",
    [
        (1, 0.0, False),  # pass
        (1, 0.5, False),  # pass
        (4, 1.24, False),  # pass
        (2.0, 0.5, True),  # pass, but throws warning
    ],
)
def test_correct_clientappresources(num_cpus: int, num_gpus: float, warn: bool) -> None:
    """Test if settings for ClientAppResources are valid."""
    ctx: contextlib.AbstractContextManager[object]
    if warn:
        ctx = pytest.warns(UserWarning)
    else:
        ctx = contextlib.nullcontext()

    with ctx:
        _ = ClientAppResources(num_cpus=num_cpus, num_gpus=num_gpus)


@pytest.mark.parametrize(
    "num_cpus, num_gpus",
    [
        (0, 0.0),  # fail, CPUs must be > 1
        (1, -1.0),  # fail, GPUs must be >= 0
        (None, 0.0),  # fail, num_cpus can't be None
        (1, None),  # fail, num_gpus can't be None
    ],
)
def test_incorrect_clientappresources(num_cpus: int, num_gpus: float) -> None:
    """Test if settings for ClientAppResources are flagged as invalid."""
    with pytest.raises(ValueError):
        _ = ClientAppResources(num_cpus=num_cpus, num_gpus=num_gpus)


@pytest.mark.parametrize(
    "backend_name, raise_valueerror",
    [
        (None, False),  # default backend
        ("ray", False),  # pass
    ],
)
def test_backendconfig_creation(
    backend_name: Optional[str], raise_valueerror: bool
) -> None:
    """Test backendconfig creation with default and supported backends."""
    ctx: contextlib.AbstractContextManager[object]
    ctx = pytest.raises(ValueError) if raise_valueerror else contextlib.nullcontext()
    with ctx:
        _ = (
            BackendConfig()
            if backend_name is None
            else BackendConfig(name=backend_name)
        )
