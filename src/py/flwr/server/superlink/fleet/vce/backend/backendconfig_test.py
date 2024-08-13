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


import pytest
from contextlib import nullcontext
from typing import Optional, Union
from .backendconfig import ClientAppResources, BackendConfig

@pytest.mark.parametrize(
    "num_cpus, num_gpus, to_catch",
    [
        (1, 0.0, None), # pass
        (2.0, 0.5, UserWarning), # pass, but throws warning
        (0, 0.0, ValueError), # fail, CPUs must be > 1
        (1, -1.0, ValueError), # fail, GPUs must be >= 0
        (None, 0.0, ValueError), # fail, num_cpus can't be None
        (1, None, ValueError), # fail, num_gpus can't be None
    ],
)
def test_clientappresources_validity(
    num_cpus: int, num_gpus: float, to_catch: Optional[Union[Exception, UserWarning]]
) -> None:
    """Test if settings for ClientAppResources are valid"""
    
    if to_catch:
        ctx = pytest.raises(to_catch) if to_catch is ValueError else pytest.warns(UserWarning)
    else:
        ctx = nullcontext()

    with ctx:
        _ = ClientAppResources(num_cpus=num_cpus, num_gpus=num_gpus)
