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
from abc import ABC, abstractmethod
from typing import Dict


class SecAggStrategy(ABC):
    @abstractmethod
    def get_sec_agg_param(self) -> Dict[str, int]:
        '''Produce a dictionary storing parameters for the secure aggregation protocol
        min_num: Minimum number of clients to be available at the end of protocol
        min_frac: Minimum fraction of clients available with respect to sampled number at the end of protocol
        share_num: Number of shares to be generated for secret
        threshold: Number of shares needed to reconstruct secret
        clipping_range: Range of weight vector initially
        target_range: Range of weight vector after quantization
        mod_range: Field of cryptographic primitives
        max_weights_factor: maximum weights factor mulitplied on weights vector
        timeout: not used, but timeout for gRPC in the future

        Note: do not use secagg_id or sample_num, as it will be used overwritten on server side'''
