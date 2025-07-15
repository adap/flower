# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Constants for differential privacy."""


KEY_CLIPPING_NORM = "clipping_norm"
KEY_NORM_BIT = "norm_bit"
CLIENTS_DISCREPANCY_WARNING = (
    "The number of clients returning parameters (%s)"
    " differs from the number of sampled clients (%s)."
    " This could impact the differential privacy guarantees,"
    " potentially leading to privacy leakage or inadequate noise calibration."
)
