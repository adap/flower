# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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
"""Flower constants."""


from __future__ import annotations

MISSING_EXTRA_REST = """
Extra dependencies required for using the REST-based Fleet API are missing.

To use the REST API, install `flwr` with the `rest` extra:

    `pip install flwr[rest]`.
"""

TRANSPORT_TYPE_GRPC_BIDI = "grpc-bidi"
TRANSPORT_TYPE_GRPC_RERE = "grpc-rere"
TRANSPORT_TYPE_REST = "rest"
TRANSPORT_TYPE_VCE = "vce"
TRANSPORT_TYPES = [
    TRANSPORT_TYPE_GRPC_BIDI,
    TRANSPORT_TYPE_GRPC_RERE,
    TRANSPORT_TYPE_REST,
    TRANSPORT_TYPE_VCE,
]

MESSAGE_TYPE_GET_PROPERTIES = "get_properties"
MESSAGE_TYPE_GET_PARAMETERS = "get_parameters"
MESSAGE_TYPE_FIT = "fit"
MESSAGE_TYPE_EVALUATE = "evaluate"


class SType:
    """Serialisation type."""

    NUMPY = "numpy.ndarray"

    def __new__(cls) -> SType:
        """Prevent instantiation."""
        raise TypeError(f"{cls.__name__} cannot be instantiated.")
