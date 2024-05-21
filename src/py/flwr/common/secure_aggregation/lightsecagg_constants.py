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
"""Constants for the LightSecAgg protocol."""


from __future__ import annotations

RECORD_KEY_STATE = "lightsecagg_state"
RECORD_KEY_CONFIGS = "lightsecagg_configs"
RATIO_QUANTIZATION_RANGE = 1073741824  # 1 << 30


class Stage:
    """Stages for the SecAgg+ protocol."""

    SETUP = "setup"
    EXCHANGE_SUB_MASKS = "exchange_sub_masks"
    COLLECT_MASKED_MODELS = "collect_masked_models"
    UNMASK = "unmask"

    @classmethod
    def all(cls) -> tuple[str, str, str, str]:
        """Return all stages."""
        return (
            cls.SETUP,
            cls.EXCHANGE_SUB_MASKS,
            cls.COLLECT_MASKED_MODELS,
            cls.UNMASK,
        )

    def __new__(cls) -> Stage:
        """Prevent instantiation."""
        raise TypeError(f"{cls.__name__} cannot be instantiated.")


class Key:
    """Keys for the configs in the ConfigsRecord."""

    STAGE = "stage"
    SAMPLE_NUMBER = "sample_num"
    PRIVACY_GUARANTEE = "privacy_guarantee"
    MIN_ACTIVE_CLIENTS = "min_active_clients"
    PRIME_NUMBER = "prime_number"
    MODEL_SIZE = "model_size"

    CLIPPING_RANGE = "clipping_range"
    TARGET_RANGE = "target_range"
    MAX_WEIGHT = "max_weight"

    PUBLIC_KEY = "pk"
    DESTINATION_LIST = "dsts"
    CIPHERTEXT_LIST = "ctxts"
    SOURCE_LIST = "srcs"
    MASKED_PARAMETERS = "masked_params"
    ACTIVE_NODE_ID_LIST = "active_nids"
    AGGREGATED_ENCODED_MASK = "aggregated encoded mask"

    def __new__(cls) -> Key:
        """Prevent instantiation."""
        raise TypeError(f"{cls.__name__} cannot be instantiated.")
