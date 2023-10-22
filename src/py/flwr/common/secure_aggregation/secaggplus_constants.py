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
"""Constants for the SecAgg/SecAgg+ protocol."""


# Names of stages
STAGE_SETUP = "setup"
STAGE_SHARE_KEYS = "share_keys"
STAGE_COLLECT_MASKED_INPUT = "collect_masked_input"
STAGE_UNMASK = "unmask"
STAGES = (STAGE_SETUP, STAGE_SHARE_KEYS, STAGE_COLLECT_MASKED_INPUT, STAGE_UNMASK)

# All valid keys in received/replied `named_values` dictionaries
KEY_STAGE = "stage"
KEY_SAMPLE_NUMBER = "sample_num"
KEY_SECURE_ID = "secure_id"
KEY_SHARE_NUMBER = "share_num"
KEY_THRESHOLD = "threshold"
KEY_CLIPPING_RANGE = "clipping_range"
KEY_TARGET_RANGE = "target_range"
KEY_MOD_RANGE = "mod_range"
KEY_PUBLIC_KEY_1 = "pk1"
KEY_PUBLIC_KEY_2 = "pk2"
KEY_DESTINATION_LIST = "dsts"
KEY_CIPHERTEXT_LIST = "ctxts"
KEY_SOURCE_LIST = "srcs"
KEY_PARAMETERS = "params"
KEY_MASKED_PARAMETERS = "masked_params"
KEY_ACTIVE_SECURE_ID_LIST = "active_sids"
KEY_DEAD_SECURE_ID_LIST = "dead_sids"
KEY_SECURE_ID_LIST = "sids"
KEY_SHARE_LIST = "shares"
