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
"""The SecAgg+ protocol handler tests."""

import unittest
from itertools import product
from typing import Callable, Dict, List

from flwr.client.mod import make_ffn
from flwr.common.configsrecord import ConfigsRecord
from flwr.common.constant import TASK_TYPE_FIT
from flwr.common.context import Context
from flwr.common.message import Message, Metadata
from flwr.common.recordset import RecordSet
from flwr.common.secure_aggregation.secaggplus_constants import (
    KEY_ACTIVE_SECURE_ID_LIST,
    KEY_CIPHERTEXT_LIST,
    KEY_CLIPPING_RANGE,
    KEY_DEAD_SECURE_ID_LIST,
    KEY_MOD_RANGE,
    KEY_SAMPLE_NUMBER,
    KEY_SECURE_ID,
    KEY_SHARE_NUMBER,
    KEY_SOURCE_LIST,
    KEY_STAGE,
    KEY_TARGET_RANGE,
    KEY_THRESHOLD,
    RECORD_KEY_CONFIGS,
    RECORD_KEY_STATE,
    STAGE_COLLECT_MASKED_INPUT,
    STAGE_SETUP,
    STAGE_SHARE_KEYS,
    STAGE_UNMASK,
    STAGES,
)
from flwr.common.typing import ConfigsRecordValues

from .secaggplus_mod import SecAggPlusState, check_configs, secaggplus_mod


def get_test_handler(
    ctxt: Context,
) -> Callable[[Dict[str, ConfigsRecordValues]], Dict[str, ConfigsRecordValues]]:
    """."""

    def empty_ffn(_: Message, _2: Context) -> Message:
        return Message(
            metadata=Metadata(
                run_id=0,
                task_id="",
                group_id="",
                node_id=0,
                ttl="",
                task_type=TASK_TYPE_FIT,
            ),
            content=RecordSet(),
        )

    app = make_ffn(empty_ffn, [secaggplus_mod])

    def func(configs: Dict[str, ConfigsRecordValues]) -> Dict[str, ConfigsRecordValues]:
        in_msg = Message(
            metadata=Metadata(
                run_id=0,
                task_id="",
                group_id="",
                node_id=0,
                ttl="",
                task_type=TASK_TYPE_FIT,
            ),
            content=RecordSet(configs={RECORD_KEY_CONFIGS: ConfigsRecord(configs)}),
        )
        out_msg = app(in_msg, ctxt)
        return out_msg.content.get_configs(RECORD_KEY_CONFIGS).data

    return func


def _make_ctxt() -> Context:
    cfg = ConfigsRecord(SecAggPlusState().to_dict())
    return Context(RecordSet(configs={RECORD_KEY_STATE: cfg}))


def _make_set_state_fn(
    ctxt: Context,
) -> Callable[[str], None]:
    def set_stage(stage: str) -> None:
        state_dict = ctxt.state.get_configs(RECORD_KEY_STATE).data
        state = SecAggPlusState(**state_dict)
        state.current_stage = stage
        ctxt.state.set_configs(RECORD_KEY_STATE, ConfigsRecord(state.to_dict()))

    return set_stage


class TestSecAggPlusHandler(unittest.TestCase):
    """Test the SecAgg+ protocol handler."""

    def test_stage_transition(self) -> None:
        """Test stage transition."""
        ctxt = _make_ctxt()
        handler = get_test_handler(ctxt)
        set_stage = _make_set_state_fn(ctxt)

        assert STAGES == (
            STAGE_SETUP,
            STAGE_SHARE_KEYS,
            STAGE_COLLECT_MASKED_INPUT,
            STAGE_UNMASK,
        )

        valid_transitions = {
            # From one stage to the next stage
            (STAGE_UNMASK, STAGE_SETUP),
            (STAGE_SETUP, STAGE_SHARE_KEYS),
            (STAGE_SHARE_KEYS, STAGE_COLLECT_MASKED_INPUT),
            (STAGE_COLLECT_MASKED_INPUT, STAGE_UNMASK),
            # From any stage to the initial stage
            # Such transitions will log a warning.
            (STAGE_SETUP, STAGE_SETUP),
            (STAGE_SHARE_KEYS, STAGE_SETUP),
            (STAGE_COLLECT_MASKED_INPUT, STAGE_SETUP),
        }

        invalid_transitions = set(product(STAGES, STAGES)).difference(valid_transitions)

        # Test valid transitions
        # If the next stage is valid, the function should update the current stage
        # and then raise KeyError or other exceptions when trying to execute SA.
        for current_stage, next_stage in valid_transitions:
            set_stage(current_stage)

            with self.assertRaises(KeyError):
                handler({KEY_STAGE: next_stage})

        # Test invalid transitions
        # If the next stage is invalid, the function should raise ValueError
        for current_stage, next_stage in invalid_transitions:
            set_stage(current_stage)

            with self.assertRaises(ValueError):
                handler({KEY_STAGE: next_stage})

    def test_stage_setup_check(self) -> None:
        """Test content checking for the setup stage."""
        ctxt = _make_ctxt()
        handler = get_test_handler(ctxt)
        set_stage = _make_set_state_fn(ctxt)

        valid_key_type_pairs = [
            (KEY_SAMPLE_NUMBER, int),
            (KEY_SECURE_ID, int),
            (KEY_SHARE_NUMBER, int),
            (KEY_THRESHOLD, int),
            (KEY_CLIPPING_RANGE, float),
            (KEY_TARGET_RANGE, int),
            (KEY_MOD_RANGE, int),
        ]

        type_to_test_value: Dict[type, ConfigsRecordValues] = {
            int: 10,
            bool: True,
            float: 1.0,
            str: "test",
            bytes: b"test",
        }

        valid_configs: Dict[str, ConfigsRecordValues] = {
            key: type_to_test_value[value_type]
            for key, value_type in valid_key_type_pairs
        }

        # Test valid `named_values`
        try:
            check_configs(STAGE_SETUP, valid_configs.copy())
        # pylint: disable-next=broad-except
        except Exception as exc:
            self.fail(f"check_named_values() raised {type(exc)} unexpectedly!")

        # Set the stage
        valid_configs[KEY_STAGE] = STAGE_SETUP

        # Test invalid `named_values`
        for key, value_type in valid_key_type_pairs:
            invalid_configs = valid_configs.copy()

            # Test wrong value type for the key
            for other_type, other_value in type_to_test_value.items():
                if other_type == value_type:
                    continue
                invalid_configs[key] = other_value

                set_stage(STAGE_UNMASK)
                with self.assertRaises(TypeError):
                    handler(invalid_configs.copy())

            # Test missing key
            invalid_configs.pop(key)

            set_stage(STAGE_UNMASK)
            with self.assertRaises(KeyError):
                handler(invalid_configs.copy())

    def test_stage_share_keys_check(self) -> None:
        """Test content checking for the share keys stage."""
        ctxt = _make_ctxt()
        handler = get_test_handler(ctxt)
        set_stage = _make_set_state_fn(ctxt)

        valid_configs: Dict[str, ConfigsRecordValues] = {
            "1": [b"public key 1", b"public key 2"],
            "2": [b"public key 1", b"public key 2"],
            "3": [b"public key 1", b"public key 2"],
        }

        # Test valid `named_values`
        try:
            check_configs(STAGE_SHARE_KEYS, valid_configs.copy())
        # pylint: disable-next=broad-except
        except Exception as exc:
            self.fail(f"check_named_values() raised {type(exc)} unexpectedly!")

        # Set the stage
        valid_configs[KEY_STAGE] = STAGE_SHARE_KEYS

        # Test invalid `named_values`
        invalid_values: List[ConfigsRecordValues] = [
            b"public key 1",
            [b"public key 1"],
            [b"public key 1", b"public key 2", b"public key 3"],
        ]

        for value in invalid_values:
            invalid_configs = valid_configs.copy()
            invalid_configs["1"] = value

            set_stage(STAGE_SETUP)
            with self.assertRaises(TypeError):
                handler(invalid_configs.copy())

    def test_stage_collect_masked_input_check(self) -> None:
        """Test content checking for the collect masked input stage."""
        ctxt = _make_ctxt()
        handler = get_test_handler(ctxt)
        set_stage = _make_set_state_fn(ctxt)

        valid_configs: Dict[str, ConfigsRecordValues] = {
            KEY_CIPHERTEXT_LIST: [b"ctxt!", b"ctxt@", b"ctxt#", b"ctxt?"],
            KEY_SOURCE_LIST: [32, 51324, 32324123, -3],
        }

        # Test valid `named_values`
        try:
            check_configs(STAGE_COLLECT_MASKED_INPUT, valid_configs.copy())
        # pylint: disable-next=broad-except
        except Exception as exc:
            self.fail(f"check_named_values() raised {type(exc)} unexpectedly!")

        # Set the stage
        valid_configs[KEY_STAGE] = STAGE_COLLECT_MASKED_INPUT

        # Test invalid `named_values`
        # Test missing keys
        for key in list(valid_configs.keys()):
            if key == KEY_STAGE:
                continue
            invalid_configs = valid_configs.copy()
            invalid_configs.pop(key)

            set_stage(STAGE_SHARE_KEYS)
            with self.assertRaises(KeyError):
                handler(invalid_configs)

        # Test wrong value type for the key
        for key in valid_configs:
            if key == KEY_STAGE:
                continue
            invalid_configs = valid_configs.copy()
            invalid_configs[key] = [3.1415926]

            set_stage(STAGE_SHARE_KEYS)
            with self.assertRaises(TypeError):
                handler(invalid_configs)

    def test_stage_unmask_check(self) -> None:
        """Test content checking for the unmasking stage."""
        ctxt = _make_ctxt()
        handler = get_test_handler(ctxt)
        set_stage = _make_set_state_fn(ctxt)

        valid_configs: Dict[str, ConfigsRecordValues] = {
            KEY_ACTIVE_SECURE_ID_LIST: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            KEY_DEAD_SECURE_ID_LIST: [32, 51324, 32324123, -3],
        }

        # Test valid `named_values`
        try:
            check_configs(STAGE_UNMASK, valid_configs.copy())
        # pylint: disable-next=broad-except
        except Exception as exc:
            self.fail(f"check_named_values() raised {type(exc)} unexpectedly!")

        # Set the stage
        valid_configs[KEY_STAGE] = STAGE_UNMASK

        # Test invalid `named_values`
        # Test missing keys
        for key in list(valid_configs.keys()):
            if key == KEY_STAGE:
                continue
            invalid_configs = valid_configs.copy()
            invalid_configs.pop(key)

            set_stage(STAGE_COLLECT_MASKED_INPUT)
            with self.assertRaises(KeyError):
                handler(invalid_configs)

        # Test wrong value type for the key
        for key in valid_configs:
            if key == KEY_STAGE:
                continue
            invalid_configs = valid_configs.copy()
            invalid_configs[key] = [True, False, True, False]

            set_stage(STAGE_COLLECT_MASKED_INPUT)
            with self.assertRaises(TypeError):
                handler(invalid_configs)
