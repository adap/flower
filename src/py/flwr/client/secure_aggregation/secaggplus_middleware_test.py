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
"""The SecAgg+ protocol handler tests."""

import unittest
from itertools import product
from typing import Callable, Dict, List

from flwr.client.middleware import make_app
from flwr.client.typing import Bwd, Fwd
from flwr.client.workload_state import WorkloadState
from flwr.common import serde
from flwr.common.secure_aggregation.secaggplus_constants import (
    KEY_ACTIVE_SECURE_ID_LIST,
    KEY_CIPHERTEXT_LIST,
    KEY_CLIPPING_RANGE,
    KEY_DEAD_SECURE_ID_LIST,
    KEY_MOD_RANGE,
    KEY_SAMPLE_NUMBER,
    KEY_SECAGGPLUS_STATE,
    KEY_SECURE_ID,
    KEY_SHARE_NUMBER,
    KEY_SOURCE_LIST,
    KEY_STAGE,
    KEY_TARGET_RANGE,
    KEY_THRESHOLD,
    STAGE_COLLECT_MASKED_INPUT,
    STAGE_SETUP,
    STAGE_SHARE_KEYS,
    STAGE_UNMASK,
    STAGES,
)
from flwr.common.typing import Value
from flwr.proto.task_pb2 import SecureAggregation, Task, TaskIns, TaskRes

from .secaggplus_middleware import (
    SecAggPlusState,
    check_named_values,
    secaggplus_middleware,
)


def get_test_handler(
    state: SecAggPlusState,
) -> Callable[[Dict[str, Value]], Dict[str, Value]]:
    """."""

    def empty_app(_: Fwd) -> Bwd:
        return Bwd(task_res=TaskRes(), state=WorkloadState(state={}))

    app = make_app(empty_app, [secaggplus_middleware])
    workload_state = WorkloadState(state={KEY_SECAGGPLUS_STATE: state})  # type: ignore

    def func(named_values: Dict[str, Value]) -> Dict[str, Value]:
        bwd = app(
            Fwd(
                task_ins=TaskIns(
                    task=Task(
                        sa=SecureAggregation(
                            named_values=serde.named_values_to_proto(named_values)
                        )
                    )
                ),
                state=workload_state,
            )
        )
        return serde.named_values_from_proto(bwd.task_res.task.sa.named_values)

    return func


class TestSecAggPlusHandler(unittest.TestCase):
    """Test the SecAgg+ protocol handler."""

    def test_stage_transition(self) -> None:
        """Test stage transition."""
        state = SecAggPlusState()
        handler = get_test_handler(state)

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
            state.current_stage = current_stage

            with self.assertRaises(KeyError):
                handler({KEY_STAGE: next_stage})

            assert state.current_stage == next_stage

        # Test invalid transitions
        # If the next stage is invalid, the function should raise ValueError
        for current_stage, next_stage in invalid_transitions:
            state.current_stage = current_stage

            with self.assertRaises(ValueError):
                handler({KEY_STAGE: next_stage})

            assert state.current_stage == current_stage

    def test_stage_setup_check(self) -> None:
        """Test content checking for the setup stage."""
        state = SecAggPlusState()
        handler = get_test_handler(state)

        valid_key_type_pairs = [
            (KEY_SAMPLE_NUMBER, int),
            (KEY_SECURE_ID, int),
            (KEY_SHARE_NUMBER, int),
            (KEY_THRESHOLD, int),
            (KEY_CLIPPING_RANGE, float),
            (KEY_TARGET_RANGE, int),
            (KEY_MOD_RANGE, int),
        ]

        type_to_test_value: Dict[type, Value] = {
            int: 10,
            bool: True,
            float: 1.0,
            str: "test",
            bytes: b"test",
        }

        valid_named_values: Dict[str, Value] = {
            key: type_to_test_value[value_type]
            for key, value_type in valid_key_type_pairs
        }

        # Test valid `named_values`
        try:
            check_named_values(STAGE_SETUP, valid_named_values.copy())
        # pylint: disable-next=broad-except
        except Exception as exc:
            self.fail(f"check_named_values() raised {type(exc)} unexpectedly!")

        # Set the stage
        valid_named_values[KEY_STAGE] = STAGE_SETUP

        # Test invalid `named_values`
        for key, value_type in valid_key_type_pairs:
            invalid_named_values = valid_named_values.copy()

            # Test wrong value type for the key
            for other_type, other_value in type_to_test_value.items():
                if other_type == value_type:
                    continue
                invalid_named_values[key] = other_value

                state.current_stage = STAGE_UNMASK
                with self.assertRaises(TypeError):
                    handler(invalid_named_values.copy())

            # Test missing key
            invalid_named_values.pop(key)

            state.current_stage = STAGE_UNMASK
            with self.assertRaises(KeyError):
                handler(invalid_named_values.copy())

    def test_stage_share_keys_check(self) -> None:
        """Test content checking for the share keys stage."""
        state = SecAggPlusState()
        handler = get_test_handler(state)

        valid_named_values: Dict[str, Value] = {
            "1": [b"public key 1", b"public key 2"],
            "2": [b"public key 1", b"public key 2"],
            "3": [b"public key 1", b"public key 2"],
        }

        # Test valid `named_values`
        try:
            check_named_values(STAGE_SHARE_KEYS, valid_named_values.copy())
        # pylint: disable-next=broad-except
        except Exception as exc:
            self.fail(f"check_named_values() raised {type(exc)} unexpectedly!")

        # Set the stage
        valid_named_values[KEY_STAGE] = STAGE_SHARE_KEYS

        # Test invalid `named_values`
        invalid_values: List[Value] = [
            b"public key 1",
            [b"public key 1"],
            [b"public key 1", b"public key 2", b"public key 3"],
        ]

        for value in invalid_values:
            invalid_named_values = valid_named_values.copy()
            invalid_named_values["1"] = value

            state.current_stage = STAGE_SETUP
            with self.assertRaises(TypeError):
                handler(invalid_named_values.copy())

    def test_stage_collect_masked_input_check(self) -> None:
        """Test content checking for the collect masked input stage."""
        state = SecAggPlusState()
        handler = get_test_handler(state)

        valid_named_values: Dict[str, Value] = {
            KEY_CIPHERTEXT_LIST: [b"ctxt!", b"ctxt@", b"ctxt#", b"ctxt?"],
            KEY_SOURCE_LIST: [32, 51324, 32324123, -3],
        }

        # Test valid `named_values`
        try:
            check_named_values(STAGE_COLLECT_MASKED_INPUT, valid_named_values.copy())
        # pylint: disable-next=broad-except
        except Exception as exc:
            self.fail(f"check_named_values() raised {type(exc)} unexpectedly!")

        # Set the stage
        valid_named_values[KEY_STAGE] = STAGE_COLLECT_MASKED_INPUT

        # Test invalid `named_values`
        # Test missing keys
        for key in list(valid_named_values.keys()):
            if key == KEY_STAGE:
                continue
            invalid_named_values = valid_named_values.copy()
            invalid_named_values.pop(key)

            state.current_stage = STAGE_SHARE_KEYS
            with self.assertRaises(KeyError):
                handler(invalid_named_values)

        # Test wrong value type for the key
        for key in valid_named_values:
            if key == KEY_STAGE:
                continue
            invalid_named_values = valid_named_values.copy()
            invalid_named_values[key] = [3.1415926]

            state.current_stage = STAGE_SHARE_KEYS
            with self.assertRaises(TypeError):
                handler(invalid_named_values)

    def test_stage_unmask_check(self) -> None:
        """Test content checking for the unmasking stage."""
        state = SecAggPlusState()
        handler = get_test_handler(state)

        valid_named_values: Dict[str, Value] = {
            KEY_ACTIVE_SECURE_ID_LIST: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            KEY_DEAD_SECURE_ID_LIST: [32, 51324, 32324123, -3],
        }

        # Test valid `named_values`
        try:
            check_named_values(STAGE_UNMASK, valid_named_values.copy())
        # pylint: disable-next=broad-except
        except Exception as exc:
            self.fail(f"check_named_values() raised {type(exc)} unexpectedly!")

        # Set the stage
        valid_named_values[KEY_STAGE] = STAGE_UNMASK

        # Test invalid `named_values`
        # Test missing keys
        for key in list(valid_named_values.keys()):
            if key == KEY_STAGE:
                continue
            invalid_named_values = valid_named_values.copy()
            invalid_named_values.pop(key)

            state.current_stage = STAGE_COLLECT_MASKED_INPUT
            with self.assertRaises(KeyError):
                handler(invalid_named_values)

        # Test wrong value type for the key
        for key in valid_named_values:
            if key == KEY_STAGE:
                continue
            invalid_named_values = valid_named_values.copy()
            invalid_named_values[key] = [True, False, True, False]

            state.current_stage = STAGE_COLLECT_MASKED_INPUT
            with self.assertRaises(TypeError):
                handler(invalid_named_values)
