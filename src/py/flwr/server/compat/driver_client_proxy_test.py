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
"""DriverClientProxy tests."""


import unittest
import unittest.mock
from typing import Any, Callable, Iterable, Optional, Union, cast
from unittest.mock import Mock

import numpy as np

import flwr
from flwr.common import Error, Message, Metadata, RecordSet
from flwr.common import recordset_compat as compat
from flwr.common.typing import (
    Code,
    Config,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    Properties,
    Status,
)
from flwr.server.compat.driver_client_proxy import DriverClientProxy

MESSAGE_PARAMETERS = Parameters(tensors=[b"abc"], tensor_type="np")

CLIENT_PROPERTIES = cast(Properties, {"tensor_type": "numpy.ndarray"})
CLIENT_STATUS = Status(code=Code.OK, message="OK")

ERROR_REPLY = Error(code=0, reason="mock error")

RUN_ID = 61016
NODE_ID = 1
INSTRUCTION_MESSAGE_ID = "mock instruction message id"
REPLY_MESSAGE_ID = "mock reply message id"


class DriverClientProxyTestCase(unittest.TestCase):
    """Tests for DriverClientProxy."""

    def setUp(self) -> None:
        """Set up mocks for tests."""
        driver = Mock()
        driver.get_node_ids.return_value = [1]
        driver.create_message.side_effect = self._create_message_dummy
        client = DriverClientProxy(
            node_id=NODE_ID, driver=driver, anonymous=False, run_id=61016
        )

        self.driver = driver
        self.client = client
        self.created_msg: Optional[Message] = None
        self.called_times: int = 0

    def test_get_properties(self) -> None:
        """Test positive case."""
        # Prepare
        res = GetPropertiesRes(status=CLIENT_STATUS, properties=CLIENT_PROPERTIES)
        self.driver.push_messages.side_effect = self._get_push_messages(res)
        request_properties: Config = {"tensor_type": "str"}
        ins = GetPropertiesIns(config=request_properties)

        # Execute
        value = self.client.get_properties(ins, timeout=None, group_id=0)

        # Assert
        self._common_assertions(ins)
        self.assertEqual(value.properties["tensor_type"], "numpy.ndarray")

    def test_get_parameters(self) -> None:
        """Test positive case."""
        # Prepare
        res = GetParametersRes(
            status=CLIENT_STATUS,
            parameters=MESSAGE_PARAMETERS,
        )
        self.driver.push_messages.side_effect = self._get_push_messages(res)
        ins = GetParametersIns(config={})

        # Execute
        value = self.client.get_parameters(ins, timeout=None, group_id=0)

        # Assert
        self._common_assertions(ins)
        self.assertEqual(value, res)

    def test_fit(self) -> None:
        """Test positive case."""
        # Prepare
        res = FitRes(
            status=CLIENT_STATUS,
            parameters=MESSAGE_PARAMETERS,
            num_examples=10,
            metrics={},
        )
        self.driver.push_messages.side_effect = self._get_push_messages(res)
        parameters = flwr.common.ndarrays_to_parameters([np.ones((2, 2))])
        ins = FitIns(parameters, {})

        # Execute
        value = self.client.fit(ins=ins, timeout=None, group_id=0)

        # Assert
        self._common_assertions(ins)
        self.assertEqual(value, res)

    def test_evaluate(self) -> None:
        """Test positive case."""
        # Prepare
        res = EvaluateRes(
            status=CLIENT_STATUS,
            loss=0.0,
            num_examples=0,
            metrics={},
        )
        self.driver.push_messages.side_effect = self._get_push_messages(res)
        parameters = Parameters(tensors=[b"random params%^&*F"], tensor_type="np")
        ins = EvaluateIns(parameters, {})

        # Execute
        value = self.client.evaluate(ins, timeout=None, group_id=0)

        # Assert
        self._common_assertions(ins)
        self.assertEqual(value, res)

    def test_get_properties_and_fail(self) -> None:
        """Test negative case."""
        # Prepare
        self.driver.push_messages.side_effect = self._get_push_messages(
            None, error_reply=True
        )
        request_properties: Config = {"tensor_type": "str"}
        ins = GetPropertiesIns(config=request_properties)

        # Execute and assert
        self.assertRaises(
            Exception, self.client.get_properties, ins, timeout=None, group_id=0
        )
        self._common_assertions(ins)

    def test_get_parameters_and_fail(self) -> None:
        """Test negative case."""
        # Prepare
        self.driver.push_messages.side_effect = self._get_push_messages(
            None, error_reply=True
        )
        ins = GetParametersIns(config={})

        # Execute and assert
        self.assertRaises(
            Exception, self.client.get_parameters, ins, timeout=None, group_id=0
        )
        self._common_assertions(ins)

    def test_fit_and_fail(self) -> None:
        """Test negative case."""
        # Prepare
        self.driver.push_messages.side_effect = self._get_push_messages(
            None, error_reply=True
        )
        parameters = flwr.common.ndarrays_to_parameters([np.ones((2, 2))])
        ins = FitIns(parameters, {})

        # Execute and assert
        self.assertRaises(Exception, self.client.fit, ins, timeout=None, group_id=0)
        self._common_assertions(ins)

    def test_evaluate_and_fail(self) -> None:
        """Test negative case."""
        # Prepare
        self.driver.push_messages.side_effect = self._get_push_messages(
            None, error_reply=True
        )
        parameters = Parameters(tensors=[b"random params%^&*F"], tensor_type="np")
        ins = EvaluateIns(parameters, {})

        # Execute and assert
        self.assertRaises(
            Exception, self.client.evaluate, ins, timeout=None, group_id=0
        )
        self._common_assertions(ins)

    def _create_message_dummy(  # pylint: disable=R0913
        self,
        content: RecordSet,
        message_type: str,
        dst_node_id: int,
        group_id: str,
        ttl: Optional[float] = None,
    ) -> Message:
        """Create a new message.

        This is a method for the Mock object.
        """
        self.called_times += 1
        ttl_ = 123456 if ttl is None else ttl
        metadata = Metadata(
            run_id=RUN_ID,
            message_id="",  # Will be set by the server
            src_node_id=0,
            dst_node_id=dst_node_id,
            reply_to_message="",
            group_id=group_id,
            ttl=ttl_,
            message_type=message_type,
        )
        self.created_msg = Message(metadata=metadata, content=content)
        return self.created_msg

    def _get_push_messages(
        self,
        res: Union[GetParametersRes, GetPropertiesRes, FitRes, EvaluateRes, None],
        error_reply: bool = False,
    ) -> Callable[[Iterable[Message]], Iterable[str]]:
        """Get the push_messages function that sets the return value of pull_messages
        when called."""

        def push_messages(messages: Iterable[Message]) -> Iterable[str]:
            msg = list(messages)[0]
            if error_reply:
                recordset = None
                ret = msg.create_error_reply(ERROR_REPLY)
            elif isinstance(res, GetParametersRes):
                recordset = compat.getparametersres_to_recordset(res, True)
            elif isinstance(res, GetPropertiesRes):
                recordset = compat.getpropertiesres_to_recordset(res)
            elif isinstance(res, FitRes):
                recordset = compat.fitres_to_recordset(res, True)
            elif isinstance(res, EvaluateRes):
                recordset = compat.evaluateres_to_recordset(res)
            else:
                raise ValueError(f"Unsupported type: {type(res)}")
            if recordset is not None:
                ret = msg.create_reply(recordset)
            ret.metadata.__dict__["_message_id"] = REPLY_MESSAGE_ID

            # Set the return value of `pull_messages`
            self.driver.pull_messages.return_value = [ret]
            return [INSTRUCTION_MESSAGE_ID]

        return push_messages

    def _common_assertions(self, original_ins: Any) -> None:
        """Check common assertions."""
        # Check if the created message contains the orignal *Ins
        assert self.created_msg is not None
        actual_ins = {  # type: ignore
            GetPropertiesIns: compat.recordset_to_getpropertiesins,
            GetParametersIns: compat.recordset_to_getparametersins,
            FitIns: (lambda x: compat.recordset_to_fitins(x, True)),
            EvaluateIns: (lambda x: compat.recordset_to_evaluateins(x, True)),
        }[type(original_ins)](self.created_msg.content)
        self.assertEqual(self.called_times, 1)
        self.assertEqual(actual_ins, original_ins)

        # Check if push_messages is called once with expected args/kwargs.
        self.driver.push_messages.assert_called_once()
        try:
            self.driver.push_messages.assert_any_call([self.created_msg])
        except AssertionError:
            self.driver.push_messages.assert_any_call(messages=[self.created_msg])

        # Check if pull_messages is called once with expected args/kwargs.
        self.driver.pull_messages.assert_called_once()
        try:
            self.driver.pull_messages.assert_called_with([INSTRUCTION_MESSAGE_ID])
        except AssertionError:
            self.driver.pull_messages.assert_called_with(
                message_ids=[INSTRUCTION_MESSAGE_ID]
            )
