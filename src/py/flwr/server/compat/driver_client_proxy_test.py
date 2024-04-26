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
"""DriverClientProxy tests."""


import unittest
import unittest.mock
from typing import Optional, Union, cast
from unittest.mock import Mock

import numpy as np

import flwr
from flwr.common import Message, Metadata, RecordSet
from flwr.common import recordset_compat as compat
from flwr.common.constant import MessageType, MessageTypeLegacy
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
from flwr.proto import (  # pylint: disable=E0611
    driver_pb2,
    error_pb2,
    recordset_pb2,
    task_pb2,
)
from flwr.server.compat.driver_client_proxy import DriverClientProxy

MESSAGE_PARAMETERS = Parameters(tensors=[b"abc"], tensor_type="np")

CLIENT_PROPERTIES = cast(Properties, {"tensor_type": "numpy.ndarray"})
CLIENT_STATUS = Status(code=Code.OK, message="OK")

RUN_ID = 61016
NODE_ID = 1
INSTRUCTION_MESSAGE_ID = "mock instruction message id"
REPLY_MESSAGE_ID = "mock reply message id"


def _make_reply_message(
    res: Union[GetParametersRes, GetPropertiesRes, FitRes, EvaluateRes]
) -> Message:
    if isinstance(res, GetParametersRes):
        message_type = MessageTypeLegacy.GET_PARAMETERS
        recordset = compat.getparametersres_to_recordset(res, True)
    elif isinstance(res, GetPropertiesRes):
        message_type = MessageTypeLegacy.GET_PROPERTIES
        recordset = compat.getpropertiesres_to_recordset(res)
    elif isinstance(res, FitRes):
        message_type = MessageType.TRAIN
        recordset = compat.fitres_to_recordset(res, True)
    elif isinstance(res, EvaluateRes):
        message_type = MessageType.EVALUATE
        recordset = compat.evaluateres_to_recordset(res)
    else:
        raise ValueError(f"Unsupported type: {type(res)}")
    metadata = Metadata(
        run_id=RUN_ID,
        message_id=REPLY_MESSAGE_ID,
        src_node_id=NODE_ID,
        dst_node_id=0,
        reply_to_message=INSTRUCTION_MESSAGE_ID,
        group_id=0,
        ttl=99,
        message_type=message_type,
    )
    return Message(metadata, recordset)


def _create_message_dummy(
    content: RecordSet,
    message_type: str,
    dst_node_id: int,
    group_id: str,
    ttl: Optional[float] = None,
) -> Message:
    """Create a new message.

    This is a method for the Mock object.
    """
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
    return Message(metadata=metadata, content=content)


def validate_task_res(
    task_res: task_pb2.TaskRes,  # pylint: disable=E1101
) -> None:
    """Validate if a TaskRes is empty or not."""
    if not task_res.HasField("task"):
        raise ValueError("Invalid TaskRes, field `task` missing")
    if task_res.task.HasField("error"):
        raise ValueError("Exception during client-side task execution")
    if not task_res.task.HasField("recordset"):
        raise ValueError("Invalid TaskRes, both `recordset` and `error` are missing")


class DriverClientProxyTestCase(unittest.TestCase):
    """Tests for DriverClientProxy."""

    def setUp(self) -> None:
        """Set up mocks for tests."""
        driver = Mock()
        driver.get_node_ids.return_value = [1]
        driver.create_message = _create_message_dummy
        driver.push_messages.return_value = [INSTRUCTION_MESSAGE_ID]
        client = DriverClientProxy(
            node_id=1, driver=driver, anonymous=False, run_id=61016
        )

        self.driver = driver
        self.client = client

    def test_get_properties(self) -> None:
        """Test positive case."""
        # Prepare
        reply_msg = _make_reply_message(
            GetPropertiesRes(status=CLIENT_STATUS, properties=CLIENT_PROPERTIES)
        )
        self.driver.pull_messages.return_value = [reply_msg]
        request_properties: Config = {"tensor_type": "str"}
        ins = GetPropertiesIns(config=request_properties)

        # Execute
        res = self.client.get_properties(ins, timeout=None, group_id=0)

        # Assert
        self.driver.push_messages.assert_called_once()
        self.driver.pull_messages.assert_called_once()
        self.driver.pull_messages.assert_called_with([INSTRUCTION_MESSAGE_ID])
        self.assertEqual(res.properties["tensor_type"], "numpy.ndarray")

    def test_get_parameters(self) -> None:
        """Test positive case."""
        # Prepare
        expected_res = GetParametersRes(
            status=CLIENT_STATUS,
            parameters=MESSAGE_PARAMETERS,
        )
        reply_msg = _make_reply_message(expected_res)
        self.driver.pull_messages.return_value = [reply_msg]
        ins = GetParametersIns(config={})

        # Execute
        value = self.client.get_parameters(ins=ins, timeout=None, group_id=0)

        # Assert
        self.driver.push_messages.assert_called_once()
        self.driver.pull_messages.assert_called_once()
        self.driver.pull_messages.assert_called_with([INSTRUCTION_MESSAGE_ID])
        self.assertEqual(value, expected_res)

    def test_fit(self) -> None:
        """Test positive case."""
        # Prepare
        expected_res = FitRes(
            status=CLIENT_STATUS,
            parameters=MESSAGE_PARAMETERS,
            num_examples=10,
            metrics={},
        )
        reply_msg = _make_reply_message(expected_res)
        self.driver.pull_messages.return_value = [reply_msg]
        parameters = flwr.common.ndarrays_to_parameters([np.ones((2, 2))])
        ins = FitIns(parameters, {})

        # Execute
        value = self.client.fit(ins=ins, timeout=None, group_id=1)

        # Assert
        self.driver.push_messages.assert_called_once()
        self.driver.pull_messages.assert_called_once()
        self.driver.pull_messages.assert_called_with([INSTRUCTION_MESSAGE_ID])
        self.assertEqual(value, expected_res)

    def test_evaluate(self) -> None:
        """Test positive case."""
        # Prepare
        if self.driver.driver_helper is None:  # type: ignore
            raise ValueError()
        self.driver.driver_helper.push_task_ins.return_value = (  # type: ignore
            driver_pb2.PushTaskInsResponse(  # pylint: disable=E1101
                task_ids=["19341fd7-62e1-4eb4-beb4-9876d3acda32"]
            )
        )
        self.driver.driver_helper.pull_task_res.return_value = (  # type: ignore
            driver_pb2.PullTaskResResponse(  # pylint: disable=E1101
                task_res_list=[
                    task_pb2.TaskRes(  # pylint: disable=E1101
                        task_id="554bd3c8-8474-4b93-a7db-c7bec1bf0012",
                        group_id=str(1),
                        run_id=0,
                        task=_make_message(
                            EvaluateRes(
                                status=CLIENT_STATUS,
                                loss=0.0,
                                num_examples=0,
                                metrics={},
                            )
                        ),
                    )
                ]
            )
        )
        client = DriverClientProxy(
            node_id=1, driver=self.driver, anonymous=True, run_id=0
        )
        parameters = Parameters(tensors=[], tensor_type="np")
        evaluate_ins = EvaluateIns(parameters, {})

        # Execute
        evaluate_res = client.evaluate(evaluate_ins, timeout=None, group_id=1)

        # Assert
        assert 0.0 == evaluate_res.loss
        assert 0 == evaluate_res.num_examples

    def test_validate_task_res_valid(self) -> None:
        """Test valid TaskRes."""
        metrics_record = recordset_pb2.MetricsRecord(  # pylint: disable=E1101
            data={
                "loss": recordset_pb2.MetricsRecordValue(  # pylint: disable=E1101
                    double=1.0
                )
            }
        )
        task_res = task_pb2.TaskRes(  # pylint: disable=E1101
            task_id="554bd3c8-8474-4b93-a7db-c7bec1bf0012",
            group_id="",
            run_id=0,
            task=task_pb2.Task(  # pylint: disable=E1101
                recordset=recordset_pb2.RecordSet(  # pylint: disable=E1101
                    parameters={},
                    metrics={"loss": metrics_record},
                    configs={},
                )
            ),
        )

        # Execute & assert
        try:
            validate_task_res(task_res=task_res)
        except ValueError:
            self.fail()

    def test_validate_task_res_missing_task(self) -> None:
        """Test invalid TaskRes (missing task)."""
        # Prepare
        task_res = task_pb2.TaskRes(  # pylint: disable=E1101
            task_id="554bd3c8-8474-4b93-a7db-c7bec1bf0012",
            group_id="",
            run_id=0,
        )

        # Execute & assert
        with self.assertRaises(ValueError):
            validate_task_res(task_res=task_res)

    def test_validate_task_res_missing_recordset(self) -> None:
        """Test invalid TaskRes (missing recordset)."""
        # Prepare
        task_res = task_pb2.TaskRes(  # pylint: disable=E1101
            task_id="554bd3c8-8474-4b93-a7db-c7bec1bf0012",
            group_id="",
            run_id=0,
            task=task_pb2.Task(),  # pylint: disable=E1101
        )

        # Execute & assert
        with self.assertRaises(ValueError):
            validate_task_res(task_res=task_res)

    def test_validate_task_res_missing_content(self) -> None:
        """Test invalid TaskRes (missing content)."""
        # Prepare
        task_res = task_pb2.TaskRes(  # pylint: disable=E1101
            task_id="554bd3c8-8474-4b93-a7db-c7bec1bf0012",
            group_id="",
            run_id=0,
            task=task_pb2.Task(  # pylint: disable=E1101
                error=error_pb2.Error(  # pylint: disable=E1101
                    code=0,
                    reason="Some reason",
                )
            ),
        )

        # Execute & assert
        with self.assertRaises(ValueError):
            validate_task_res(task_res=task_res)
