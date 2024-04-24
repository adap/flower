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
from typing import Union, cast
from unittest.mock import MagicMock

import numpy as np

import flwr
from flwr.common import recordset_compat as compat
from flwr.common import serde
from flwr.common.constant import MessageType, MessageTypeLegacy
from flwr.common.typing import (
    Code,
    Config,
    EvaluateIns,
    EvaluateRes,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesRes,
    Parameters,
    Properties,
    Status,
)
from flwr.proto import (  # pylint: disable=E0611
    driver_pb2,
    error_pb2,
    node_pb2,
    recordset_pb2,
    task_pb2,
)
from flwr.server.compat.driver_client_proxy import DriverClientProxy, validate_task_res
from flwr.server.driver import GrpcDriver

MESSAGE_PARAMETERS = Parameters(tensors=[b"abc"], tensor_type="np")

CLIENT_PROPERTIES = cast(Properties, {"tensor_type": "numpy.ndarray"})
CLIENT_STATUS = Status(code=Code.OK, message="OK")


def _make_task(
    res: Union[GetParametersRes, GetPropertiesRes, FitRes, EvaluateRes]
) -> task_pb2.Task:  # pylint: disable=E1101
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
    return task_pb2.Task(  # pylint: disable=E1101
        task_type=message_type,
        recordset=serde.recordset_to_proto(recordset),
    )


class DriverClientProxyTestCase(unittest.TestCase):
    """Tests for DriverClientProxy."""

    def setUp(self) -> None:
        """Set up mocks for tests."""
        self.driver = GrpcDriver()
        self.driver.driver_helper = MagicMock()
        self.driver.run_id = 0
        self.driver.driver_helper.get_nodes.return_value = (
            driver_pb2.GetNodesResponse(  # pylint: disable=E1101
                nodes=[
                    node_pb2.Node(node_id=1, anonymous=False)  # pylint: disable=E1101
                ]
            )
        )

    def test_get_properties(self) -> None:
        """Test positive case."""
        # Prepare
        if self.driver.driver_helper is None:
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
                        group_id=str(0),
                        run_id=0,
                        task=_make_task(
                            GetPropertiesRes(
                                status=CLIENT_STATUS, properties=CLIENT_PROPERTIES
                            )
                        ),
                    )
                ]
            )
        )
        client = DriverClientProxy(
            node_id=1, driver=self.driver, anonymous=True, run_id=0
        )
        request_properties: Config = {"tensor_type": "str"}
        ins: flwr.common.GetPropertiesIns = flwr.common.GetPropertiesIns(
            config=request_properties
        )

        # Execute
        value: flwr.common.GetPropertiesRes = client.get_properties(
            ins, timeout=None, group_id=0
        )

        # Assert
        assert value.properties["tensor_type"] == "numpy.ndarray"

    def test_get_parameters(self) -> None:
        """Test positive case."""
        # Prepare
        if self.driver.driver_helper is None:
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
                        group_id=str(0),
                        run_id=0,
                        task=_make_task(
                            GetParametersRes(
                                status=CLIENT_STATUS,
                                parameters=MESSAGE_PARAMETERS,
                            )
                        ),
                    )
                ]
            )
        )
        client = DriverClientProxy(
            node_id=1, driver=self.driver, anonymous=True, run_id=0
        )
        get_parameters_ins = GetParametersIns(config={})

        # Execute
        value: flwr.common.GetParametersRes = client.get_parameters(
            ins=get_parameters_ins, timeout=None, group_id=0
        )

        # Assert
        assert value.parameters.tensors[0] == b"abc"

    def test_fit(self) -> None:
        """Test positive case."""
        # Prepare
        if self.driver.driver_helper is None:
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
                        task=_make_task(
                            FitRes(
                                status=CLIENT_STATUS,
                                parameters=MESSAGE_PARAMETERS,
                                num_examples=10,
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
        parameters = flwr.common.ndarrays_to_parameters([np.ones((2, 2))])
        ins: flwr.common.FitIns = flwr.common.FitIns(parameters, {})

        # Execute
        fit_res = client.fit(ins=ins, timeout=None, group_id=1)

        # Assert
        assert fit_res.parameters.tensor_type == "np"
        assert fit_res.parameters.tensors[0] == b"abc"
        assert fit_res.num_examples == 10

    def test_evaluate(self) -> None:
        """Test positive case."""
        # Prepare
        if self.driver.driver_helper is None:
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
                        task=_make_task(
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
