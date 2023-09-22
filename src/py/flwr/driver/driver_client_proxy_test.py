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
"""DriverClientProxy tests."""


import unittest
from unittest.mock import MagicMock

import numpy as np

import flwr
from flwr.common.typing import Config, GetParametersIns
from flwr.driver.driver_client_proxy import DriverClientProxy
from flwr.proto import driver_pb2, node_pb2, task_pb2
from flwr.proto.transport_pb2 import ClientMessage, Parameters, Scalar

MESSAGE_PARAMETERS = Parameters(tensors=[b"abc"], tensor_type="np")

CLIENT_PROPERTIES = {"tensor_type": Scalar(string="numpy.ndarray")}


class DriverClientProxyTestCase(unittest.TestCase):
    """Tests for DriverClientProxy."""

    def setUp(self) -> None:
        """Set up mocks for tests."""
        self.driver = MagicMock()
        self.driver.get_nodes.return_value = driver_pb2.GetNodesResponse(
            nodes=[node_pb2.Node(node_id=1, anonymous=False)]
        )

    def test_get_properties(self) -> None:
        """Test positive case."""
        # Prepare
        self.driver.push_task_ins.return_value = driver_pb2.PushTaskInsResponse(
            task_ids=["19341fd7-62e1-4eb4-beb4-9876d3acda32"]
        )
        self.driver.pull_task_res.return_value = driver_pb2.PullTaskResResponse(
            task_res_list=[
                task_pb2.TaskRes(
                    task_id="554bd3c8-8474-4b93-a7db-c7bec1bf0012",
                    group_id="",
                    workload_id="",
                    task=task_pb2.Task(
                        legacy_client_message=ClientMessage(
                            get_properties_res=ClientMessage.GetPropertiesRes(
                                properties=CLIENT_PROPERTIES
                            )
                        )
                    ),
                )
            ]
        )
        client = DriverClientProxy(
            node_id=1, driver=self.driver, anonymous=True, workload_id=""
        )
        request_properties: Config = {"tensor_type": "str"}
        ins: flwr.common.GetPropertiesIns = flwr.common.GetPropertiesIns(
            config=request_properties
        )

        # Execute
        value: flwr.common.GetPropertiesRes = client.get_properties(ins, timeout=None)

        # Assert
        assert value.properties["tensor_type"] == "numpy.ndarray"

    def test_get_parameters(self) -> None:
        """Test positive case."""
        # Prepare
        self.driver.push_task_ins.return_value = driver_pb2.PushTaskInsResponse(
            task_ids=["19341fd7-62e1-4eb4-beb4-9876d3acda32"]
        )
        self.driver.pull_task_res.return_value = driver_pb2.PullTaskResResponse(
            task_res_list=[
                task_pb2.TaskRes(
                    task_id="554bd3c8-8474-4b93-a7db-c7bec1bf0012",
                    group_id="",
                    workload_id="",
                    task=task_pb2.Task(
                        legacy_client_message=ClientMessage(
                            get_parameters_res=ClientMessage.GetParametersRes(
                                parameters=MESSAGE_PARAMETERS,
                            )
                        )
                    ),
                )
            ]
        )
        client = DriverClientProxy(
            node_id=1, driver=self.driver, anonymous=True, workload_id=""
        )
        get_parameters_ins = GetParametersIns(config={})

        # Execute
        value: flwr.common.GetParametersRes = client.get_parameters(
            ins=get_parameters_ins, timeout=None
        )

        # Assert
        assert value.parameters.tensors[0] == b"abc"

    def test_fit(self) -> None:
        """Test positive case."""
        # Prepare
        self.driver.push_task_ins.return_value = driver_pb2.PushTaskInsResponse(
            task_ids=["19341fd7-62e1-4eb4-beb4-9876d3acda32"]
        )
        self.driver.pull_task_res.return_value = driver_pb2.PullTaskResResponse(
            task_res_list=[
                task_pb2.TaskRes(
                    task_id="554bd3c8-8474-4b93-a7db-c7bec1bf0012",
                    group_id="",
                    workload_id="",
                    task=task_pb2.Task(
                        legacy_client_message=ClientMessage(
                            fit_res=ClientMessage.FitRes(
                                parameters=MESSAGE_PARAMETERS,
                                num_examples=10,
                            )
                        )
                    ),
                )
            ]
        )
        client = DriverClientProxy(
            node_id=1, driver=self.driver, anonymous=True, workload_id=""
        )
        parameters = flwr.common.ndarrays_to_parameters([np.ones((2, 2))])
        ins: flwr.common.FitIns = flwr.common.FitIns(parameters, {})

        # Execute
        fit_res = client.fit(ins=ins, timeout=None)

        # Assert
        assert fit_res.parameters.tensor_type == "np"
        assert fit_res.parameters.tensors[0] == b"abc"
        assert fit_res.num_examples == 10

    def test_evaluate(self) -> None:
        """Test positive case."""
        # Prepare
        self.driver.push_task_ins.return_value = driver_pb2.PushTaskInsResponse(
            task_ids=["19341fd7-62e1-4eb4-beb4-9876d3acda32"]
        )
        self.driver.pull_task_res.return_value = driver_pb2.PullTaskResResponse(
            task_res_list=[
                task_pb2.TaskRes(
                    task_id="554bd3c8-8474-4b93-a7db-c7bec1bf0012",
                    group_id="",
                    workload_id="",
                    task=task_pb2.Task(
                        legacy_client_message=ClientMessage(
                            evaluate_res=ClientMessage.EvaluateRes(
                                loss=0.0, num_examples=0
                            )
                        )
                    ),
                )
            ]
        )
        client = DriverClientProxy(
            node_id=1, driver=self.driver, anonymous=True, workload_id=""
        )
        parameters = flwr.common.Parameters(tensors=[], tensor_type="np")
        evaluate_ins: flwr.common.EvaluateIns = flwr.common.EvaluateIns(parameters, {})

        # Execute
        evaluate_res = client.evaluate(evaluate_ins, timeout=None)

        # Assert
        assert 0.0 == evaluate_res.loss
        assert 0 == evaluate_res.num_examples
