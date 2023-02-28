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
"""Tests for networked Flower client implementation."""


import unittest
from unittest.mock import MagicMock

import numpy as np

import flwr
from flwr.common.typing import Config, GetParametersIns
from flwr.proto import driver_pb2, task_pb2
from flwr.proto.transport_pb2 import ClientMessage, Parameters, Scalar
from flwr.server.grpc_server.grpc_bridge import ResWrapper

from .driver.driver_client_proxy import DriverClientProxy

MESSAGE_PARAMETERS = Parameters(tensors=[], tensor_type="np")
MESSAGE_FIT_RES = ClientMessage(
    fit_res=ClientMessage.FitRes(
        parameters=MESSAGE_PARAMETERS,
        num_examples=10,
    )
)
CLIENT_PROPERTIES = {"tensor_type": Scalar(string="numpy.ndarray")}
MESSAGE_PROPERTIES_RES = ClientMessage(
    get_properties_res=ClientMessage.GetPropertiesRes(properties=CLIENT_PROPERTIES)
)

RES_WRAPPER_FIT_RES = ResWrapper(client_message=MESSAGE_FIT_RES)
RES_WRAPPER_PROPERTIES_RES = ResWrapper(client_message=MESSAGE_PROPERTIES_RES)


TIMEOUT = 10


class DriverClientProxyTestCase(unittest.TestCase):
    """Tests for DriverClientProxy."""

    def setUp(self) -> None:
        """Setup mocks for tests."""
        self.driver = MagicMock()
        # Set return_value for usually blocking get_client_message method
        self.driver.get_nodes = driver_pb2.GetNodesResponse(node_ids=[1])
        # Set return_value for get_properties

    def test_get_parameters(self) -> None:
        """This test is currently quite simple and should be improved."""
        # Prepare
        client = DriverClientProxy(node_id=1, driver=self.driver, anonymous=True)
        get_parameters_ins = GetParametersIns(config={})

        # Execute
        value: flwr.common.GetParametersRes = client.get_parameters(
            ins=get_parameters_ins, timeout=TIMEOUT
        )

        # Assert
        assert not value.parameters.tensors

    def test_fit(self) -> None:
        """This test is currently quite simple and should be improved."""
        # Prepare
        self.driver.push_task_ins.return_value.return_value = (
            driver_pb2.PushTaskInsResponse(task_ids=[""])
        )
        self.driver.pull_task_res.return_value = driver_pb2.PullTaskResResponse(
            task_res_list=[
                task_pb2.TaskRes(
                    task_id="",
                    group_id="",
                    workload_id="",
                    task=task_pb2.Task(legacy_client_message=MESSAGE_FIT_RES),
                )
            ]
        )
        client = DriverClientProxy(node_id=1, driver=self.driver, anonymous=True)
        parameters = flwr.common.ndarrays_to_parameters([np.ones((2, 2))])
        ins: flwr.common.FitIns = flwr.common.FitIns(parameters, {})

        # Execute
        fit_res = client.fit(ins=ins, timeout=TIMEOUT)

        # Assert
        assert fit_res.parameters.tensor_type == "np"
        assert flwr.common.parameters_to_ndarrays(fit_res.parameters) == []
        assert fit_res.num_examples == 10

    def test_evaluate(self) -> None:
        """This test is currently quite simple and should be improved."""
        # Prepare
        client = DriverClientProxy(node_id=1, driver=self.driver, anonymous=True)
        parameters = flwr.common.Parameters(tensors=[], tensor_type="np")
        evaluate_ins: flwr.common.EvaluateIns = flwr.common.EvaluateIns(parameters, {})

        # Execute
        evaluate_res = client.evaluate(evaluate_ins, timeout=TIMEOUT)

        # Assert
        assert (0, 0.0) == (
            evaluate_res.num_examples,
            evaluate_res.loss,
        )

    def test_get_properties(self) -> None:
        """This test is currently quite simple and should be improved."""
        # Prepare
        self.driver.push_task_ins.return_value = driver_pb2.PushTaskInsResponse(
            task_ids=[""]
        )
        self.driver.pull_task_res.return_value = driver_pb2.PullTaskResResponse(
            task_res_list=[
                task_pb2.TaskRes(
                    task_id="",
                    group_id="",
                    workload_id="",
                    task=task_pb2.Task(legacy_client_message=MESSAGE_PROPERTIES_RES),
                )
            ]
        )
        client = DriverClientProxy(node_id=1, driver=self.driver, anonymous=True)
        request_properties: Config = {"tensor_type": "str"}
        ins: flwr.common.GetPropertiesIns = flwr.common.GetPropertiesIns(
            config=request_properties
        )

        # Execute
        value: flwr.common.GetPropertiesRes = client.get_properties(
            ins, timeout=TIMEOUT
        )

        # Assert
        assert value.properties["tensor_type"] == "numpy.ndarray"
