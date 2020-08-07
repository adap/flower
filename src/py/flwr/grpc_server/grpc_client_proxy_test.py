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
from flwr.grpc_server.grpc_client_proxy import GrpcClientProxy
from flwr.proto.transport_pb2 import ClientMessage, Parameters

MESSAGE_PARAMETERS = Parameters(tensors=[], tensor_type="np")
MESSAGE_FIT_RES = ClientMessage(
    fit_res=ClientMessage.FitRes(
        parameters=MESSAGE_PARAMETERS,
        num_examples=10,
        num_examples_ceil=16,
        fit_duration=12.3,
    )
)


class GrpcClientProxyTestCase(unittest.TestCase):
    """Tests for GrpcClientProxy."""

    def setUp(self):
        """Setup mocks for tests."""
        self.bridge_mock = MagicMock()
        # Set return_value for usually blocking get_client_message method
        self.bridge_mock.request.return_value = MESSAGE_FIT_RES

    def test_get_parameters(self):
        """This test is currently quite simple and should be improved"""
        # Prepare
        client = GrpcClientProxy(cid="1", bridge=self.bridge_mock)

        # Execute
        value: flwr.common.ParametersRes = client.get_parameters()

        # Assert
        assert value.parameters.tensors == []

    def test_fit(self):
        """This test is currently quite simple and should be improved"""
        # Prepare
        client = GrpcClientProxy(cid="1", bridge=self.bridge_mock)
        parameters = flwr.common.weights_to_parameters([np.ones((2, 2))])
        ins: flwr.common.FitIns = (parameters, {})

        # Execute
        parameters_prime, num_examples, _, _ = client.fit(ins=ins)

        # Assert
        assert parameters_prime.tensor_type == "np"
        assert flwr.common.parameters_to_weights(parameters_prime) == []
        assert num_examples == 10

    def test_evaluate(self):
        """This test is currently quite simple and should be improved"""
        # Prepare
        client = GrpcClientProxy(cid="1", bridge=self.bridge_mock)
        parameters = flwr.common.Parameters(tensors=[], tensor_type="np")
        evaluate_ins: flwr.common.EvaluateIns = (parameters, {})

        # Execute
        value = client.evaluate(evaluate_ins)

        # Assert
        assert (0, 0.0, 0.0) == value
