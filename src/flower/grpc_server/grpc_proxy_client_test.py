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

import flower
from flower.grpc_server.grpc_proxy_client import GRPCProxyClient
from flower.proto.transport_pb2 import ClientMessage, Parameters

MESSAGE_PARAMETERS = Parameters(tensors=[], tensor_type="np")
MESSAGE_FIT_RES = ClientMessage(
    fit_res=ClientMessage.FitRes(parameters=MESSAGE_PARAMETERS, num_examples=10)
)


class GRPCProxyClientTestCase(unittest.TestCase):
    """Tests for GRPCProxyClient."""

    def setUp(self):
        """Setup mocks for tests."""
        self.bridge_mock = MagicMock()
        # Set return_value for usually blocking get_client_message method
        self.bridge_mock.request.return_value = MESSAGE_FIT_RES

    def test_get_parameters(self):
        """This test is currently quite simple and should be improved"""
        # Prepare
        client = GRPCProxyClient(cid="1", bridge=self.bridge_mock)

        # Execute
        value: flower.ParametersRes = client.get_parameters()

        # Assert
        assert value.parameters.tensors == []

    def test_fit(self):
        """This test is currently quite simple and should be improved"""
        # Prepare
        client = GRPCProxyClient(cid="1", bridge=self.bridge_mock)
        parameters = flower.weights_to_parameters([np.ones((2, 2))])
        ins: flower.FitIns = (parameters, {})

        # Execute
        parameters_prime, num_examples = client.fit(ins=ins)

        # Assert
        assert parameters_prime.tensor_type == "np"
        assert flower.parameters_to_weights(parameters_prime) == []
        assert num_examples == 10

    def test_evaluate(self):
        """This test is currently quite simple and should be improved"""
        # Prepare
        client = GRPCProxyClient(cid="1", bridge=self.bridge_mock)
        parameters = flower.Parameters(tensors=[], tensor_type="np")
        evaluate_ins: flower.EvaluateIns = (parameters, {})

        # Execute
        value = client.evaluate(evaluate_ins)

        # Assert
        assert (0, 0.0) == value
