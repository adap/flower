"""Test Clients."""

import unittest
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils

from dasha.client import DashaClient
from dasha.compressors import decompress
from dasha.models import ClassificationModel

_CPU_DEVICE = "cpu"


class DummyNet(ClassificationModel):
    """Dummy Net."""

    def __init__(self, input_shape: List[int]) -> None:
        super().__init__(input_shape)
        self._weight = nn.Parameter(torch.Tensor([2]))

    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return 0.5 * torch.mean((self._weight.view(-1, 1) * features - targets) ** 2)


class TestDashaClient(unittest.TestCase):
    """Test."""

    def setUp(self) -> None:
        """Init."""
        self._function = DummyNet([1])
        self._features = [[1], [2]]
        self._targets = [[1], [2]]
        dataset = data_utils.TensorDataset(
            torch.Tensor(self._features), torch.Tensor(self._targets)
        )
        self._client = DashaClient(
            function=self._function, dataset=dataset, device=_CPU_DEVICE
        )

    def testGetParameters(self) -> None:
        """Test."""
        parameters = self._client.get_parameters(config={})
        self.assertEqual(len(parameters), 1)
        self.assertAlmostEqual(float(parameters[0]), 2)

    def testSetParameters(self) -> None:
        """Test."""
        parameter = 3.0
        parameters = [np.array([parameter])]
        self._client._set_parameters(parameters)
        self.assertAlmostEqual(
            float(self._function._weight.detach().numpy()), parameter
        )

    def testEvaluate(self) -> None:
        """Test."""
        parameter = 3.0
        parameters_list = [np.array([parameter])]
        loss, num_samples, _ = self._client.evaluate(parameters_list, config={})
        self.assertEqual(num_samples, 2)
        loss_actual = sum(
            [
                0.5 * (parameter * self._features[i][0] - self._targets[i][0]) ** 2
                for i in range(len(self._targets))
            ]
        ) / len(self._targets)
        self.assertAlmostEqual(float(loss), loss_actual)

    def testFit(self) -> None:
        """Test."""
        parameter = 3.0
        parameters_list = [np.array([parameter])]
        gradients, num_samples, _ = self._client.fit(
            parameters_list, config={self._client.SEND_FULL_GRADIENT: True}
        )
        self.assertEqual(num_samples, 2)
        gradients = decompress(gradients)
        gradient_actual = sum(
            [
                self._features[i][0]
                * (parameter * self._features[i][0] - self._targets[i][0])
                for i in range(len(self._targets))
            ]
        ) / len(self._targets)
        self.assertAlmostEqual(float(gradients[0]), gradient_actual)


class DummyNetTwoParameters(ClassificationModel):
    """Dummy Net."""

    def __init__(self, input_shape: List[int]) -> None:
        super().__init__(input_shape)
        self._weight_one = nn.Parameter(torch.Tensor([1]))
        self._weight_two = nn.Parameter(torch.Tensor([3]))

    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return (
            0.5
            * torch.mean(
                (
                    self._weight_two.view(-1, 1)
                    * self._weight_one.view(-1, 1)
                    * features
                    - targets
                )
                ** 2
            )
            + 1.0
        )


class TestDashaClientWithTwoParameters(unittest.TestCase):
    """Test."""

    def setUp(self) -> None:
        """Init."""
        self._function = DummyNetTwoParameters([1])
        self._features = [[1], [2]]
        self._targets = [[1], [2]]
        dataset = data_utils.TensorDataset(
            torch.Tensor(self._features), torch.Tensor(self._targets)
        )
        self._client = DashaClient(
            function=self._function, dataset=dataset, device=_CPU_DEVICE
        )

    def testGetParameters(self) -> None:
        """Test."""
        parameters = self._client.get_parameters(config={})
        self.assertEqual(len(parameters), 1)
        self.assertAlmostEqual(float(parameters[0][0]), 1)
        self.assertAlmostEqual(float(parameters[0][1]), 3)

    def testSetParameters(self) -> None:
        """Test."""
        parameter = [3.0, 10.0]
        parameters = [np.array(parameter)]
        self._client._set_parameters(parameters)
        self.assertAlmostEqual(
            float(self._function._weight_one.detach().numpy()), parameter[0]
        )
        self.assertAlmostEqual(
            float(self._function._weight_two.detach().numpy()), parameter[1]
        )


if __name__ == "__main__":
    unittest.main()
