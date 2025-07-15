"""Test Models."""

import unittest

import torch

from dasha.models import LinearNetWithNonConvexLoss, ResNet18WithLogisticLoss


class TestSmokeLinearNetWithNonConvexLoss(unittest.TestCase):
    """Test."""

    def test(self) -> None:
        """Test."""
        features = torch.rand(3, 42)
        targets = torch.Tensor([1, -1, 1])
        model = LinearNetWithNonConvexLoss([42])
        loss = model(features, targets)
        loss.backward()
        parameters = list(model.parameters())
        self.assertEqual(len(parameters), 2)
        self.assertTrue(parameters[0].grad is not None)
        accuracy = model.accuracy(features, targets)
        self.assertTrue(accuracy >= 0 - 1e-2 and accuracy <= 1.0 + 1e-2)


class TestSmokeResNet18WithLogisticLoss(unittest.TestCase):
    """Test."""

    def test(self) -> None:
        """Test."""
        features = torch.rand(3, 3, 32, 32)
        targets = torch.Tensor([0, 8, 6])
        model = ResNet18WithLogisticLoss(features[0].shape)
        loss = model(features, targets)
        loss.backward()
        model.accuracy(features, targets)


if __name__ == "__main__":
    unittest.main()
