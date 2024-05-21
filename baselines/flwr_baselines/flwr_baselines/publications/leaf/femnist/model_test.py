"""FEMNIST model test."""

# pylint: disable=no-self-use
import unittest

import torch
from hamcrest import assert_that, equal_to

from flwr_baselines.publications.leaf.femnist.model import Net


class ModelTest(unittest.TestCase):
    """Test model used for training on FEMNIST dataset."""

    def test_parameters_match(self):
        """Test if the number of parameters match the expected."""
        net = Net(num_classes=62)
        params_expected = 6_603_710
        params_in_model = sum([p.numel() for p in net.parameters()])
        assert_that(params_in_model, equal_to(params_expected))

    def test_62_outputs(self):
        """Test if the number of classes is as expected."""
        n_classes = 62
        net = Net(num_classes=n_classes)
        output = net(torch.randn(1, 1, 28, 28))
        assert_that(output.shape, equal_to(torch.Size([1, n_classes])))


if __name__ == "__main__":
    unittest.main()
