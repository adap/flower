import unittest

import torch
from femnist.model import Net
from hamcrest import assert_that, equal_to


class ModelTest(unittest.TestCase):
    def test_parameters_match(self):
        net = Net(num_classes=62)
        params_expected = 6_603_710
        params_in_model = sum([p.numel() for p in net.parameters()])
        assert_that(params_in_model, equal_to(params_expected))

    def test_62_outputs(self):
        n_classes = 62
        net = Net(num_classes=n_classes)
        output = net(torch.randn(1, 1, 28, 28))
        assert_that(output.shape, equal_to(torch.Size([1, n_classes])))


if __name__ == "__main__":
    unittest.main()
