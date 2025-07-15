"""Test Dasha Baseline."""

import multiprocessing
import os
import unittest
from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from dasha.dataset import _load_test_dataset
from dasha.dataset_preparation import DatasetType
from dasha.main import run_parallel
from dasha.models import ClassificationModel
from dasha.tests.test_clients import DummyNetTwoParameters

TESTDATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")


def _config_env():
    if _config_env.has_been_called:
        return
    multiprocessing.set_start_method("spawn")
    torch.multiprocessing.set_sharing_strategy("file_system")
    _config_env.has_been_called = True


setattr(_config_env, "has_been_called", False)  # noqa: B010


def gradient_descent(step_size, num_rounds):
    """Run GD."""
    dummy_net = DummyNetTwoParameters([1])
    dataset = _load_test_dataset(
        OmegaConf.create(
            {
                "dataset": {
                    "type": DatasetType.TEST.value,
                }
            }
        )
    )
    features, labels = dataset[:]
    results = []
    for _ in range(num_rounds):
        dummy_net.zero_grad()
        loss = dummy_net(features, labels)
        loss.backward()
        for weight in dummy_net.parameters():
            weight.data.sub_(step_size * weight.grad)
        results.append(float(dummy_net(features, labels).detach().numpy()))
    return results


def _test_level_is_low():
    return int(os.getenv("TEST_DASHA_LEVEL", 0)) < 1


class TestDashaBaseline(unittest.TestCase):
    """Test."""

    @unittest.skipIf(_test_level_is_low(), "Flaky due to parallelism")
    def testBaseline(self) -> None:
        """Test."""
        _config_env()
        step_size = 0.1
        num_rounds = 20
        reference_results = gradient_descent(step_size, num_rounds)

        cfg = OmegaConf.create(
            {
                "local_address": None,
                "dataset": {
                    "type": DatasetType.TEST.value,
                },
                "num_clients": 2,
                "num_rounds": num_rounds,
                "compressor": {
                    "_target_": "dasha.compressors.IdentityUnbiasedCompressor",
                },
                "model": {
                    "_target_": "dasha.tests.test_clients.DummyNetTwoParameters",
                },
                "method": {
                    "strategy": {
                        "_target_": "dasha.strategy.DashaAggregator",
                        "step_size": step_size,
                    },
                    "client": {"_target_": "dasha.client.DashaClient", "device": "cpu"},
                },
            }
        )
        results = run_parallel(cfg)
        losses = [loss for (_, loss) in results.losses_distributed]
        # TODO: Maybe fix it. I don't know in which round
        # Flower will start training in advance,
        # so I check different subarrays for equality.
        self.assertTrue(
            np.any(
                [
                    np.allclose(reference_results[: len(losses) - i], losses[i:])
                    for i in range(20)
                ]
            )
        )


class TestDashaBaselineWithRandK(unittest.TestCase):
    """Test."""

    @unittest.skipIf(_test_level_is_low(), "Flaky due to parallelism")
    def testBaseline(self) -> None:
        """Test."""
        _config_env()
        step_size = 0.01
        num_rounds = 100

        cfg = OmegaConf.create(
            {
                "local_address": None,
                "dataset": {
                    "type": DatasetType.TEST.value,
                },
                "num_clients": 2,
                "num_rounds": num_rounds,
                "model": {
                    "_target_": "dasha.tests.test_clients.DummyNetTwoParameters",
                },
                "compressor": {
                    "_target_": "dasha.compressors.RandKCompressor",
                    "number_of_coordinates": 1,
                },
                "method": {
                    "strategy": {
                        "_target_": "dasha.strategy.DashaAggregator",
                        "step_size": step_size,
                    },
                    "client": {"_target_": "dasha.client.DashaClient", "device": "cpu"},
                },
            }
        )
        results = run_parallel(cfg)
        losses = [loss for (_, loss) in results.losses_distributed]
        self.assertGreater(losses[0], 2.0)
        self.assertLess(losses[-1], 1.0 + 1e-5)


class ClassificationDummyNet(ClassificationModel):
    """Dummy Net."""

    def __init__(self, input_shape: List[int]) -> None:
        super().__init__(input_shape)
        self._bias = nn.Parameter(torch.Tensor([0]))
        self._loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self._loss(features + self._bias, targets)


class TestMomentumHelpsInStochasticDashaBaselineWithRandK(unittest.TestCase):
    """Test."""

    @unittest.skipIf(_test_level_is_low(), "Flaky due to parallelism")
    def testBaseline(self) -> None:
        """Test."""
        _config_env()
        step_size = 1.0
        num_rounds = 1000

        params: Any = {
            "local_address": None,
            "dataset": {
                "type": DatasetType.RANDOM_TEST.value,
            },
            "num_clients": 2,
            "num_rounds": num_rounds,
            "model": {
                "_target_": "dasha.tests.test_dasha_baseline.ClassificationDummyNet",
            },
            "compressor": {
                "_target_": "dasha.compressors.RandKCompressor",
                "number_of_coordinates": 1,
            },
            "method": {
                "strategy": {
                    "_target_": "dasha.strategy.DashaAggregator",
                    "step_size": step_size,
                },
                "client": {
                    "_target_": "dasha.client.StochasticDashaClient",
                    "device": "cpu",
                    "evaluate_full_dataset": True,
                    "stochastic_momentum": None,
                    "mega_batch_size": 10,
                },
            },
        }

        mean_loss = []
        for stochastic_momentum in [0.01, 0.1, 1.0]:
            params["method"]["client"]["stochastic_momentum"] = stochastic_momentum
            cfg = OmegaConf.create(params)
            results = run_parallel(cfg)
            losses = [loss for (_, loss) in results.losses_distributed]
            mean_loss.append(np.mean(losses[-100:]))
        self.assertLess(mean_loss[0], mean_loss[1])
        self.assertLess(mean_loss[1], mean_loss[2])


class TestMegaBatchHelpsInStochasticMarinaBaselineWithRandK(unittest.TestCase):
    """Test."""

    @unittest.skipIf(_test_level_is_low(), "Flaky due to parallelism")
    def testBaseline(self) -> None:
        """Test."""
        _config_env()
        step_size = 1.0
        num_rounds = 1000

        params: Any = {
            "local_address": None,
            "dataset": {
                "type": DatasetType.RANDOM_TEST.value,
            },
            "num_clients": 2,
            "num_rounds": num_rounds,
            "model": {
                "_target_": "dasha.tests.test_dasha_baseline.ClassificationDummyNet",
            },
            "compressor": {
                "_target_": "dasha.compressors.RandKCompressor",
                "number_of_coordinates": 1,
            },
            "method": {
                "strategy": {
                    "_target_": "dasha.strategy.MarinaAggregator",
                    "step_size": step_size,
                },
                "client": {
                    "_target_": "dasha.client.StochasticMarinaClient",
                    "device": "cpu",
                    "evaluate_full_dataset": True,
                    "mega_batch_size": None,
                },
            },
        }

        mean_loss = []
        for mega_batch_size in [10, 1]:
            params["method"]["client"]["mega_batch_size"] = mega_batch_size
            cfg = OmegaConf.create(params)
            results = run_parallel(cfg)
            losses = [loss for (_, loss) in results.losses_distributed]
            mean_loss.append(np.mean(losses[-100:]))
        self.assertLess(mean_loss[0], mean_loss[1])


class TestMarinaBaselineWithRandK(unittest.TestCase):
    """Test."""

    @unittest.skipIf(_test_level_is_low(), "Flaky due to parallelism")
    def testBaseline(self) -> None:
        """Test."""
        _config_env()
        step_size = 0.01
        num_rounds = 100
        number_of_coordinates = 1

        cfg = OmegaConf.create(
            {
                "local_address": None,
                "dataset": {
                    "type": DatasetType.TEST.value,
                },
                "num_clients": 2,
                "num_rounds": num_rounds,
                "model": {
                    "_target_": "dasha.tests.test_clients.DummyNetTwoParameters",
                },
                "compressor": {
                    "_target_": "dasha.compressors.RandKCompressor",
                    "number_of_coordinates": number_of_coordinates,
                },
                "method": {
                    "strategy": {
                        "_target_": "dasha.strategy.MarinaAggregator",
                        "step_size": step_size,
                    },
                    "client": {
                        "_target_": "dasha.client.MarinaClient",
                        "device": "cpu",
                    },
                },
            }
        )
        results = run_parallel(cfg)
        losses = [loss for (_, loss) in results.losses_distributed]
        self.assertGreater(losses[0], 2.0)
        self.assertLess(losses[-1], 1.0 + 1e-5)


if __name__ == "__main__":
    unittest.main()
