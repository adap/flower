"""Test Dataset."""

import os
import unittest

import numpy as np
from omegaconf import OmegaConf

from dasha.dataset import LIBSVMDatasetName, load_dataset, random_split
from dasha.dataset_preparation import DatasetType

TESTDATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
cfg = OmegaConf.create(
    {
        "dataset": {
            "type": DatasetType.LIBSVM.value,
            "path_to_dataset": TESTDATA_PATH,
            "dataset_name": LIBSVMDatasetName.MUSHROOMS.value,
        }
    }
)


class TestMushroomsDataset(unittest.TestCase):
    """Test."""

    def testLoad(self) -> None:
        """Test."""
        dataset = load_dataset(cfg)
        features, labels = dataset[:]
        self.assertEqual(np.sort(np.unique(labels.numpy())).tolist(), [0, 1])
        self.assertEqual(list(features.shape), [8124, 112])

    def testSplit(self) -> None:
        """Test."""
        dataset = load_dataset(cfg)
        datasets = random_split(dataset, num_clients=5)
        self.assertEqual(sum([len(d) for d in datasets]), 8124)


if __name__ == "__main__":
    unittest.main()
