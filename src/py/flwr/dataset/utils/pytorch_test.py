# Copyright 2021 Adap GmbH. All Rights Reserved.
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
"""Test cases for Pytorch related functions."""
# pylint: disable=invalid-name

import unittest
from os import mkdir
from os.path import join
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from .common import XY
from .pytorch import convert_torchvision_dataset_to_xy


class PytorchUtilFunctions(unittest.TestCase):
    def test_convert_torchvision_dataset_to_xy(self) -> None:
        """Test convert_torchvision_dataset_to_xy function."""
        # Prepare
        dataset = None
        list_classes = ["cat", "dog"]
        list_np_images = []
        list_pil_images = []
        for idx, label in enumerate(list_classes):
            # Create image array
            tmp_np_img = idx * np.ones((32, 32, 3), dtype=np.uint8)
            list_np_images.append(tmp_np_img)

            # Converts to PIL
            tmp_pil_img = Image.fromarray(tmp_np_img)
            list_pil_images.append(tmp_pil_img)

        # Save images and load with DatasetFolder
        with TemporaryDirectory() as tmpdirname:
            for idx, image_class in enumerate(list_classes):
                class_dir: str = join(tmpdirname, image_class)
                mkdir(class_dir)
                img_path: str = join(class_dir, "0.png")
                list_pil_images[idx].save(img_path)

            dataset = ImageFolder(tmpdirname)

            # Execute
            X, Y = convert_torchvision_dataset_to_xy(dataset)

            # Assert
            for idx, np_img in enumerate(list_np_images):
                dataset_img, dataset_label = dataset[idx]
                dataset_np = np.array(dataset_img)

                x, y = X[idx], Y[idx]

                np.testing.assert_array_equal(dataset_np, x)
                assert dataset_label == y


if __name__ == "__main__":
    unittest.main(verbosity=2)
