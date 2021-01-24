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

# pylint: disable=invalid-name

from numpy import array, asarray, concatenate, expand_dims, int64
from torch.utils.data import DataLoader, Dataset

from .common import XY


def convert_torchvision_dataset_to_xy(dataset: Dataset) -> XY:
    """Converts standard Torchvision datasets into XY.

    Args:
        dataset (torch.utils.data.Dataset): Original Torchvision dataset.
            It is assumed that original elements are of type
            (<PIL.Image.Image image>, int). Elements are converted to
            (<(W,H,C), dtype=uint8>, dtype=np.int64) so they can be used with
            torchvision.transforms.

    Returns:
        XY: Dataset in the usual Tuple[ndarray, ndarray] format.
    """
    samples = []
    target = []
    for img, label in dataset:
        samples.append([array(img)])
        target.append(label)

    return concatenate(samples), expand_dims(asarray(target, dtype=int64), axis=1)
