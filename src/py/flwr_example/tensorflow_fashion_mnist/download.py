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
"""Helper script to download CIFAR-10/100."""


from logging import INFO

import tensorflow as tf

from flwr.common.logger import log

tf.get_logger().setLevel("ERROR")


def main() -> None:
    """Download data."""
    log(INFO, "Download Fashion-MNIST")
    tf.keras.datasets.fashion_mnist.load_data()


if __name__ == "__main__":
    main()
