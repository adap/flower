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
"""ResNet."""


from typing import Optional, Tuple

import tensorflow as tf


# pylint: disable-msg=unused-argument
def resnet50v2(
    input_shape: Tuple[int, int, int], num_classes: int, seed: Optional[int] = None
) -> tf.keras.Model:
    """Create a ResNet-50 (v2) instance"""
    model = tf.keras.applications.ResNet50V2(
        weights=None, include_top=True, input_shape=input_shape, classes=num_classes
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
