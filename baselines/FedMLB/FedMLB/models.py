"""Define custom models being used."""

from typing import Optional

import tensorflow as tf

BATCH_NORM_DECAY = 0.997
BATCH_NORM_EPSILON = 1e-5
LAYER_NORM_EPSILON = 1e-5
GROUP_NORM_EPSILON = 1e-5


def get_norm_layer(norm, channel_axis):
    """Return the requested norm layer."""
    if norm == "batch":
        return tf.keras.layers.BatchNormalization(
            axis=channel_axis, momentum=BATCH_NORM_DECAY
        )

    if norm == "layer":
        return tf.keras.layers.LayerNormalization(
            axis=channel_axis, epsilon=LAYER_NORM_EPSILON
        )

    return tf.keras.layers.GroupNormalization(groups=2, epsilon=GROUP_NORM_EPSILON)


class ResBlock(tf.keras.Model):
    """Implement a ResBlock."""

    def __init__(
        self,
        filters,
        downsample,
        norm="group",
        is_first_layer=False,
        l2_weight_decay=1e-3,
        stride=1,
        seed: Optional[int] = None,
    ):
        super().__init__()

        if tf.keras.backend.image_data_format() == "channels_last":
            channel_axis = 3
        else:
            channel_axis = 1

        self.conv1 = tf.keras.layers.Conv2D(
            filters,
            kernel_size=(3, 3),
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
            kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
        )

        if downsample:
            self.shortcut = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(
                        filters,
                        kernel_size=(1, 1),
                        strides=stride,
                        padding="valid",
                        use_bias=False,
                        kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                        kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
                    ),
                    get_norm_layer(norm, channel_axis=channel_axis),
                ]
            )
        else:
            self.shortcut = tf.keras.Sequential()

        self.gn1 = get_norm_layer(norm, channel_axis=channel_axis)

        self.conv2 = tf.keras.layers.Conv2D(
            filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
            kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
        )
        self.gn2 = get_norm_layer(norm, channel_axis=channel_axis)

    def call(self, input):
        """Call the model on new inputs and returns the outputs as tensors."""
        shortcut = self.shortcut(input)

        input = self.conv1(input)
        input = self.gn1(input)
        input = tf.keras.layers.ReLU()(input)

        input = self.conv2(input)
        input = self.gn2(input)

        input = input + shortcut
        return tf.keras.layers.ReLU()(input)


class ResNet18(tf.keras.Model):
    """Implement a ResNet18 architecture as in FedMLB paper."""

    def __init__(
        self, outputs=10, l2_weight_decay=1e-3, seed: Optional[int] = None, norm=""
    ):
        super().__init__()
        if seed is not None:
            tf.random.set_seed(seed)
        if tf.keras.backend.image_data_format() == "channels_last":
            channel_axis = 3
        else:
            channel_axis = 1

        self.layer0 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    64,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                    use_bias=False,
                    kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                    kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
                ),
                get_norm_layer(norm, channel_axis=channel_axis),
                tf.keras.layers.ReLU(),
            ],
            name="layer0",
        )

        self.layer1 = tf.keras.Sequential(
            [
                ResBlock(
                    64,
                    downsample=False,
                    l2_weight_decay=l2_weight_decay,
                    stride=1,
                    norm=norm,
                ),
                ResBlock(
                    64,
                    downsample=False,
                    l2_weight_decay=l2_weight_decay,
                    stride=1,
                    norm=norm,
                ),
            ],
            name="layer1",
        )

        self.layer2 = tf.keras.Sequential(
            [
                ResBlock(
                    128,
                    downsample=True,
                    l2_weight_decay=l2_weight_decay,
                    stride=2,
                    norm=norm,
                ),
                ResBlock(
                    128,
                    downsample=False,
                    l2_weight_decay=l2_weight_decay,
                    stride=1,
                    norm=norm,
                ),
            ],
            name="layer2",
        )

        self.layer3 = tf.keras.Sequential(
            [
                ResBlock(
                    256,
                    downsample=True,
                    l2_weight_decay=l2_weight_decay,
                    stride=2,
                    norm=norm,
                ),
                ResBlock(
                    256,
                    downsample=False,
                    l2_weight_decay=l2_weight_decay,
                    stride=1,
                    norm=norm,
                ),
            ],
            name="layer3",
        )

        self.layer4 = tf.keras.Sequential(
            [
                ResBlock(
                    512,
                    downsample=True,
                    l2_weight_decay=l2_weight_decay,
                    stride=2,
                    norm=norm,
                ),
                ResBlock(
                    512,
                    downsample=False,
                    l2_weight_decay=l2_weight_decay,
                    stride=1,
                    norm=norm,
                ),
            ],
            name="layer4",
        )

        self.gap = tf.keras.Sequential(
            [tf.keras.layers.GlobalAveragePooling2D()], name="gn_relu_gap"
        )
        self.fc = tf.keras.layers.Dense(
            outputs,
            kernel_initializer=tf.keras.initializers.RandomNormal(
                stddev=0.01, seed=seed
            ),
            kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
            bias_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
        )

    def call(self, input):
        """Call the model on new inputs and returns the outputs as tensors."""
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = self.fc(input)

        return input


class ResNet18MLB(tf.keras.Model):
    """Implement a custom ResNet18 architecture as in FedMLB paper."""

    def __init__(
        self, outputs=10, l2_weight_decay=1e-3, seed: Optional[int] = None, norm=""
    ):
        super().__init__()
        if seed is not None:
            tf.random.set_seed(seed)
        if tf.keras.backend.image_data_format() == "channels_last":
            channel_axis = 3
        else:
            channel_axis = 1

        self.layer0 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    64,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                    use_bias=False,
                    kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                    kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
                ),
                get_norm_layer(norm, channel_axis=channel_axis),
                tf.keras.layers.ReLU(),
            ],
            name="layer0",
        )

        self.layer1 = tf.keras.Sequential(
            [
                ResBlock(
                    64,
                    downsample=False,
                    l2_weight_decay=l2_weight_decay,
                    stride=1,
                    norm=norm,
                ),
                ResBlock(
                    64,
                    downsample=False,
                    l2_weight_decay=l2_weight_decay,
                    stride=1,
                    norm=norm,
                ),
            ],
            name="layer1",
        )

        self.layer2 = tf.keras.Sequential(
            [
                ResBlock(
                    128,
                    downsample=True,
                    l2_weight_decay=l2_weight_decay,
                    stride=2,
                    norm=norm,
                ),
                ResBlock(
                    128,
                    downsample=False,
                    l2_weight_decay=l2_weight_decay,
                    stride=1,
                    norm=norm,
                ),
            ],
            name="layer2",
        )

        self.layer3 = tf.keras.Sequential(
            [
                ResBlock(
                    256,
                    downsample=True,
                    l2_weight_decay=l2_weight_decay,
                    stride=2,
                    norm=norm,
                ),
                ResBlock(
                    256,
                    downsample=False,
                    l2_weight_decay=l2_weight_decay,
                    stride=1,
                    norm=norm,
                ),
            ],
            name="layer3",
        )

        self.layer4 = tf.keras.Sequential(
            [
                ResBlock(
                    512,
                    downsample=True,
                    l2_weight_decay=l2_weight_decay,
                    stride=2,
                    norm=norm,
                ),
                ResBlock(
                    512,
                    downsample=False,
                    l2_weight_decay=l2_weight_decay,
                    stride=1,
                    norm=norm,
                ),
            ],
            name="layer4",
        )

        self.gap = tf.keras.Sequential(
            [tf.keras.layers.GlobalAveragePooling2D()], name="gn_relu_gap"
        )
        self.fc = tf.keras.layers.Dense(
            outputs,
            kernel_initializer=tf.keras.initializers.RandomNormal(
                stddev=0.01, seed=seed
            ),
            kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
            bias_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
        )

    def call(self, input_X, return_feature=False, level=0):
        """Call the model on new inputs and returns the outputs as tensors."""
        if level <= 0:
            out0 = self.layer0(input_X)
        else:
            out0 = input_X
        if level <= 1:
            out1 = self.layer1(out0)
        else:
            out1 = out0
        if level <= 2:
            out2 = self.layer2(out1)
        else:
            out2 = out1
        if level <= 3:
            out3 = self.layer3(out2)
        else:
            out3 = out2
        if level <= 4:
            out4 = self.layer4(out3)
            out4 = self.gap(out4)
        else:
            out4 = out3

        logit = self.fc(out4)

        if return_feature:
            return out0, out1, out2, out3, out4, logit
        else:
            return logit


def create_resnet18(
    num_classes=100,
    input_shape=(None, 32, 32, 3),
    norm="group",
    l2_weight_decay=0.0,
    seed: Optional[int] = None,
):
    """Return a built ResNet model."""
    resnet18 = ResNet18(
        outputs=num_classes, l2_weight_decay=l2_weight_decay, seed=seed, norm=norm
    )
    resnet18.build(input_shape)
    return resnet18


def create_resnet18_mlb(
    num_classes=100,
    input_shape=(None, 32, 32, 3),
    norm="group",
    l2_weight_decay=0.0,
    seed: Optional[int] = None,
):
    """Return a built ResNetMLB model."""
    resnet18 = ResNet18MLB(
        outputs=num_classes, l2_weight_decay=l2_weight_decay, seed=seed, norm=norm
    )
    resnet18.build(input_shape)
    return resnet18
