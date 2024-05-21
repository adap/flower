"""Handle dataset loading and preprocessing utility."""

import os
from typing import Union

import numpy as np
import tensorflow as tf


def load_selected_client_statistics(
    selected_client: int, alpha: float, dataset: str, total_clients: int
):
    """Return the amount of local examples for the selected client.

    Clients are referenced with a client_id. Loads a numpy array saved on disk. This
    could be done directly by doing len(ds.to_list()) but it's more expensive at run
    time.
    """
    path = os.path.join(
        dataset + "_mlb_dirichlet_train",
        str(total_clients),
        str(round(alpha, 2)),
        "distribution_train.npy",
    )
    smpls_loaded = np.load(path)
    local_examples_all_clients = np.sum(smpls_loaded, axis=1)
    return local_examples_all_clients[selected_client]


# pylint: disable=W0221
class PaddedRandomCropCustom(tf.keras.layers.Layer):
    """Custom keras layer to random crop the input image, same as FedMLB paper."""

    def __init__(
        self, seed: Union[int, None] = None, height: int = 32, width: int = 32, **kwargs
    ):
        super().__init__(**kwargs)
        self.seed = seed
        self.height = height
        self.width = width

    def call(self, inputs: tf.Tensor, training: bool = True):
        """Call the layer on new inputs and returns the outputs as tensors."""
        if training:
            inputs = tf.image.resize_with_crop_or_pad(
                image=inputs,
                target_height=self.height + 4,
                target_width=self.width + 4,
            )
            inputs = tf.image.random_crop(
                value=inputs, size=[self.height, self.width, 3], seed=self.seed
            )

            return inputs
        return inputs


# pylint: disable=W0221
class PaddedCenterCropCustom(tf.keras.layers.Layer):
    """Custom keras layer to center crop the input image, same as FedMLB paper."""

    def __init__(self, height: int = 64, width: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.height = height
        self.width = width

    def call(self, inputs: tf.Tensor):
        """Call the layer on new inputs and returns the outputs as tensors."""
        input_tensor = tf.image.resize_with_crop_or_pad(
            image=inputs, target_height=self.height, target_width=self.width
        )

        input_shape = tf.shape(inputs)
        h_diff = input_shape[0] - self.height
        w_diff = input_shape[1] - self.width

        h_start = tf.cast(h_diff / 2, tf.int32)
        w_start = tf.cast(w_diff / 2, tf.int32)
        return tf.image.crop_to_bounding_box(
            input_tensor, h_start, w_start, self.height, self.width
        )


def load_client_datasets_from_files(  # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    dataset: str,
    sampled_client: int,
    batch_size: int,
    total_clients: int = 100,
    alpha: float = 0.3,
    split: str = "train",
    seed: Union[int, None] = None,
):
    """Load the partition of the dataset for the sampled client.

    Sampled client represented by its client_id. Examples are preprocessed via
    normalization layer. Returns a batched dataset.
    """

    def element_fn_norm_cifar100(image, label):
        """Normalize cifar100 images."""
        norm_layer = tf.keras.layers.Normalization(
            mean=[0.5071, 0.4865, 0.4409],
            variance=[np.square(0.2673), np.square(0.2564), np.square(0.2762)],
        )
        return norm_layer(tf.cast(image, tf.float32) / 255.0), label

    def element_fn_norm_tiny_imagenet(image, label):
        """Normalize tiny-imagenet images."""
        norm_layer = tf.keras.layers.Normalization(
            mean=[0.4802, 0.4481, 0.3975],
            variance=[np.square(0.2770), np.square(0.2691), np.square(0.2821)],
        )
        return norm_layer(tf.cast(image, tf.float32) / 255.0), tf.expand_dims(
            label, axis=-1
        )

    # transform images
    rotate = tf.keras.layers.RandomRotation(0.028, seed=seed)
    flip = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

    if dataset in ["cifar100"]:
        crop = PaddedRandomCropCustom(seed=seed)
    else:  # tiny-imagenet
        crop = PaddedRandomCropCustom(seed=seed, height=64, width=64)

    rotate_flip_crop = tf.keras.Sequential(
        [
            rotate,
            crop,
            flip,
        ]
    )

    center_crop = tf.keras.layers.CenterCrop(64, 64)

    def center_crop_data(image, label):
        """Crop images."""
        return center_crop(image), label

    def transform_data(image, label):
        """Transform images."""
        return rotate_flip_crop(image), label

    path = os.path.join(
        dataset + "_mlb_dirichlet_train",
        str(total_clients),
        str(round(alpha, 2)),
        split,
    )

    loaded_ds = tf.data.Dataset.load(
        path=os.path.join(path, str(sampled_client)),
        element_spec=None,
        compression=None,
        reader_func=None,
    )

    if dataset in ["cifar100"]:
        if split == "test":
            return loaded_ds.map(element_fn_norm_cifar100).batch(
                batch_size, drop_remainder=False
            )
        loaded_ds = (
            loaded_ds.shuffle(buffer_size=1024, seed=seed)
            .map(element_fn_norm_cifar100)
            .map(transform_data)
            .batch(batch_size, drop_remainder=False)
        )
        loaded_ds = loaded_ds.prefetch(tf.data.AUTOTUNE)
        return loaded_ds
    # dataset in ["tiny_imagenet"]
    if split == "test":
        return (
            loaded_ds.map(element_fn_norm_tiny_imagenet)
            .map(center_crop_data)
            .batch(batch_size, drop_remainder=False)
        )
    loaded_ds = (
        loaded_ds.shuffle(buffer_size=1024, seed=seed)
        .map(element_fn_norm_tiny_imagenet)
        .map(transform_data)
        .batch(batch_size, drop_remainder=False)
    )
    loaded_ds = loaded_ds.prefetch(tf.data.AUTOTUNE)
    return loaded_ds
