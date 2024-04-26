"""Required imports for data.py script."""

import itertools
import time

import numpy as np
import tensorflow as tf


class History(tf.keras.callbacks.History):
    """A custom implementation of the Keras History callback for logging history.

    This class extends tf.keras.callbacks.History to provide custom functionalities for
    tracking the training process. It captures and stores metrics at the end of each
    epoch.
    """

    def __init__(self):
        super().__init__()
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        """Handle the end of an epoch during model training.

        Updates the history attribute with information about the completed epoch,
        including any provided logs such as loss and accuracy metrics.

        Parameters
        ----------
        - epoch (int): The index of the recently completed epoch.
        - logs (dict, optional): A dictionary containing metrics and their values
        for the completed epoch.
        """
        logs = logs or {}
        self.epoch.append(epoch)
        for key, value in logs.items():
            self.history.setdefault(key, []).append(value)


class Network:
    """A network class for building a custom audio classification model.

    This class constructs a convolutional neural network for audio data classification.
    It includes methods to define convolutional blocks and assemble the full model.

    Attributes
    ----------
    - num_classes (int): Number of classes for classification.
    - dropout_rate (float): Dropout rate for regularization.
    - l2_rate (float): L2 regularization rate.
    - input_shape (tuple): Shape of the input data.
    - sequence_length (int): Length of the input sequences.
    """

    # pylint: disable=too-many-arguments, too-many-locals
    def __init__(
        self,
        num_classes,
        dropout_rate=0.1,
        l2_rate=0.0001,
        input_shape=(None, 64, 1),
        sequence_length=16000,
    ):
        self._num_classes = num_classes
        self._dropout_rate = dropout_rate
        self._l2_rate = l2_rate
        self._input_shape = input_shape
        self._sequence_length = sequence_length

    @staticmethod
    def conv_block(
        input_shape, num_features, l2_rate, dropout_rate=0.1, add_max_pool=True
    ):
        """Define a convolutional block for the neural network.

        Constructs a block with convolutional layers, normalization, activation,
        and optional pooling and dropout, based on the specified parameters.

        Parameters
        ----------
        - input_shape (tuple): Shape of the input to the convolutional block.
        - num_features (int): Number of filters in the convolutional layers.
        - l2_rate (float): L2 regularization rate.
        - dropout_rate (float, optional): Dropout rate for regularization.
        - add_max_pool (bool, optional): Whether to include max pooling in the block.

        Returns
        -------
        - tf.keras.Model: A Keras model representing the convolutional block.
        """
        inputs = tf.keras.layers.Input(shape=input_shape)
        x_t = tf.keras.layers.Conv2D(
            num_features,
            (1, 4),
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(l2_rate),
        )(inputs)
        x_t = tf.keras.layers.GroupNormalization(groups=4)(x_t)
        x_t = tf.keras.layers.Activation("relu")(x_t)
        x_f = tf.keras.layers.Conv2D(
            num_features,
            (4, 1),
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(l2_rate),
        )(inputs)
        x_f = tf.keras.layers.GroupNormalization(groups=4)(x_f)
        x_f = tf.keras.layers.Activation("relu")(x_f)
        x = tf.keras.layers.Concatenate(axis=-1)([x_t, x_f])
        x = tf.keras.layers.Conv2D(
            num_features,
            (1, 1),
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(l2_rate),
        )(x)
        x = tf.keras.layers.GroupNormalization(groups=4)(x)
        x = tf.keras.layers.Activation("relu")(x)
        if add_max_pool:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.SpatialDropout2D(dropout_rate)(x)
        return tf.keras.Model(inputs, x)

    def get_network(self):
        """Assembles and returns the custom audio classification model.

        Constructs the neural network layer by layer using the class attributes and
        predefined convolutional blocks.

        Returns
        -------
        - tf.keras.Model: The assembled TensorFlow Keras model
          ready for training or inference.
        """
        inputs = tf.keras.layers.Input(shape=self._input_shape, name="audio_input")
        x = __class__.conv_block(
            self._input_shape, 24, self._l2_rate, dropout_rate=self._dropout_rate
        )(inputs)
        x = __class__.conv_block(
            x.shape[1:], 32, self._l2_rate, dropout_rate=self._dropout_rate
        )(x)
        x = __class__.conv_block(
            x.shape[1:], 64, self._l2_rate, dropout_rate=self._dropout_rate
        )(x)
        x = __class__.conv_block(
            x.shape[1:], 128, self._l2_rate, dropout_rate=self._dropout_rate
        )(x)
        x = tf.keras.layers.GlobalMaxPool2D()(x)
        outputs = tf.keras.layers.Dense(self._num_classes)(x)
        model = tf.keras.Model(inputs, outputs, name="audio_classifier")
        return model

    def get_evaluation_network(self, input_shape=(None, None, 64, 1)):
        """Create an evaluation model for the network.

        Constructs a model for evaluation which applies the primary network to a
        time-distributed input and computes the mean output.

        Parameters
        ----------
        - input_shape (tuple, optional):
          Shape of the input data for the evaluation model.

        Returns
        -------
        - tf.keras.Model: The evaluation model based on the primary network.
        """
        model = self.get_network()
        input_layer = tf.keras.layers.Input(shape=input_shape, name="audio_input")
        output = tf.reduce_mean(
            tf.keras.layers.TimeDistributed(model)(input_layer), axis=1
        )
        return tf.keras.Model(input_layer, output)

    @staticmethod
    def get_init_weights(num_classes, input_shape=(None, 64, 1), l2_rate=0.0001):
        """Generate and return initial weights for the network.

        Constructs a temporary model for the sole purpose of initializing weights,
        based on the specified number of classes, input shape, and regularization rate.

        Parameters
        ----------
        - num_classes (int): Number of classes for the classification task.
        - input_shape (tuple, optional): Shape of the input data.
        - l2_rate (float, optional): L2 regularization rate.

        Returns
        -------
        - List: A list of numpy arrays representing the initial weights of the model.
        """
        inputs = tf.keras.layers.Input(shape=input_shape, name="audio_input")
        x = Network.conv_block(input_shape, 24, l2_rate)(inputs)
        x = Network.conv_block(x.shape[1:], 32, l2_rate)(x)
        x = Network.conv_block(x.shape[1:], 64, l2_rate)(x)
        x = Network.conv_block(x.shape[1:], 128, l2_rate)(x)
        x = tf.keras.layers.GlobalMaxPool2D()(x)
        outputs = tf.keras.layers.Dense(num_classes)(x)
        model = tf.keras.Model(inputs, outputs, name="audio_classifier")
        return model.get_weights()


class PslNetwork(tf.keras.Model):  # pylint: disable=abstract-method
    """A TensorFlow Keras model for Pseudo-Label Semi-supervised Learning (PSL).

    This class extends tf.keras.Model to implement a specific architecture and
    training procedure for semi-supervised learning using pseudo-labeling.

    Attributes
    ----------
    - num_classes (int): Number of classes for classification.
    - aux_loss_weight (float): Weight for the auxiliary loss in training.
    - dropout_rate (float): Dropout rate for regularization.
    - model (tf.keras.Model): The underlying audio classification model.
    - confidence (float): Confidence threshold for pseudo-labeling.
    """

    def __init__(self, num_classes, aux_loss_weight=0.5, dropout_rate=0.1):
        super().__init__()
        self.model = Network(
            num_classes=num_classes, dropout_rate=dropout_rate
        ).get_network()
        self.confidence = 0.5
        self.aux_loss_weight = aux_loss_weight
        self.history = History()
        self.optimizer = None
        self.loss_fn = None

    # pylint: disable=arguments-differ
    def compile(self, optimizer):
        """Configure the model for training.

        Sets the optimizer and loss function for the model.

        Parameters
        ----------
        - optimizer: The optimizer to use for training the model.
        """
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits

    @staticmethod
    def cosine_confidence_scheduler(
        step, total_steps, confidence_min=0.5, confidence_max=0.9
    ):
        """Determine the confidence level for pseudo-labeling using a cosine schedule.

        Gradually adjusts the confidence threshold over the course of training based
        on the cosine function.

        Parameters
        ----------
        - step (int): The current training step or round.
        - total_steps (int): The total number of steps or rounds
                             in the training process.
        - confidence_min (float, optional): Minimum confidence threshold.
        - confidence_max (float, optional): Maximum confidence threshold.

        Returns
        -------
        - float: The computed confidence threshold for the current step.
        """
        return confidence_max - (confidence_max - confidence_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi)
        )

    # pylint: disable=too-many-arguments, too-many-locals, arguments-differ
    def fit(
        self, train_datasets, epochs, c_round, rounds, num_batches=None, T=1, verbose=0
    ):
        """Trains the model on given datasets for a specified number of epochs.

        This custom training loop handles both labeled and unlabeled data, applying
        pseudo-labeling for semi-supervised learning.

        Parameters
        ----------
        - train_datasets (tuple): A tuple containing labeled
                                  and unlabeled training datasets.
        - epochs (int): Number of epochs to train the model.
        - c_round (int): Current round of training in a federated learning scenario.
        - rounds (int): Total number of rounds in the federated learning scenario.
        - num_batches (int, optional): Number of batches per epoch.
        - T (float, optional): Temperature parameter for softmax in pseudo-labeling.
        - verbose (int, optional): Verbosity mode.

        Returns
        -------
        - History: An object containing training history information.
        """
        l_dataset, u_dataset = train_datasets
        self.history.on_train_begin()
        for epoch in range(epochs):
            start, epoch_loss = time.time(), 0.0
            # Threshold value to get good psuedo labels, part of Eq-5
            self.confidence = __class__.cosine_confidence_scheduler(
                step=c_round, total_steps=rounds
            )
            progbar = tf.keras.utils.Progbar(
                target=num_batches - 1,
                width=30,
                verbose=verbose if verbose != 2 else 0,
                interval=0.05,
                stateful_metrics=None,
                unit_name="step",
            )
            if verbose == 1:
                tf.print(f"Epoch {epoch+1}/{epochs}")
            step = 0
            # Run train step on each batch
            # tf.data.Dataset.repeat
            # (https://www.tensorflow.org/api_docs/python/tf/data/Dataset#repeat)
            # can also be used instead of itertools.cycle
            for step, ((l_batch_x, l_batch_y), (u_batch_x, u_batch_y)) in enumerate(
                zip(itertools.cycle(l_dataset), u_dataset)
            ):
                batch_x, batch_y = tf.concat([l_batch_x, u_batch_x], 0), tf.concat(
                    [l_batch_y, u_batch_y], 0
                )
                loss = self.train_step(x=batch_x, y=batch_y, T=T)
                progbar.update(current=step, values=[("loss", float(loss["loss"]))])
                epoch_loss += float(loss["loss"])
            # Store result in history
            self.history.on_epoch_end(
                epoch=epoch,
                logs={"loss": epoch_loss / num_batches, "accuracy": float(0.0)},
            )
            if verbose == 2:
                tf.print(
                    f"Epoch {epoch + 1}/{epochs}\n"
                    f"{step}/{step} - {time.time() - start:.1f}s - "
                    f"loss: {epoch_loss / num_batches:.4f} - "
                    f"accuracy: {float(0.0):.4f}"
                )
        return self.history

    # pylint: disable=too-many-arguments, too-many-locals, arguments-differ
    @tf.function
    def train_step(self, x, y, T=4):
        """Perform a single training step on a batch of data.

        Applies the forward pass, loss computation, and backpropagation to update
        the model weights using both labeled and unlabeled data.

        Parameters
        ----------
        - x (Tensor): Input data.
        - y (Tensor): Corresponding labels (or -1 for unlabeled data).
        - T (float, optional): Temperature parameter for softmax in pseudo-labeling.

        Returns
        -------
        - dict: A dictionary containing the computed losses for the training step.
        """
        with tf.GradientTape() as tape:
            logits, labels = self.model(x, training=True), y
            labeled_mask, unlabeled_mask = tf.not_equal(-1, labels), tf.cast(
                tf.equal(-1, labels), dtype=tf.float32
            )
            num_labeled, num_unlabeled = tf.math.reduce_sum(
                tf.cast(labeled_mask, dtype=tf.float32)
            ), tf.math.reduce_sum(unlabeled_mask)
            masked_logits, masked_labels = tf.boolean_mask(
                logits, labeled_mask
            ), tf.boolean_mask(labels, labeled_mask)
            # Equation 4: Loss function for labelled images.
            loss_labeled = (
                tf.reduce_sum(self.loss_fn(labels=masked_labels, logits=masked_logits))
                / num_labeled
            )
            # Equation 5: Psuedo labels for the unlabelled images.
            pseudo_labels = tf.stop_gradient(input=tf.nn.softmax(logits / T))
            pseudo_mask = tf.cast(
                tf.reduce_max(pseudo_labels, axis=1) >= self.confidence,
                dtype=tf.float32,
            )
            # Equation 6: Loss function for unlabelled images.
            loss_unlabeled = (
                tf.reduce_sum(
                    (
                        tf.cast(
                            self.loss_fn(
                                labels=tf.argmax(pseudo_labels, axis=1), logits=logits
                            ),
                            dtype=tf.float32,
                        )
                        * pseudo_mask
                    )
                    * unlabeled_mask
                )
                / num_unlabeled
            )
            # Equation 7: loss of kth client in rth round (Eq-4 + Eq-6)
            total_loss = tf.add_n(
                [
                    ((1 - self.aux_loss_weight) * loss_labeled)
                    + (self.aux_loss_weight * loss_unlabeled)
                ]
                + self.model.losses
            )
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {
            "loss": total_loss,
            "labelled_loss": loss_labeled,
            "unlabelled_loss": loss_unlabeled,
        }

    def get_evaluation_model(self, input_shape=(None, None, 64, 1)):
        """Create an evaluation model based on the trained PSL network.

        Constructs a model for evaluation by applying the trained network in a
        time-distributed manner to process variable-length inputs.

        Parameters
        ----------
        - input_shape (tuple, optional): Shape of the input
                                         data for the evaluation model.

        Returns
        -------
        - tf.keras.Model: The evaluation model based on the PSL network.
        """
        input_layer = tf.keras.layers.Input(shape=input_shape)
        output = tf.reduce_mean(
            tf.keras.layers.TimeDistributed(self.model)(input_layer), axis=1
        )
        return tf.keras.Model(input_layer, output)
