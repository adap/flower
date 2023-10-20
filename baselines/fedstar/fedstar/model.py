import time
import itertools
import numpy as np
import tensorflow as tf

class History(tf.keras.callbacks.History):
    def __init__(self):
        super(tf.keras.callbacks.History, self).__init__()
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


class Network:
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
    def _conv_block(
        input_shape, num_features, l2_rate, dropout_rate=0.1, add_max_pool=True
    ):
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
        inputs = tf.keras.layers.Input(shape=self._input_shape, name="audio_input")
        x = __class__._conv_block(
            self._input_shape, 24, self._l2_rate, dropout_rate=self._dropout_rate
        )(inputs)
        x = __class__._conv_block(
            x.shape[1:], 32, self._l2_rate, dropout_rate=self._dropout_rate
        )(x)
        x = __class__._conv_block(
            x.shape[1:], 64, self._l2_rate, dropout_rate=self._dropout_rate
        )(x)
        x = __class__._conv_block(
            x.shape[1:], 128, self._l2_rate, dropout_rate=self._dropout_rate
        )(x)
        x = tf.keras.layers.GlobalMaxPool2D()(x)
        outputs = tf.keras.layers.Dense(self._num_classes)(x)
        model = tf.keras.Model(inputs, outputs, name="audio_classifier")
        return model

    def get_evaluation_network(self, input_shape=(None, None, 64, 1)):
        model = self.get_network()
        input = tf.keras.layers.Input(shape=input_shape, name="audio_input")
        output = tf.reduce_mean(tf.keras.layers.TimeDistributed(model)(input), axis=1)
        return tf.keras.Model(input, output)

    @staticmethod
    def get_init_weights(num_classes, input_shape=(None, 64, 1), l2_rate=0.0001):
        inputs = tf.keras.layers.Input(shape=input_shape, name="audio_input")
        x = Network._conv_block(input_shape, 24, l2_rate)(inputs)
        x = Network._conv_block(x.shape[1:], 32, l2_rate)(x)
        x = Network._conv_block(x.shape[1:], 64, l2_rate)(x)
        x = Network._conv_block(x.shape[1:], 128, l2_rate)(x)
        x = tf.keras.layers.GlobalMaxPool2D()(x)
        outputs = tf.keras.layers.Dense(num_classes)(x)
        model = tf.keras.Model(inputs, outputs, name="audio_classifier")
        return model.get_weights()


class PSL_Network(tf.keras.Model):
    def __init__(self, num_classes, aux_loss_weight=0.5, dropout_rate=0.1):
        super(PSL_Network, self).__init__()
        self.model = Network(
            num_classes=num_classes, dropout_rate=dropout_rate
        ).get_network()
        self.confidence = 0.5
        self.aux_loss_weight = aux_loss_weight
        self.history = History()

    def compile(self, optimizer):
        super(PSL_Network, self).compile()
        self.optimizer = optimizer
        self.loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits

    @staticmethod
    def cosine_confidence_scheduler(
        step, total_steps, confidence_min=0.5, confidence_max=0.9
    ):
        return confidence_max - (confidence_max - confidence_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi)
        )

    def fit(
        self, train_datasets, epochs, c_round, rounds, num_batches=None, T=1, verbose=0
    ):
        l_dataset, u_dataset = train_datasets
        self.history.on_train_begin()
        for epoch in range(epochs):
            start, epoch_loss = time.time(), 0.0
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
                tf.print("Epoch %d/%d" % (epoch + 1, epochs))
            # Run train step on each batch
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
                    "Epoch %d/%d\n%d/%d - %.1fs - loss: %.4f - accuracy: %.4f"
                    % (
                        epoch + 1,
                        epochs,
                        step,
                        step,
                        time.time() - start,
                        epoch_loss / num_batches,
                        float(0.0),
                    )
                )
        return self.history

    @tf.function
    def train_step(self, x, y, T=4):
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
            loss_labeled = (
                tf.reduce_sum(self.loss_fn(labels=masked_labels, logits=masked_logits))
                / num_labeled
            )
            pseudo_labels = tf.stop_gradient(input=tf.nn.softmax(logits / T))
            pseudo_mask = tf.cast(
                tf.reduce_max(pseudo_labels, axis=1) >= self.confidence,
                dtype=tf.float32,
            )
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
        input = tf.keras.layers.Input(shape=input_shape)
        output = tf.reduce_mean(
            tf.keras.layers.TimeDistributed(self.model)(input), axis=1
        )
        return tf.keras.Model(input, output)
