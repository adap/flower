"""Implement FedAvg+KD as tf.keras.Model."""

import tensorflow as tf


# pylint: disable=W0223
class FedAvgKDModel(tf.keras.Model):
    """FedAvg+KD implementation from the paper https://arxiv.org/abs/2207.06936.

    Based on the original implementation at https://github.com/jinkyu032/FedMLB.
    In practice, this is applying regular knowledge distillation (KD)
    -- [Hinton et al.] https://arxiv.org/abs/1503.02531 --
    at client-side, with the global model of that round working as teacher
    on local data, providing regularization for the student model
    (i.e., client model).
    """

    def __init__(
        self,
        model: tf.keras.Model,
        h_model: tf.keras.Model,
        kd_loss: tf.keras.losses.Loss,
        gamma: float = 0.2,
    ):
        super().__init__()
        self.local_model = model
        self.global_model = h_model
        self.kd_loss = kd_loss
        self.gamma = gamma

    def train_step(self, data):
        """Implement logic for one training step.

        This method can be overridden to support custom training logic.
        """
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        global_output = self.global_model(x, training=True)

        with tf.GradientTape() as tape:
            local_output = self.local_model(x, training=True)  # Forward pass

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            ce_loss = self.compiled_loss(
                y, local_output, regularization_losses=self.local_model.losses
            )
            kd_loss = self.kd_loss(
                tf.nn.softmax(global_output, axis=1),
                tf.nn.softmax(local_output, axis=1),
            )
            fedkd_loss = (1 - self.gamma) * ce_loss + self.gamma * kd_loss

        # Compute gradients
        trainable_vars = self.local_model.trainable_variables
        gradients = tape.gradient(fedkd_loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, local_output)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """Implement logic for one evaluation step.

        This method can be overridden to support custom evaluation logic.
        """
        x, y = data
        y_pred = self.local_model(x, training=False)  # Forward pass
        self.compiled_loss(y, y_pred, regularization_losses=self.local_model.losses)
        self.compiled_metrics.update_state(y, y_pred)
        # self.compiled_metrics
        return {m.name: m.result() for m in self.metrics}

    def get_weights(self):
        """Return the weights of the local model."""
        return self.local_model.get_weights()
