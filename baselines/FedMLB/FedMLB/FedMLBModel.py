"""Implement FedMLB as tf.keras.Model."""

import tensorflow as tf


class FedMLBModel(tf.keras.Model):
    """FedMLB implementation from the paper https://arxiv.org/abs/2207.06936.

    Based on the original implementation at https://github.com/jinkyu032/FedMLB.
    """

    def __init__(self, local_model_mlb, global_model_mlb):
        super(FedMLBModel, self).__init__()
        # both local_model mlb_model are instance of custom mlb model
        self.local_model = local_model_mlb
        self.global_model = global_model_mlb

    def compile(self, optimizer, loss, metrics, kd_loss, lambda_1=0.1, lambda_2=0.3):
        """Configure the model for training.

        Parameters ---------- optimizer : string (name of optimizer) or optimizer
        instance. See tf.keras.optimizers. loss: Loss function. May be a string (name
        of loss function), or a tf.keras.losses.Loss instance. See tf.keras.losses.
        metrics: List of metrics to be evaluated by the model during training and
        testing. Each of this can be a string (name of a built-in function), function
        or a tf.keras.metrics.Metric instance. See tf.keras.metrics. kd_loss :
        tf.keras.losses.Loss Loss function, a tf.keras.losses.Loss instance. lambda_1
        : float See Eq. 8 in the original paper. It weights the impact of hybrid
        cross-entropy loss. lambda_2 : float Used in FedMLB. See Eq. 8 in the
        original paper. It weights the impact of hybrid KL loss.
        """
        super(FedMLBModel, self).compile(
            optimizer=optimizer, loss=loss, metrics=metrics
        )
        self.kd_loss = kd_loss
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )

    def train_step(self, data):
        """Implement logic for one training step.

        This method can be overridden to support custom training logic.
        """
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            out_of_local = self.local_model(x, return_feature=True)
            local_features = out_of_local[:-1]
            logits = out_of_local[-1]

            ce_branch = []
            kl_branch = []
            num_branch = len(local_features)

            ce_loss = self.compiled_loss(
                y, logits, regularization_losses=self.local_model.losses
            )

            ## Compute loss from hybrid branches
            for it in range(num_branch):
                # tf.print("it ", it)
                this_logits = self.global_model(local_features[it], level=it + 1)
                this_ce = self.cross_entropy(y, this_logits)
                this_kl = self.kd_loss(
                    tf.nn.softmax(this_logits), tf.nn.softmax(logits)
                )
                ce_branch.append(this_ce)
                kl_branch.append(this_kl)

            ce_hybrid_loss = tf.reduce_mean(tf.stack(ce_branch))
            kd_loss = tf.reduce_mean(tf.stack(kl_branch))
            fedmlb_loss = (
                ce_loss + self.lambda_1 * ce_hybrid_loss + self.lambda_2 * kd_loss
            )

        trainable_vars = self.local_model.trainable_variables
        gradients = tape.gradient(fedmlb_loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, logits)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """Implement logic for one evaluation step.

        This method can be overridden to support custom evaluation logic.
        """
        x, y = data

        y_pred = self.local_model(x)  # Forward pass
        self.compiled_loss(y, y_pred, regularization_losses=self.local_model.losses)
        self.compiled_metrics.update_state(y, y_pred)
        # self.compiled_metrics
        return {m.name: m.result() for m in self.metrics}

    def get_weights(self):
        """Return the weights of the local model."""
        return self.local_model.get_weights()

    def get_global_weights(self):
        """Return the weights of the global model."""
        return self.global_model.get_weights()
