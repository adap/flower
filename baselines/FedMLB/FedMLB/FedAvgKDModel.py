import tensorflow as tf


class FedAvgKDModel(tf.keras.Model):

    def __init__(self, model, h_model):
        super(FedAvgKDModel, self).__init__()
        self.local_model = model
        self.global_model = h_model

    def compile(self, optimizer, loss, metrics, kd_loss, gamma=0.2):
        super(FedAvgKDModel, self).compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.kd_loss = kd_loss
        self.gamma = gamma

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        global_output = self.global_model(x, training=True)

        with tf.GradientTape() as tape:
            local_output = self.local_model(x, training=True)  # Forward pass

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            ce_loss = self.compiled_loss(y, local_output, regularization_losses=self.local_model.losses)
            kd_loss = self.kd_loss(tf.nn.softmax(global_output, axis=1), tf.nn.softmax(local_output, axis=1))
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
        x, y = data
        y_pred = self.local_model(x, training=False)  # Forward pass
        self.compiled_loss(y, y_pred, regularization_losses=self.local_model.losses)
        self.compiled_metrics.update_state(y, y_pred)
        # self.compiled_metrics
        return {m.name: m.result() for m in self.metrics}

    def get_weights(self):
        return self.local_model.get_weights()


