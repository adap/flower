import numpy as np
from wandb.keras import WandbCallback


from flwr_serverless.keras.example import MnistModelBuilder

from experiments.utils.base_experiment_runner import BaseExperimentRunner


class CentralizedRunner(BaseExperimentRunner):
    def __init__(self, config, num_nodes, dataset):
        super().__init__(config, dataset)
        self.num_nodes = 1
        self.test_steps = 10

    def run(self):
        self.train_and_eval()

    def train_and_eval(self):
        image_size = self.x_train.shape[1]
        x_train = np.reshape(self.x_train, [-1, image_size, image_size, 1])
        x_test = np.reshape(self.x_test, [-1, image_size, image_size, 1])
        x_train = x_train.astype(np.float32) / 255
        x_test = x_test.astype(np.float32) / 255

        model = MnistModelBuilder(self.lr).run()

        model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            steps_per_epoch=self.steps_per_epoch,
            callbacks=[WandbCallback()],
            validation_data=(
                self.x_test[: self.test_steps * self.batch_size, ...],
                self.y_test[: self.test_steps * self.batch_size, ...],
            ),
            validation_steps=self.test_steps,
            validation_batch_size=self.batch_size,
        )
        # memorization test
        loss, accuracy = model.evaluate(
            x_test, self.y_test, batch_size=self.batch_size, steps=self.steps_per_epoch
        )
