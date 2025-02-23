from wandb.keras import WandbCallback

from experiments.utils.base_experiment_runner import BaseExperimentRunner
from experiments.utils.custom_wandb_callback import CustomWandbCallback


class NonFederatedRunner(BaseExperimentRunner):
    def __init__(self, config, num_nodes, dataset):
        super().__init__(config, num_nodes, dataset)

    def run(self):
        self.models = self.create_models()
        self.train()
        self.evaluate()

    def train(self):
        (
            self.partitioned_x_train,
            self.partitioned_y_train,
            self.x_test,
            self.y_test,
        ) = self.create_partitioned_datasets()

        for i_node in range(self.num_nodes):
            train_loader = self.get_train_dataloader_for_node(i_node)
            self.models[i_node].fit(
                train_loader,
                epochs=self.epochs,
                steps_per_epoch=self.steps_per_epoch,
                callbacks=[
                    CustomWandbCallback(i_node),
                ],
                validation_data=(self.x_test, self.y_test),
                validation_steps=self.steps_per_epoch,
                validation_batch_size=self.batch_size,
            )

    def evaluate(self):
        for i_node in range(self.num_nodes):
            loss1, accuracy1 = self.models[i_node].evaluate(
                self.x_test,
                self.y_test,
                batch_size=self.batch_size,
                steps=self.steps_per_epoch,
            )
