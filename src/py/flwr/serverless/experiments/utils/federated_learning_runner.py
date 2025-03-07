from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from typing import List, Any
from tensorflow import keras
from tensorflow.keras.utils import set_random_seed

from flwr.common import ndarrays_to_parameters
from flwr.server.strategy import (
    FedAvg,
    FedAdam,
    FedAvgM,
    FedOpt,
    FedMedian,
)


from flwr.serverless.federated_node.async_federated_node import AsyncFederatedNode
from flwr.serverless.federated_node.sync_federated_node import SyncFederatedNode
from flwr.serverless.shared_folder.in_memory_folder import InMemoryFolder
from flwr.serverless.keras.federated_learning_callback import FlwrFederatedCallback
from flwr.serverless.experiments.utils.base_experiment_runner import BaseExperimentRunner, Config
from flwr.serverless.experiments.utils.custom_wandb_callback import CustomWandbCallback


class FederatedLearningRunner(BaseExperimentRunner):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.storage_backend: Any = InMemoryFolder()
        # In one round, each node trains on its local data for one epoch.
        self.num_rounds = self.epochs  # number of federated rounds (similar to epochs)
        self.lag = 0.1  # lag between nodes in pseudo-concurrent mode

    def run(self):
        config: Config = self.config
        print("\n====== Starting Federated Learning Experiment ======")
        print(f"Configuration:\n- Nodes: {self.config.num_nodes}")
        print(f"- Strategy: {self.config.strategy}")
        print(f"- Mode: {'Asynchronous' if self.config.use_async else 'Synchronous'}")
        print(f"- Data split: {self.config.data_split}")
        print(f"- Rounds: {self.num_rounds}")
        
        if config.random_seed is not None:
            print(f"- Random seed: {config.random_seed}")
            set_random_seed(config.random_seed)

        if config.track:
            import wandb

            strategy = self.config.strategy
            num_nodes = self.config.num_nodes
            data_split = self.config.data_split
            sync_or_async: str = "async" if self.config.use_async else "sync"
            name = f"{sync_or_async}_{strategy}_{num_nodes}_nodes_{data_split}"
            if data_split == "skewed":
                name += f"_{self.config.skew_factor}"
            wandb.init(
                project=self.config.project,
                entity=os.getenv("WANDB_ENTITY", "example_entity"),
                name=name,
                config=config.__dict__,
            )
        self.models = self.create_models()
        self.set_strategy()
        (
            self.partitioned_x_train,
            self.partitioned_y_train,
            self.x_test,
            self.y_test,
        ) = self.split_data()
        print("x_test shape:", self.x_test.shape)
        print("y_test shape:", self.y_test.shape)
        self.train_federated_models()
        self.evaluate()
        if config.track:
            wandb.finish()

    def set_strategy(self):
        if self.strategy_name == "fedavg":
            self.strategies = [FedAvg() for _ in range(self.num_nodes)]
        elif self.strategy_name == "fedavgm":
            self.strategies = [FedAvgM() for _ in range(self.num_nodes)]
        elif self.strategy_name == "fedadam":
            self.strategies = [
                FedAdam(
                    initial_parameters=ndarrays_to_parameters(
                        self.models[i].get_weights()
                    )
                )
                for i in range(self.num_nodes)
            ]
        elif self.strategy_name == "fedopt":
            self.strategies = [
                FedOpt(
                    initial_parameters=ndarrays_to_parameters(
                        self.models[i].get_weights()
                    )
                )
                for i in range(self.num_nodes)
            ]
        elif self.strategy_name == "fedmedian":
            self.strategies = [FedMedian() for _ in range(self.num_nodes)]
        # elif self.strategy_name == "fedyogi":
        #     self.strategy = FedYogi()
        # elif self.strategy_name == "fedadagrad":
        #     self.strategy = FedAdagrad()
        else:
            raise ValueError(f"Strategy not supported: {self.strategy_name}")

    def split_data(self):
        config: Config = self.config
        print("\n----- Data Splitting Phase -----")
        print(f"Splitting data using '{self.data_split}' strategy")
        
        if self.data_split == "random":
            print("Randomly distributing data across nodes")
            return self.random_split()
        elif self.data_split == "partitioned":
            print("Creating partitioned datasets with natural distribution")
            return self.create_partitioned_datasets()
        elif self.data_split == "skewed":
            print(f"Creating skewed partition with factor {config.skew_factor}")
            return self.create_skewed_partition_split(skew_factor=config.skew_factor)
        else:
            raise ValueError("Data split not supported")

    def train_federated_models(
        self,
    ) -> List[keras.Model]:
        print("\n----- Starting Federated Training -----")
        print(f"Training type: {self.federated_type}")
        print(f"Number of rounds: {self.num_rounds}")
        print(f"Steps per epoch: {self.steps_per_epoch}")
        print(f"Batch size: {self.batch_size}")

        if self.federated_type == "pseudo-concurrent":
            print("Training federated models pseudo-concurrently with lag:", self.lag)
            return self._train_federated_models_pseudo_concurrently(self.models)
        elif self.federated_type == "concurrent":
            print("Training federated models concurrently using ThreadPoolExecutor")
            return self._train_federated_models_concurrently(self.models)
        else:
            print("Training federated models sequentially")
            return self._train_federated_models_sequentially(self.models)

    def _train_federated_models_concurrently(
        self, model_federated: List[keras.Model]
    ) -> List[keras.Model]:
        nodes = self.create_nodes()
        num_partitions = self.num_nodes

        callbacks_per_client = [
            FlwrFederatedCallback(
                nodes[i],
                num_examples_per_epoch=self.steps_per_epoch * self.batch_size,
            )
            for i in range(num_partitions)
        ]

        train_loaders = [
            self.get_train_dataloader_for_node(i) for i in range(num_partitions)
        ]

        with ThreadPoolExecutor(max_workers=self.num_nodes) as ex:
            futures = []
            for i_node in range(self.num_nodes):
                callbacks = [
                    callbacks_per_client[i_node],
                ]
                if self.config.track:
                    callbacks.append(CustomWandbCallback(i_node))

                # assert self.test_steps is None
                if self.config.test_steps is not None:
                    x_test = self.x_test[
                        : self.config.test_steps * self.config.batch_size
                    ]
                    y_test = self.y_test[
                        : self.config.test_steps * self.config.batch_size
                    ]
                else:
                    x_test = self.x_test
                    y_test = self.y_test
                future = ex.submit(
                    model_federated[i_node].fit,
                    x=train_loaders[i_node],
                    epochs=self.num_rounds,
                    steps_per_epoch=self.steps_per_epoch,
                    callbacks=callbacks,
                    validation_data=(x_test, y_test),
                    # validation_data=(self.x_test, self.y_test),
                    # validation_steps=self.test_steps,
                    validation_batch_size=self.batch_size,
                )
                futures.append(future)

            train_results = []
            for future in as_completed(futures):
                train_results.append(future.result())

        return model_federated

    def _train_federated_models_pseudo_concurrently(
        self, model_federated: List[keras.Model]
    ) -> List[keras.Model]:
        nodes = self.create_nodes()
        num_partitions = self.num_nodes
        if self.test_steps is None:
            x_test = self.x_test
            y_test = self.y_test
        else:
            x_test = self.x_test[: self.test_steps * self.batch_size, ...]
            y_test = self.y_test[: self.test_steps * self.batch_size, ...]

        callbacks_per_client = [
            FlwrFederatedCallback(
                nodes[i],
                num_examples_per_epoch=self.steps_per_epoch * self.batch_size,
                x_test=x_test,
                y_test=y_test,
            )
            for i in range(num_partitions)
        ]

        num_federated_rounds = self.num_rounds
        num_epochs_per_round = 1
        train_loaders = [
            self.get_train_dataloader_for_node(i) for i in range(num_partitions)
        ]

        seqs = [[]] * self.num_nodes
        for i_node in range(self.num_nodes):
            seqs[i_node] = [
                (i_node, j + i_node * self.lag) for j in range(num_federated_rounds)
            ]
        # mix them up
        execution_sequence = []
        for i_node in range(self.num_nodes):
            execution_sequence.extend(seqs[i_node])
        execution_sequence = [
            x[0] for x in sorted(execution_sequence, key=lambda x: x[1])
        ]
        print(f"Execution sequence: {execution_sequence}")
        if self.test_steps is None:
            x_test = self.x_test
            y_test = self.y_test
        else:
            x_test = self.x_test[: self.test_steps * self.batch_size, ...]
            y_test = self.y_test[: self.test_steps * self.batch_size, ...]
        for i_node in execution_sequence:
            print("Training node", i_node, "is running...")
            model_federated[i_node].fit(
                x=train_loaders[i_node],
                epochs=num_epochs_per_round,
                steps_per_epoch=self.steps_per_epoch,
                callbacks=[callbacks_per_client[i_node]],
                validation_data=(x_test, y_test),
                validation_steps=self.test_steps,
                validation_batch_size=self.batch_size,
            )

            if i_node == 0:
                print("Evaluating on the combined test set:")
                assert (
                    len(y_test.shape) == 1
                ), f"y_test should be 1D, got {y_test.shape}"
                evaluation_metrics = model_federated[0].evaluate(
                    x_test,
                    y_test,
                    batch_size=self.batch_size,
                    steps=self.test_steps,
                    return_dict=True,
                )

        return model_federated

    def _train_federated_models_sequentially(
        self, model_federated: List[keras.Model]
    ) -> List[keras.Model]:
        nodes = self.create_nodes()
        num_partitions = self.num_nodes  # is this needed?

        callbacks_per_client = [
            FlwrFederatedCallback(
                nodes[i], num_examples_per_epoch=self.batch_size * self.steps_per_epoch
            )
            for i in range(num_partitions)
        ]

        num_federated_rounds = self.num_rounds
        num_epochs_per_round = 1
        train_loaders = [
            self.get_train_dataloader_for_node(i) for i in range(num_partitions)
        ]

        if self.config.track:
            from wandb.keras import WandbCallback
            
            wandb_callbacks = [WandbCallback() for i in range(num_partitions)]
        for i_round in range(num_federated_rounds):
            print("\n============ Round", i_round)
            for i_partition in range(num_partitions):
                callbacks = [
                    callbacks_per_client[i_partition],
                ]
                if self.config.track:
                    callbacks.append(wandb_callbacks[i_partition])
                model_federated[i_partition].fit(
                    train_loaders[i_partition],
                    epochs=num_epochs_per_round,
                    steps_per_epoch=self.steps_per_epoch,
                    callbacks=callbacks,
                )
            print("Evaluating on the combined test set:")
            assert (
                len(self.y_test.shape) == 1
            ), f"y_test should be 1D, got {self.y_test.shape}"
            model_federated[0].evaluate(
                self.x_test,
                self.y_test,
                batch_size=self.batch_size,
                steps=self.steps_per_epoch,
            )

        return model_federated

    def create_nodes(self):
        if self.use_async:
            nodes = [
                AsyncFederatedNode(
                    shared_folder=self.storage_backend, strategy=self.strategies[i]
                )
                for i in range(self.num_nodes)
            ]
        else:
            nodes = [
                SyncFederatedNode(
                    shared_folder=self.storage_backend,
                    strategy=self.strategies[i],
                    num_nodes=self.num_nodes,
                )
                for i in range(self.num_nodes)
            ]
        return nodes

    def evaluate(self):
        print("\n----- Final Evaluation Phase -----")
        for i_node in [0]:  # range(self.num_nodes):
            print(f"Evaluating model from node {i_node} on test set:")
            loss1, accuracy1 = self.models[i_node].evaluate(
                self.x_test,
                self.y_test,
                batch_size=self.batch_size,
                steps=self.test_steps,
            )
            print(f"Test Loss: {loss1:.4f}")
            print(f"Test Accuracy: {accuracy1:.4f}")

            if self.config.track:
                import wandb
                to_log = {
                    "test_accuracy": accuracy1,
                    "test_loss": loss1,
                }
                wandb.log(to_log)
        print("\n====== Federated Learning Experiment Complete ======")
