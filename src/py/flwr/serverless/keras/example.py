from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Any, Callable
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow import keras

from flwr.server.strategy import Strategy
from flwr.server.strategy import FedAvg, FedAdam, FedAvgM
from flwr.serverless.federated_node.async_federated_node import AsyncFederatedNode
from flwr.serverless.federated_node.sync_federated_node import SyncFederatedNode
from flwr.serverless.shared_folder.in_memory_folder import InMemoryFolder
from flwr.serverless.shared_folder.local_folder import LocalFolder
from flwr.serverless.keras.federated_learning_callback import FlwrFederatedCallback


@dataclass
class FederatedLearningTestRun:
    num_nodes: int = 2
    epochs: int = 8
    num_rounds: int = 8  # number of federated rounds
    batch_size: int = 32
    steps_per_epoch: int = 10
    lr: float = 0.001
    test_steps: int = 10

    strategy: Strategy = FedAvg()
    storage_backend: Any = None
    use_async_node: bool = True
    # Whether to train federated models concurrently or sequentially.
    train_concurrently: bool = False
    train_pseudo_concurrently: bool = False
    lag: float = 0.1

    model_builder_fn: Callable = None
    replicate_num_channels: bool = False
    save_model_before_aggregation: bool = False
    save_model_after_aggregation: bool = False

    def __post_init__(self):
        if self.model_builder_fn is None:
            self.model_builder_fn = MnistModelBuilder(lr=self.lr).run
        if self.storage_backend is None:
            self.storage_backend = InMemoryFolder()
        self.histories = {}

    def run(self):
        (
            self.partitioned_x_train,
            self.partitioned_y_train,
            self.x_test,
            self.y_test,
        ) = self.create_partitioned_datasets()
        model_standalone: List[keras.Model] = self.create_standalone_models()
        model_federated: List[keras.Model] = self.create_federated_models()
        model_standalone = self.train_standalone_models(model_standalone)
        model_federated = self.train_federated_models(model_federated)
        print("Evaluating on the combined test set (standalone models):")
        accuracy_standalone = self.evaluate_models(model_standalone)
        for i_node in range(len(accuracy_standalone)):
            print(
                "Standalone accuracy for node {}: {}".format(
                    i_node, accuracy_standalone[i_node]
                )
            )
        print("Evaluating on the combined test set (federated model):")
        # Evaluating only the first model.
        accuracy_federated = self.evaluate_models(model_federated)
        for i_node in range(self.num_nodes):  # [len(accuracy_federated) - 1]:
            print(
                "Federated accuracy for node {}: {}".format(
                    i_node, accuracy_federated[i_node]
                )
            )

        return accuracy_standalone, accuracy_federated

    def create_partitioned_datasets(self):
        num_partitions = self.num_nodes

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # x_train.shape: (60000, 28, 28)
        # print(y_train.shape) # (60000,)
        # Normalize
        image_size = x_train.shape[1]
        x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
        x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
        if self.replicate_num_channels:
            x_train = np.tile(x_train, (1, 1, 1, 3))
            x_test = np.tile(x_test, (1, 1, 1, 3))
        x_train = x_train.astype(np.float32) / 255
        x_test = x_test.astype(np.float32) / 255
        partitioned_x_train, partitioned_y_train = split_training_data_into_paritions(
            x_train, y_train, num_partitions=num_partitions
        )
        return partitioned_x_train, partitioned_y_train, x_test, y_test

    def create_standalone_models(self):
        return [self.model_builder_fn() for _ in range(self.num_nodes)]

    def get_train_dataloader_for_node(self, node_idx: int):
        partition_idx = node_idx
        batch_size = self.batch_size
        partitioned_x_train = self.partitioned_x_train
        partitioned_y_train = self.partitioned_y_train
        while True:
            for i in range(0, len(partitioned_x_train[partition_idx]), batch_size):
                yield partitioned_x_train[partition_idx][
                    i : i + batch_size
                ], partitioned_y_train[partition_idx][i : i + batch_size]

    def create_federated_models(self):
        models = [self.model_builder_fn() for _ in range(self.num_nodes)]
        self.models = models
        return models

    def train_standalone_models(
        self, model_standalone: List[keras.Model]
    ) -> List[keras.Model]:
        for i_node in range(self.num_nodes):
            train_loader_standalone = self.get_train_dataloader_for_node(i_node)
            self.histories[i_node] = model_standalone[i_node].fit(
                train_loader_standalone,
                epochs=self.epochs,
                steps_per_epoch=self.steps_per_epoch,
            )

        return model_standalone

    def train_federated_models(
        self, model_federated: List[keras.Model]
    ) -> List[keras.Model]:
        if self.train_pseudo_concurrently:
            print("Training federated models pseudo-concurrently.")
            return self._train_federated_models_pseudo_concurrently(model_federated)
        elif self.train_concurrently:
            print("Training federated models concurrently")
            return self._train_federated_models_concurrently(model_federated)
        else:
            print("Training federated models sequentially")
            return self._train_federated_models_sequentially(model_federated)

    def _train_federated_models_concurrently(
        self, model_federated: List[keras.Model]
    ) -> List[keras.Model]:
        strategy = self.strategy
        storage_backend = self.storage_backend
        if self.use_async_node:
            nodes = []
            for _ in range(self.num_nodes):
                if isinstance(storage_backend, LocalFolder):
                    # duplicate
                    storage_backend = LocalFolder(directory=storage_backend.directory)
                nodes.append(
                    AsyncFederatedNode(shared_folder=storage_backend, strategy=strategy)
                )
        else:
            nodes = []
            for _ in range(self.num_nodes):
                if isinstance(storage_backend, LocalFolder):
                    # duplicate
                    storage_backend = LocalFolder(directory=storage_backend.directory)
                nodes.append(
                    SyncFederatedNode(
                        shared_folder=storage_backend,
                        strategy=strategy,
                        num_nodes=self.num_nodes,
                    )
                )

        self.nodes = nodes
        for i, node in enumerate(nodes):
            print(f"node {i}: folder {node.model_store}")
        num_partitions = self.num_nodes
        model_federated = [self.model_builder_fn() for _ in range(num_partitions)]
        callbacks_per_client = [
            FlwrFederatedCallback(
                nodes[i],
                x_test=self.x_test,
                y_test=self.y_test,
                num_examples_per_epoch=self.steps_per_epoch * self.batch_size,
                save_model_before_aggregation=self.save_model_before_aggregation,
            )
            for i in range(num_partitions)
        ]
        self.callbacks_per_client = callbacks_per_client

        train_loaders = [
            self.get_train_dataloader_for_node(i) for i in range(num_partitions)
        ]

        with ThreadPoolExecutor(max_workers=self.num_nodes) as ex:
            futures = []
            for i_node in range(self.num_nodes):
                # time.sleep(0.5 * i_node)
                future = ex.submit(
                    model_federated[i_node].fit,
                    x=train_loaders[i_node],
                    epochs=self.num_rounds,
                    steps_per_epoch=self.steps_per_epoch,
                    callbacks=[callbacks_per_client[i_node]],
                    validation_data=(
                        self.x_test[: self.test_steps * self.batch_size, ...],
                        self.y_test[: self.test_steps * self.batch_size, ...],
                    ),
                    validation_steps=self.test_steps,
                    validation_batch_size=self.batch_size,
                )
                futures.append(future)
            train_results = [future.result() for future in futures]

        return model_federated

    def _train_federated_models_pseudo_concurrently(
        self, model_federated: List[keras.Model]
    ) -> List[keras.Model]:
        # federated learning
        lag = self.lag
        strategy = self.strategy
        storage_backend = self.storage_backend
        if self.use_async_node:
            nodes = [
                AsyncFederatedNode(shared_folder=storage_backend, strategy=strategy)
                for _ in range(self.num_nodes)
            ]
        else:
            raise NotImplementedError()
        self.nodes = nodes
        num_partitions = self.num_nodes
        model_federated = [self.model_builder_fn() for _ in range(num_partitions)]
        callbacks_per_client = [
            FlwrFederatedCallback(
                nodes[i],
                num_examples_per_epoch=self.steps_per_epoch * self.batch_size,
                x_test=self.x_test[: self.test_steps * self.batch_size, ...],
                y_test=self.y_test[: self.test_steps * self.batch_size, ...],
            )
            for i in range(num_partitions)
        ]
        self.callbacks_per_client = callbacks_per_client

        num_federated_rounds = self.num_rounds
        num_epochs_per_round = 1
        train_loaders = [
            self.get_train_dataloader_for_node(i) for i in range(num_partitions)
        ]

        seqs = [[]] * self.num_nodes
        for i_node in range(self.num_nodes):
            seqs[i_node] = [
                (i_node, j + i_node * lag) for j in range(num_federated_rounds)
            ]
        # mix them up
        execution_sequence = []
        for i_node in range(self.num_nodes):
            execution_sequence.extend(seqs[i_node])
        execution_sequence = [
            x[0] for x in sorted(execution_sequence, key=lambda x: x[1])
        ]
        print(f"Execution sequence: {execution_sequence}")
        for i_node in execution_sequence:
            print("Training node", i_node)
            self.histories[i_node] = model_federated[i_node].fit(
                x=train_loaders[i_node],
                epochs=num_epochs_per_round,
                steps_per_epoch=self.steps_per_epoch,
                callbacks=callbacks_per_client[i_node],
                validation_data=(
                    self.x_test[: self.test_steps * self.batch_size, ...],
                    self.y_test[: self.test_steps * self.batch_size, ...],
                ),
                validation_steps=self.test_steps,
                validation_batch_size=self.batch_size,
            )

            if i_node == 0:
                print("Evaluating on the combined test set:")
                model_federated[0].evaluate(
                    self.x_test[: self.test_steps * self.batch_size, ...],
                    self.y_test[: self.test_steps * self.batch_size, ...],
                    batch_size=self.batch_size,
                    steps=10,
                )

        return model_federated

    def _train_federated_models_sequentially(
        self, model_federated: List[keras.Model]
    ) -> List[keras.Model]:
        # federated learning
        strategy = self.strategy
        storage_backend = self.storage_backend
        if self.use_async_node:
            nodes = [
                AsyncFederatedNode(shared_folder=storage_backend, strategy=strategy)
                for _ in range(self.num_nodes)
            ]
        else:
            raise NotImplementedError()
        self.nodes = nodes
        num_partitions = self.num_nodes
        model_federated = [self.model_builder_fn() for _ in range(num_partitions)]
        callbacks_per_client = [
            FlwrFederatedCallback(
                nodes[i], num_examples_per_epoch=self.batch_size * self.steps_per_epoch
            )
            for i in range(num_partitions)
        ]
        self.callbacks_per_client = callbacks_per_client

        num_federated_rounds = self.num_rounds
        num_epochs_per_round = 1
        train_loaders = [
            self.get_train_dataloader_for_node(i) for i in range(num_partitions)
        ]

        for i_round in range(num_federated_rounds):
            print("\n============ Round", i_round)
            for i_partition in range(num_partitions):
                self.histories[i_partition] = model_federated[i_partition].fit(
                    train_loaders[i_partition],
                    validation_data=(
                        self.x_test[: self.test_steps * self.batch_size, ...],
                        self.y_test[: self.test_steps * self.batch_size, ...],
                    ),
                    validation_steps=self.test_steps,
                    epochs=num_epochs_per_round,
                    steps_per_epoch=self.steps_per_epoch,
                    callbacks=callbacks_per_client[i_partition],
                )
            print("Evaluating on the combined test set:")
            model_federated[0].evaluate(
                self.x_test, self.y_test, batch_size=self.batch_size, steps=10
            )

        return model_federated

    def evaluate_models(self, models: List[keras.Model]) -> List[float]:
        accuracies = []
        for model in models:
            _, accuracy = model.evaluate(
                self.x_test,
                self.y_test,
                batch_size=self.batch_size,
                steps=self.test_steps,
            )
            accuracies.append(accuracy)
        return accuracies


class MnistModelBuilder:
    """This is a helper class to create a simple Keras model
    for MNIST digit classification.
    """

    def __init__(self, lr=0.001):
        self.lr = lr

    def run(self):
        model = self._build_model()
        return self._compile_model(model)

    def _build_model(self):
        input = Input(shape=(28, 28, 1))
        x = Conv2D(32, kernel_size=4, activation="relu")(input)
        x = MaxPooling2D()(x)
        x = Conv2D(16, kernel_size=4, activation="relu")(x)
        x = Flatten()(x)
        output = Dense(10, activation="softmax")(x)
        model = Model(inputs=input, outputs=output)
        return model

    def _compile_model(self, model):
        model.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


def split_training_data_into_paritions(x_train, y_train, num_partitions: int = 2):
    # partion 1: classes 0-4
    # partion 2: classes 5-9
    # client 1 train on classes 0-4 only, and validated on 0-9
    # client 2 train on classes 5-9 only, and validated on 0-9
    # both clients will have low accuracy on 0-9 (below 0.6)
    # but when federated, the accuracy will be higher than 0.6
    classes = list(range(10))
    num_classes_per_partition = int(len(classes) / num_partitions)
    partitioned_classes = [
        classes[i : i + num_classes_per_partition]
        for i in range(0, len(classes), num_classes_per_partition)
    ]
    partitioned_x_train = []
    partitioned_y_train = []
    for partition in partitioned_classes:
        partitioned_x_train.append(x_train[np.isin(y_train, partition)])
        partitioned_y_train.append(y_train[np.isin(y_train, partition)])
    return partitioned_x_train, partitioned_y_train
