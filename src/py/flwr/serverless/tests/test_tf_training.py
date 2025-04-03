import json
from typing import List, Tuple, Any
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from flwr.common import (
    Code,
    FitRes,
    NDArrays,
    Parameters,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, FedAdam, FedAvgM
from uuid import uuid4
from flwr.serverless.federated_node.async_federated_node import AsyncFederatedNode
from flwr.serverless.federated_node.sync_federated_node import SyncFederatedNode
from flwr.serverless.shared_folder.in_memory_folder import InMemoryFolder
from flwr.serverless.shared_folder.local_folder import LocalFolder
from flwr.serverless.keras.federated_learning_callback import FlwrFederatedCallback
from flwr.serverless.keras.example import (
    FederatedLearningTestRun,
    MnistModelBuilder,
    split_training_data_into_paritions,
)

# os.environ["CUDA_VISIBLE_DEVICES"] = ""


def test_mnist_training_clients_on_partitioned_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train.shape: (60000, 28, 28)
    # print(y_train.shape) # (60000,)
    epochs = 6
    image_size = x_train.shape[1]
    batch_size = 32
    steps_per_epoch = 8
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    x_train = x_train.astype(np.float32) / 255
    x_test = x_test.astype(np.float32) / 255

    model_standalone1 = MnistModelBuilder().run()
    model_standalone2 = MnistModelBuilder().run()

    partitioned_x_train, partitioned_y_train = split_training_data_into_paritions(
        x_train, y_train, num_partitions=2
    )
    x_train_partition_1 = partitioned_x_train[0]
    y_train_partition_1 = partitioned_y_train[0]
    x_train_partition_2 = partitioned_x_train[1]
    y_train_partition_2 = partitioned_y_train[1]

    # Using generator for its ability to resume. This is important for federated learning, otherwise in each federated round,
    # the cursor starts from the beginning every time.
    def train_generator1(batch_size):
        while True:
            for i in range(0, len(x_train_partition_1), batch_size):
                yield x_train_partition_1[i : i + batch_size], y_train_partition_1[
                    i : i + batch_size
                ]

    def train_generator2(batch_size):
        while True:
            for i in range(0, len(x_train_partition_2), batch_size):
                yield x_train_partition_2[i : i + batch_size], y_train_partition_2[
                    i : i + batch_size
                ]

    train_loader_standalone1 = train_generator1(batch_size)
    train_loader_standalone2 = train_generator2(batch_size)
    model_standalone1.fit(
        train_loader_standalone1, epochs=epochs, steps_per_epoch=steps_per_epoch
    )
    model_standalone2.fit(
        train_loader_standalone2, epochs=epochs, steps_per_epoch=steps_per_epoch
    )
    _, accuracy_standalone1 = model_standalone1.evaluate(
        x_test, y_test, batch_size=batch_size, steps=10
    )
    _, accuracy_standalone2 = model_standalone2.evaluate(
        x_test, y_test, batch_size=batch_size, steps=10
    )
    assert accuracy_standalone1 < 0.55
    assert accuracy_standalone2 < 0.55

    # federated learning
    model_client1 = MnistModelBuilder().run()
    model_client2 = MnistModelBuilder().run()

    # strategy = FedAvg()
    strategy = FedAvgM()
    # FedAdam does not work well in this setting.
    # tmp_model = CreateMnistModel().run()
    # strategy = FedAdam(initial_parameters=ndarrays_to_parameters(tmp_model.get_weights()), eta=1e-1)
    client_0 = None
    client_1 = None

    num_federated_rounds = epochs
    num_epochs_per_round = 1
    train_loader_client1 = train_generator1(batch_size=batch_size)
    train_loader_client2 = train_generator2(batch_size=batch_size)
    for i_round in range(num_federated_rounds):
        print("\n============ Round", i_round)
        # TODO: bug! dataloader starts from the beginning of the dataset! We should use a generator
        model_client1.fit(
            train_loader_client1,
            epochs=num_epochs_per_round,
            steps_per_epoch=steps_per_epoch,
        )
        model_client2.fit(
            train_loader_client2,
            epochs=num_epochs_per_round,
            steps_per_epoch=steps_per_epoch,
        )
        num_examples = batch_size * 10

        param_0: Parameters = ndarrays_to_parameters(model_client1.get_weights())
        param_1: Parameters = ndarrays_to_parameters(model_client2.get_weights())

        # Aggregation using the strategy.
        results: List[Tuple[ClientProxy, FitRes]] = [
            (
                client_0,
                FitRes(
                    status=Status(code=Code.OK, message="Success"),
                    parameters=param_0,
                    num_examples=num_examples,
                    metrics={},
                ),
            ),
            (
                client_1,
                FitRes(
                    status=Status(code=Code.OK, message="Success"),
                    parameters=param_1,
                    num_examples=num_examples,
                    metrics={},
                ),
            ),
        ]

        aggregated_parameters, _ = strategy.aggregate_fit(
            server_round=i_round + 1, results=results, failures=[]
        )
        # turn actual_aggregated back to keras.Model.
        aggregated_parameters_numpy: NDArrays = parameters_to_ndarrays(
            aggregated_parameters
        )
        # Update client model weights using the aggregated parameters.
        model_client1.set_weights(aggregated_parameters_numpy)
        model_client2.set_weights(aggregated_parameters_numpy)

    _, accuracy_federated = model_client1.evaluate(
        x_test, y_test, batch_size=32, steps=10
    )
    assert accuracy_federated > accuracy_standalone1
    assert accuracy_federated > accuracy_standalone2
    assert accuracy_federated > 0.6  # flaky test


def test_mnist_training_standalone():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train.shape: (60000, 28, 28)
    # print(y_train.shape) # (60000,)
    # Normalize
    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    x_train = x_train.astype(np.float32) / 255
    x_test = x_test.astype(np.float32) / 255
    model = MnistModelBuilder().run()

    model.fit(x_train, y_train, epochs=3, batch_size=32, steps_per_epoch=10)
    # TODO: look into the history object to get accuracy
    # memorization test
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=32, steps=10)
    # print(history[-1])
    assert accuracy > 0.6


def test_mnist_training_using_federated_nodes():
    # epochs = standalone_epochs = 3  # does not work
    epochs = standalone_epochs = 8  # works

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train.shape: (60000, 28, 28)
    # print(y_train.shape) # (60000,)
    # Normalize
    image_size = x_train.shape[1]
    batch_size = 32
    steps_per_epoch = 8

    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    x_train = x_train.astype(np.float32) / 255
    x_test = x_test.astype(np.float32) / 255

    model_standalone1 = MnistModelBuilder().run()
    model_standalone2 = MnistModelBuilder().run()

    partitioned_x_train, partitioned_y_train = split_training_data_into_paritions(
        x_train, y_train, num_partitions=2
    )
    x_train_partition_1 = partitioned_x_train[0]
    y_train_partition_1 = partitioned_y_train[0]
    x_train_partition_2 = partitioned_x_train[1]
    y_train_partition_2 = partitioned_y_train[1]

    # Using generator for its ability to resume. This is important for federated learning, otherwise in each federated round,
    # the cursor starts from the beginning every time.
    def train_generator1(batch_size):
        while True:
            for i in range(0, len(x_train_partition_1), batch_size):
                yield x_train_partition_1[i : i + batch_size], y_train_partition_1[
                    i : i + batch_size
                ]

    def train_generator2(batch_size):
        while True:
            for i in range(0, len(x_train_partition_2), batch_size):
                yield x_train_partition_2[i : i + batch_size], y_train_partition_2[
                    i : i + batch_size
                ]

    train_loader_standalone1 = train_generator1(batch_size)
    train_loader_standalone2 = train_generator2(batch_size)
    model_standalone1.fit(
        train_loader_standalone1, epochs=epochs, steps_per_epoch=steps_per_epoch
    )
    model_standalone2.fit(
        train_loader_standalone2, epochs=epochs, steps_per_epoch=steps_per_epoch
    )
    print("Evaluating on the combined test set:")
    _, accuracy_standalone1 = model_standalone1.evaluate(
        x_test, y_test, batch_size=batch_size, steps=10
    )
    _, accuracy_standalone2 = model_standalone2.evaluate(
        x_test, y_test, batch_size=batch_size, steps=10
    )
    assert accuracy_standalone1 < 0.55
    assert accuracy_standalone2 < 0.55

    # federated learning
    model_client1 = MnistModelBuilder().run()
    model_client2 = MnistModelBuilder().run()

    strategy = FedAvg()
    # strategy = FedAvgM()
    # FedAdam does not work well in this setting.
    # tmp_model = CreateMnistModel().run()
    # strategy = FedAdam(initial_parameters=ndarrays_to_parameters(tmp_model.get_weights()), eta=1e-1)

    num_federated_rounds = standalone_epochs
    num_epochs_per_round = 1
    train_loader_client1 = train_generator1(batch_size=batch_size)
    train_loader_client2 = train_generator2(batch_size=batch_size)

    storage_backend = InMemoryFolder()
    node1 = AsyncFederatedNode(shared_folder=storage_backend, strategy=strategy)
    node2 = AsyncFederatedNode(shared_folder=storage_backend, strategy=strategy)
    for i_round in range(num_federated_rounds):
        print("\n============ Round", i_round)
        model_client1.fit(
            train_loader_client1,
            epochs=num_epochs_per_round,
            steps_per_epoch=steps_per_epoch,
        )
        num_examples = batch_size * 10
        param_1: Parameters = ndarrays_to_parameters(model_client1.get_weights())
        updated_param_1, _ = node1.update_parameters(param_1, num_examples=num_examples)
        if updated_param_1 is not None:
            model_client1.set_weights(parameters_to_ndarrays(updated_param_1))
        else:
            print("node1 is waiting for other nodes to send their parameters")

        model_client2.fit(
            train_loader_client2,
            epochs=num_epochs_per_round,
            steps_per_epoch=steps_per_epoch,
        )
        num_examples = batch_size * 10
        param_2: Parameters = ndarrays_to_parameters(model_client2.get_weights())
        updated_param_2, _ = node2.update_parameters(param_2, num_examples=num_examples)
        if updated_param_2 is not None:
            model_client2.set_weights(parameters_to_ndarrays(updated_param_2))
        else:
            print("node2 is waiting for other nodes to send their parameters")

        print("Evaluating on the combined test set:")
        _, accuracy_federated = model_client1.evaluate(
            x_test, y_test, batch_size=32, steps=10
        )

    assert accuracy_federated > accuracy_standalone1
    assert accuracy_federated > accuracy_standalone2
    assert accuracy_federated > 0.6  # flaky test


def test_mnist_federated_callback_2nodes():
    epochs = 8
    accuracy_standalone, accuracy_federated = FederatedLearningTestRun(
        num_nodes=2,
        epochs=epochs,
        num_rounds=epochs,
        lr=0.001,
        strategy=FedAvg(),
    ).run()
    for i in range(len(accuracy_standalone)):
        assert accuracy_standalone[i] < 1.0 / len(accuracy_standalone) + 0.05

    assert accuracy_federated[0] > accuracy_standalone[0]
    assert accuracy_federated[0] > 1.0 / len(accuracy_standalone) + 0.05


def test_mnist_federated_callback_2nodes_synchronously(tmpdir):
    epochs = 8
    local_shared_folder = InMemoryFolder()
    # local_shared_folder = LocalFolder(directory=str(tmpdir.join("fed_test")))
    session = FederatedLearningTestRun(
        num_nodes=2,
        epochs=epochs,
        num_rounds=epochs,
        lr=0.001,
        strategy=FedAvg(),
        train_concurrently=True,
        use_async_node=False,
        save_model_before_aggregation=True,
        storage_backend=local_shared_folder,
    )
    accuracy_standalone, accuracy_federated = session.run()
    for i in range(len(accuracy_standalone)):
        assert accuracy_standalone[i] < 1.0 / len(accuracy_standalone) + 0.05

    assert accuracy_federated[0] > accuracy_standalone[0]
    assert accuracy_federated[0] > 1.0 / len(accuracy_standalone) + 0.05

    # assert metrics files are tracked
    node_id = session.nodes[0].node_id
    raw_folder = session.storage_backend.get_raw_folder()
    json_bytes = raw_folder[f"keras/{node_id}/metrics_before_aggregation_00000.json"]
    assert json_bytes is not None
    metrics_dict = json.loads(json_bytes.decode("utf-8"))
    assert metrics_dict["loss"] > 0.0
    model_bytes = raw_folder[f"keras/{node_id}/model_before_aggregation_00000.h5"]
    assert model_bytes is not None
    assert len(model_bytes) > 0

    # assert the keras logs object has "*_fed" metrics
    first_callback = session.callbacks_per_client[0]
    assert "accuracy_fed" in first_callback.logs, f"{first_callback.logs}"
    assert "val_accuracy_fed" in first_callback.logs, f"{first_callback.logs}"


def test_mnist_federated_callback_3nodes():
    epochs = 8
    accuracy_standalone, accuracy_federated = FederatedLearningTestRun(
        num_nodes=3,
        epochs=epochs,
        num_rounds=epochs,
        lr=0.001,
        strategy=FedAvg(),
    ).run()
    for i in range(len(accuracy_standalone)):
        assert accuracy_standalone[i] < 1.0 / len(accuracy_standalone) + 0.05

    assert accuracy_federated[0] > accuracy_standalone[0]
    assert accuracy_federated[0] > 1.0 / len(accuracy_standalone) + 0.05


def test_mnist_federated_callback_2nodes_lag0_1(tmpdir):
    epochs = 10
    num_nodes = 2
    accuracy_standalone, accuracy_federated = FederatedLearningTestRun(
        num_nodes=num_nodes,
        epochs=epochs,
        num_rounds=epochs,
        batch_size=32,
        steps_per_epoch=8,
        lr=0.001,
        strategy=FedAvg(),
        # storage_backend=InMemoryFolder(),
        storage_backend=LocalFolder(directory=str(tmpdir.join("fed_test"))),
        train_pseudo_concurrently=True,
        use_async_node=True,
        lag=0.1,
    ).run()
    for i in range(len(accuracy_standalone)):
        assert accuracy_standalone[i] < 1.0 / len(accuracy_standalone) + 0.05

    assert accuracy_federated[-1] > accuracy_standalone[-1]
    assert accuracy_federated[-1] > 1.0 / num_nodes + 0.05


def test_mnist_federated_callback_2nodes_lag2(tmpdir):
    epochs = 10
    num_nodes = 2
    accuracy_standalone, accuracy_federated = FederatedLearningTestRun(
        num_nodes=num_nodes,
        epochs=epochs,
        num_rounds=epochs,
        batch_size=32,
        steps_per_epoch=8,
        lr=0.001,
        strategy=FedAvg(),
        storage_backend=InMemoryFolder(),
        # storage_backend=LocalFolder(directory=str(tmpdir.join("fed_test"))),
        train_pseudo_concurrently=True,
        use_async_node=True,
        lag=2,
    ).run()
    for i in range(len(accuracy_standalone)):
        assert accuracy_standalone[i] < 1.0 / len(accuracy_standalone) + 0.05

    assert accuracy_federated[-1] > accuracy_standalone[-1]
    assert accuracy_federated[-1] > 1.0 / num_nodes + 0.05


def test_mnist_federated_callback_2nodes_concurrent(tmpdir):
    epochs = 8
    num_nodes = 2
    fed_dir = tmpdir.join("fed_test")
    accuracy_standalone, accuracy_federated = FederatedLearningTestRun(
        num_nodes=num_nodes,
        epochs=epochs,
        num_rounds=epochs,
        batch_size=32,
        steps_per_epoch=8,
        lr=0.001,
        strategy=FedAvg(),
        # storage_backend=InMemoryFolder(),
        storage_backend=LocalFolder(directory=str(fed_dir)),
        train_concurrently=True,
        # use_async_node=False,
        use_async_node=True,
    ).run()
    # print(fed_dir.listdir())
    for i in range(len(accuracy_standalone)):
        assert accuracy_standalone[i] < 1.0 / len(accuracy_standalone) + 0.05

    assert accuracy_federated[-1] > accuracy_standalone[-1]
    assert accuracy_federated[-1] > 1.0 / num_nodes + 0.05


if __name__ == "__main__":
    test_mnist_federated_callback_2nodes_concurrent()
