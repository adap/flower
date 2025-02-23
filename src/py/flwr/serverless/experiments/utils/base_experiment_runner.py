from dataclasses import dataclass
import numpy as np

# from flwr_serverless.keras.example import MnistModelBuilder
from flwr.serverless.experiments.model.simple_mnist_model import SimpleMnistModel
from flwr.serverless.experiments.model.keras_models import ResNetModelBuilder


@dataclass
class Config:
    # non shared config parameters
    num_nodes: int
    strategy: str
    project: str = "experiments"
    track: bool = False
    random_seed: int = 0

    # shared config parameters
    use_async: bool = True
    federated_type: str = "concurrent"
    dataset: str = "mnist"
    epochs: int = 100
    batch_size: int = 32
    steps_per_epoch: int = 64
    lr: float = 0.001
    test_steps: int = None
    net: str = "simple"
    data_split: str = "skewed"
    skew_factor: float = 0.9

    # Ignore, for logging purposes
    use_default_configs: bool = False


class BaseExperimentRunner:
    def __init__(self, config, tracking=False):
        if isinstance(config, dict):
            config = Config(**config)
        assert isinstance(
            config, Config
        ), f"config must be of type Config, got {type(config)}"
        self.config = config
        self.num_nodes = config.num_nodes
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.steps_per_epoch = config.steps_per_epoch
        self.lr = config.lr
        # In experiment tracking, log the actual test steps and test data size
        self.test_steps = config.test_steps
        self.use_async = config.use_async
        self.federated_type = config.federated_type
        self.strategy_name = config.strategy
        self.data_split = config.data_split
        self.dataset = config.dataset
        self.net = config.net

        self.tracking = tracking

        self.get_original_data()

    # ***currently works only for mnist***
    def create_models(self):
        if self.dataset == "mnist":
            assert self.net == "simple", f"Net not supported: {self.net} for mnist"
        if self.net == "simple":
            return [SimpleMnistModel(lr=self.lr).run() for _ in range(self.num_nodes)]
        elif self.net == "resnet50":
            return [
                ResNetModelBuilder(lr=self.lr, net="ResNet50", weights="imagenet").run()
                for _ in range(self.num_nodes)
            ]
        elif self.net == "resnet18":
            return [
                ResNetModelBuilder(lr=self.lr, net="ResNet18").run()
                for _ in range(self.num_nodes)
            ]

    def get_original_data(self):
        dataset = self.dataset
        if dataset == "mnist":
            from tensorflow.keras.datasets import mnist

            (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        elif dataset == "cifar10":
            from tensorflow.keras.datasets import cifar10

            (self.x_train, self.y_train), (
                self.x_test,
                self.y_test,
            ) = cifar10.load_data()
            self.y_train = np.squeeze(self.y_train, -1)
            self.y_test = np.squeeze(self.y_test, -1)
        assert len(self.y_train.shape) == 1, f"y_train shape: {self.y_train.shape}"
        assert len(self.y_test.shape) == 1, f"y_test shape: {self.y_test.shape}"

    def normalize_data(self, data):
        image_size = data.shape[1]
        if self.dataset == "mnist":
            reshaped_data = np.reshape(data, [-1, image_size, image_size, 1])
        elif self.dataset == "cifar10":
            reshaped_data = np.reshape(data, [-1, image_size, image_size, 3])
        else:
            raise ValueError(f"Dataset not supported: {self.dataset}")
        normalized_data = reshaped_data.astype(np.float32) / 255
        return normalized_data

    def random_split(self):
        num_partitions = self.num_nodes
        x_train = self.normalize_data(self.x_train)
        x_test = self.normalize_data(self.x_test)

        # shuffle data then partition
        num_train = x_train.shape[0]
        indices = np.random.permutation(num_train)
        x_train = x_train[indices]
        y_train = self.y_train[indices]

        partitioned_x_train = np.array_split(x_train, num_partitions)
        partitioned_y_train = np.array_split(y_train, num_partitions)

        return partitioned_x_train, partitioned_y_train, x_test, self.y_test

    def create_skewed_partition_split(
        self, skew_factor: float = 0.80, num_classes: int = 10
    ):
        # returns a "skewed" partition of data
        # Ex: 0.8 means 80% of the data for one node is 0-4 while 20% is 5-9
        # and vice versa for the other node
        # Note: A skew factor 0f 0.5 would essentially be a random split,
        # and 1 would be like a partition split
        x_train = self.normalize_data(self.x_train)
        x_test = self.normalize_data(self.x_test)

        x_train_by_label = [[] for _ in range(num_classes)]
        y_train_by_label = [[] for _ in range(num_classes)]
        for i in range(len(self.y_train)):
            label = int(self.y_train[i])
            x_train_example = x_train[i]
            x_train_by_label[label].append(x_train_example)
            y_train_by_label[label].append(label)

        # Partition just the classes into n_splits partitions.
        splitted_classes = np.array_split(np.arange(num_classes), self.num_nodes)
        print("splitted_classes", splitted_classes)
        # splitted_classes should look like [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        # Example:
        # Partition 0:
        #   mostly from 0, 1, 2, 3, 4, and a small amount of 5, 6, 7, 8, 9

        def find_partition_that_this_class_belongs_to(class_idx):
            for i, partition in enumerate(splitted_classes):
                if class_idx in partition:
                    return i

        skewed_partitioned_x_train = [[] for _ in range(self.num_nodes)]
        skewed_partitioned_y_train = [[] for _ in range(self.num_nodes)]
        for i in range(num_classes):
            for j in range(len(x_train_by_label[i])):
                class_idx = i
                partition_that_this_class_belongs_to = (
                    find_partition_that_this_class_belongs_to(class_idx)
                )

                # With probability skew_factor, assign examples to the partition,
                # otherwise randomly assign to a partition.
                if np.random.random() < skew_factor:
                    skewed_partitioned_x_train[
                        partition_that_this_class_belongs_to
                    ].append(x_train_by_label[i][j])
                    skewed_partitioned_y_train[
                        partition_that_this_class_belongs_to
                    ].append(y_train_by_label[i][j])
                else:
                    # Randomly assign to a partition.
                    randomly_assigned_partition = int(
                        np.random.random() * self.num_nodes
                    )
                    skewed_partitioned_x_train[randomly_assigned_partition].append(
                        x_train_by_label[i][j]
                    )
                    skewed_partitioned_y_train[randomly_assigned_partition].append(
                        y_train_by_label[i][j]
                    )

        # convert to numpy arrays
        for i in range(self.num_nodes):
            skewed_partitioned_x_train[i] = np.asarray(skewed_partitioned_x_train[i])
            skewed_partitioned_y_train[i] = np.asarray(skewed_partitioned_y_train[i])

        # shuffle data
        for i in range(self.num_nodes):
            num_train = skewed_partitioned_x_train[i].shape[0]
            indices = np.random.permutation(num_train)
            skewed_partitioned_x_train[i] = skewed_partitioned_x_train[i][indices]
            skewed_partitioned_y_train[i] = skewed_partitioned_y_train[i][indices]

        # Print class distribution information
        print("\n=== Data Distribution Across Partitions ===")
        print(f"Skew Factor: {skew_factor:.2f}")
        print(f"Number of Partitions: {self.num_nodes}")
        print("\nClass Distribution by Partition:")
        print("-" * 50)
        
        total_samples = [len(partition) for partition in skewed_partitioned_y_train]
        
        for i in range(self.num_nodes):
            print(f"\nPartition {i} (Total samples: {total_samples[i]})")
            print("-" * 30)
            print("Class  |  Count  |  Percentage")
            print("-" * 30)
            for j in range(10):
                count = np.sum(skewed_partitioned_y_train[i] == j)
                percentage = (count / total_samples[i]) * 100
                print(f"  {j:2d}   |  {count:5d}  |  {percentage:6.2f}%")
        
        print("\n" + "=" * 50)

        return (
            skewed_partitioned_x_train,
            skewed_partitioned_y_train,
            x_test,
            self.y_test,
        )

    def create_partitioned_datasets(self):
        num_partitions = self.num_nodes

        x_train = self.normalize_data(self.x_train)
        x_test = self.normalize_data(self.x_test)

        (
            partitioned_x_train,
            partitioned_y_train,
        ) = self.split_training_data_into_paritions(
            x_train, self.y_train, num_partitions=num_partitions
        )
        return partitioned_x_train, partitioned_y_train, x_test, self.y_test

    def get_train_dataloader_for_node(self, node_idx: int):
        partition_idx = node_idx
        partitioned_x_train = self.partitioned_x_train
        partitioned_y_train = self.partitioned_y_train
        while True:
            for i in range(0, len(partitioned_x_train[partition_idx]), self.batch_size):
                x_train_batch, y_train_batch = (
                    partitioned_x_train[partition_idx][i : i + self.batch_size],
                    partitioned_y_train[partition_idx][i : i + self.batch_size],
                )
                # print("x_train_batch.shape", x_train_batch.shape)
                # print("y_train_batch.shape", y_train_batch.shape)
                # raise Exception("stop")
                yield x_train_batch, y_train_batch

    # ***currently this only works for mnist*** and for num_nodes = 2, 10
    def split_training_data_into_paritions(
        self, x_train, y_train, num_partitions: int = 2
    ):
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
            # partition is a list of int
            if len(y_train.shape) == 2:
                selected = np.isin(y_train, partition)[:, 0]
            elif len(y_train.shape) == 1:
                selected = np.isin(y_train, partition)
            # subsetting based on the first axis
            x_train_selected = x_train[selected]
            assert (
                x_train_selected.shape[0] < x_train.shape[0]
            ), "partitioned dataset should be smaller than original dataset"
            assert x_train_selected.shape[0] == y_train[selected].shape[0]
            partitioned_x_train.append(x_train_selected)
            y_train_selected = y_train[selected]
            partitioned_y_train.append(y_train_selected)

        return partitioned_x_train, partitioned_y_train


# if __name__ == "__main__":

# base_exp = BaseExperimentRunner(config, num_nodes=2)

# base_exp.random_split()
# base_exp.create_skewed_partition_split()
