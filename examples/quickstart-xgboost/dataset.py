from datasets import disable_progress_bar
from torchvision.transforms import transforms

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (IidPartitioner, LinearPartitioner,
                                       SquarePartitioner, ExponentialPartitioner)

disable_progress_bar()

if __name__ == "__main__":
    num_partitions = 100
    SEED = 42
    # For Uniform Partitioner
    # partitioner = IidPartitioner(num_partitions=num_partitions)
    # For Linear Partitioner
    partitioner = LinearPartitioner(num_partitions=num_partitions)
    # For Square Partitioner
    # partitioner = SquarePartitioner(num_partitions=num_partitions)
    # For Exponential Partitioner
    # partitioner = ExponentialPartitioner(num_partitions=num_partitions)

    fds = FederatedDataset(
        dataset="jxie/higgs",
        partitioners={"train": partitioner}
    )

    partition_id = 0
    partition = fds.load_partition(idx=partition_id, split="train")


    # # If you need any PyTorch transforms
    # # This is an example for images
    # def pytorch_transforms(batch):
    #     transform = transforms.Compose(
    #         [
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #         ]
    #     )
    #     batch["img"] = [transform(img) for img in batch["img"]]
    #     return batch
    # partition.set_transform(pytorch_transforms)

    # If you just want the PyTorch Tensors
    partition.set_format("numpy")
    # or e.g. "numpy" for numpy, "pandas", depending on what you need

    # If you need train test on-edge split
    train_test = partition.train_test_split(test_size=0.2, seed=SEED)
    partition_train = train_test["train"]
    partition_test = train_test["test"]

    # If you need centralized
    centralized = fds.load_full("test")
    centralized.set_format("torch")

    print("Raw access")
    print(partition[0])

    # When iterating in the training loop
    # It's a dict not a tuple as in e.g. pytorch
    print("Access in a loop")
    for x_y in partition:
        x, y = x_y["inputs"], x_y["label"]
        print(x)
        print(y)
        break


    # And in case you keep this abstraction and want batches
    # The order needs to differ slightly
    # Firstly to_iterable, then with_format/set_format
    partition = fds.load_partition(idx=0, split="train")
    iterable_partition = partition.to_iterable_dataset().with_format("numpy")
    batch_size = 4

    print("Batch access in a loop")
    for x_y in iterable_partition.iter(batch_size=batch_size):
        x, y = x_y["inputs"], x_y["label"]
        print(x)
        print(y)
        break
