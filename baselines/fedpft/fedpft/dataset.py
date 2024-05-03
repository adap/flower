"""Dataset creation."""

from typing import Callable, Dict

from flwr_datasets.federated_dataset import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision import transforms


# pylint: disable=too-many-instance-attributes
class Dataset:
    """Dataset class."""

    # pylint: disable=too-many-locals, too-many-arguments
    def __init__(
        self,
        dataset: str,
        num_clients: int,
        batch_size: int,
        dirichlet_alpha: float,
        partition_by: str,
        image_column_name: str,
        transform: transforms,
        image_input_size: int,
        seed: int = 0,
        split_size: float = 0.8,
        **kwargs,
    ) -> None:
        """Load the dataset and partition it using dirichlet distribution.

        Parameters
        ----------
        dataset : str
            Name or path of the dataset to be downloaded from HuggingFace.
        num_clients: int
            Number of clients.
        batch_size: int
            Batch size of training and testing dataloaders of clients.
        dirichlet_alpha: float
            Alpha parameter of Dirichlet distribution.
        partition_by: str
            Label named used for partitioning the dataset.
        image_column_name: str
            Column name of image in the dataset.
        transform: transforms
            Transformation of each batch.
        image_input_size: int
            Input size of pre-trained model.
        seed: int, optional
            Seed for partitioning the dataset. Default is 0.
        split_size: float, optional
            The portion of dataset to be used as training and rest as test.
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.image_input_size = image_input_size
        self.transform = transform
        self.batch_size = batch_size
        self.dirichlet_alpha = dirichlet_alpha
        self.partition_by = partition_by
        self.seed = seed
        self.split_size = split_size
        self.image_column_name = image_column_name
        self.kwargs = kwargs

    def get_loaders(self):
        """Partition the datasets and return a list of dataloaders."""
        partitioner = DirichletPartitioner(
            num_partitions=self.num_clients,
            partition_by=self.partition_by,
            alpha=self.dirichlet_alpha,
            min_partition_size=10,
            self_balancing=True,
        )

        fds = FederatedDataset(
            dataset=self.dataset, partitioners={"train": partitioner}
        )
        # Create train/val for each partition and wrap it into DataLoader
        trainloaders, testloaders = [], []
        for partition_id in range(self.num_clients):
            partition = fds.load_partition(partition_id)
            partition = partition.with_transform(self.apply_batch_transforms())
            partition = partition.train_test_split(
                train_size=self.split_size, seed=self.seed
            )
            trainloaders.append(
                DataLoader(partition["train"], batch_size=self.batch_size)
            )
            testloaders.append(
                DataLoader(partition["test"], batch_size=self.batch_size)
            )

        return trainloaders, testloaders

    def apply_batch_transforms(self) -> Callable[[Dict], Dict]:
        """Apply batch transforms for each batch."""

        def batch_transform(batch):
            batch_img = [
                self.transform(
                    img.resize((self.image_input_size, self.image_input_size))
                )
                for img in batch[self.image_column_name]
            ]
            batch_label = list(batch[self.partition_by])

            return {"img": batch_img, "label": batch_label}

        return batch_transform
