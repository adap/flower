import unittest

from datasets.utils.logging import disable_progress_bar
from parameterized import parameterized_class
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

from flwr_datasets import FederatedDataset


# Using parameterized testing, two different sets of parameters are specified:
# 1. CIFAR10 dataset with the simple ToTensor transform.
# 2. CIFAR10 dataset with a composed transform that first converts an image to a tensor
#    and then normalizes it.
@parameterized_class(
    [
        {"dataset_name": "cifar10", "test_split": "test", "transforms": ToTensor()},
        {"dataset_name": "cifar10", "test_split": "test", "transforms": Compose(
            [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )},
    ]
)
class FDSToPyTorchCorrectUsage(unittest.TestCase):
    """Test the conversion from FDS to PyTorch Dataset and Dataloader."""

    dataset_name = ""
    test_split = ""
    transforms = None
    trainloader = None
    expected_img_shape_after_transform = [3, 32, 32]

    @classmethod
    def setUpClass(cls):
        """Disable progress bar to keep the log clean.
        """
        disable_progress_bar()

    def _create_trainloader(self, batch_size: int) -> DataLoader:
        """Create a trainloader from the federated dataset."""
        partition_id = 0
        fds = FederatedDataset(dataset=self.dataset_name, partitioners={"train": 100})
        partition = fds.load_partition(partition_id, "train")
        partition_train_test = partition.train_test_split(test_size=0.2)
        partition_train_test = partition_train_test.map(
            lambda img: {"img": self.transforms(img)}, input_columns="img"
        )
        trainloader = DataLoader(
            partition_train_test["train"].with_format("torch"), batch_size=batch_size,
            shuffle=True
        )
        return trainloader

    def test_create_partition_dataloader_with_transforms_shape(self)-> None:
        """Test if the DataLoader returns batches with the expected shape."""
        batch_size = 16
        trainloader = self._create_trainloader(batch_size)
        batch = next(iter(trainloader))
        images = batch["img"]
        self.assertEqual(tuple(images.shape),
                         (batch_size, *self.expected_img_shape_after_transform))

    def test_create_partition_dataloader_with_transforms_batch_type(self)-> None:
        """Test if the DataLoader returns batches of type dictionary."""
        batch_size = 16
        trainloader = self._create_trainloader(batch_size)
        batch = next(iter(trainloader))
        self.assertIsInstance(batch, dict)

    def test_create_partition_dataloader_with_transforms_data_type(self)-> None:
        """Test to verify if the data in the DataLoader batches are of type Tensor."""
        batch_size = 16
        trainloader = self._create_trainloader(batch_size)
        batch = next(iter(trainloader))
        images = batch["img"]
        self.assertIsInstance(images, Tensor)


if __name__ == '__main__':
    unittest.main()
