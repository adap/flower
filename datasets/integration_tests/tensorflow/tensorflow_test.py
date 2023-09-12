import unittest

from datasets.utils.logging import disable_progress_bar
from parameterized import parameterized_class
import tensorflow as tf

from flwr_datasets import FederatedDataset


# Using parameterized testing, two different sets of parameters are specified:
# 1. CIFAR10 dataset with the simple ToTensor transform.
# 2. CIFAR10 dataset with a composed transform that first converts an image to a tensor
#    and then normalizes it.
@parameterized_class(
    [
        {"dataset_name": "cifar10", "test_split": "test"},
        {"dataset_name": "cifar10", "test_split": "test"},
    ]
)
class FDSToPyTorchCorrectUsage(unittest.TestCase):
    """Test the conversion from FDS to PyTorch Dataset and Dataloader."""

    dataset_name = ""
    test_split = ""
    expected_img_shape_after_transform = [32, 32, 3]

    @classmethod
    def setUpClass(cls):
        """Disable progress bar to keep the log clean.
        """
        disable_progress_bar()

    def _create_tensorflow_dataset(self, batch_size: int) -> tf.data.Dataset:
        """Create a tensorflow dataset from the FederatedDataset."""
        partition_id = 0
        fds = FederatedDataset(dataset=self.dataset_name, partitioners={"train": 100})
        partition = fds.load_partition(partition_id, "train")
        tf_dataset = partition.to_tf_dataset(columns="img", label_cols="label",
                                             batch_size=batch_size,
                                             shuffle=False)
        return tf_dataset

    def test_create_partition_dataset_shape(self) -> None:
        """Test if the DataLoader returns batches with the expected shape."""
        batch_size = 16
        dataset = self._create_tensorflow_dataset(batch_size)
        batch = next(iter(dataset))
        images = batch[0]
        self.assertEqual(tuple(images.shape),
                         (batch_size, *self.expected_img_shape_after_transform))

    def test_create_partition_dataloader_with_transforms_batch_type(self) -> None:
        """Test if the DataLoader returns batches of type dictionary."""
        batch_size = 16
        dataset = self._create_tensorflow_dataset(batch_size)
        batch = next(iter(dataset))
        self.assertIsInstance(batch, tuple)

    def test_create_partition_dataloader_with_transforms_data_type(self) -> None:
        """Test to verify if the data in the DataLoader batches are of type Tensor."""
        batch_size = 16
        dataset = self._create_tensorflow_dataset(batch_size)
        batch = next(iter(dataset))
        images = batch[0]
        self.assertIsInstance(images, tf.Tensor)


if __name__ == '__main__':
    unittest.main()
