Use with PyTorch
================
There is a really quick way to integrate flwr-dataset datasets to Pytorch DataLoaders.

Standard setup::

  from flwr_datasets import FederatedDataset
  mnist_fds = FederatedDataset(dataset="mnist", partitioners={"train": 100})
  partition_idx_10 = mnist_fds.load_partition(10, "train")
  centralized_dataset = mnist_fds.load_full("test")


Transformation::

  from torch.utils.data import DataLoader
  partition_idx_10_torch = partition_idx_10.with_format("torch")
  dataloader_idx_10 = DataLoader(partition_idx_10_torch, batch_size=16)
