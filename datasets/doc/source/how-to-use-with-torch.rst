Use with PyTorch
================
Let's integrate ``flwr-datasets`` with PyTorch DataLoaders and keep your PyTorch Transform applied to the data.

Standard setup - download the dataset, choose the partitioning::

  from flwr_datasets import FederatedDataset
  mnist_fds = FederatedDataset(dataset="mnist", partitioners={"train": 100})
  partition_idx_10 = mnist_fds.load_partition(10, "train")
  centralized_dataset = mnist_fds.load_full("test")

Apply Transforms, Create DataLoader::

  from torch.utils.data import DataLoader
  from torchvision.transforms import ToTensor

  transforms = ToTensor()
  partition_idx_10_torch = partition_idx_10.map(
        lambda img: {"img": transforms(img)}, input_columns="img"
    ).with_format("torch")
  dataloader_idx_10 = DataLoader(partition_idx_10_torch, batch_size=16)


We advise you to keep the
`ToTensor() <https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html>`_ transform (especially if
you used it in your PyTorch code) because it swaps the dimensions from (H x W x C) to (C x H x W). This order is
expected by a model with a convolutional layer.

If you want to divide the dataset, you can use (at any point before passing the dataset to the DataLoader)::

  partition_train_test = partition_idx_10.train_test_split(test_size=0.2)
  partition_train = partition_train_test["train"]
  partition_test = partition_train_test["test"]

Or you can simply calculate the indices yourself::

  partition_len = len(partition_idx_10)
  partition_train = partition_idx_10[:int(0.8 * partition_len)]
  partition_test = partition_idx_10[int(0.8 * partition_len):]

And during the training loop, you need to apply one change. With a typical dataloader you get a list returned for each iteration::

  for batch in all_from_pytorch_dataloader:
    images, labels = batch
    # Equivalently
    images, labels = batch[0], batch[1]

With this dataset, you get a dictionary, and you access the data a little bit differently (via keys not by index)::

  for batch in dataloader:
    images, labels = batch["img"], batch["label"]
