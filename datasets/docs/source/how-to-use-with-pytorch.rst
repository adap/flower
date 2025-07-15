Use with PyTorch
================
Let's integrate ``flwr-datasets`` with PyTorch DataLoaders and keep your PyTorch Transform applied to the data.

Standard setup - download the dataset, choose the partitioning::

  from flwr_datasets import FederatedDataset

  fds = FederatedDataset(dataset="cifar10", partitioners={"train": 10})
  partition = fds.load_partition(0, "train")
  centralized_dataset = fds.load_split("test")

Determine the names of the features (you can alternatively do that directly on the Hugging Face website). The name can
vary e.g. "img" or "image", "label" or "labels"::

  partition.features

In case of CIFAR10, you should see the following output.

.. code-block:: none

  {'img': Image(decode=True, id=None),
  'label': ClassLabel(names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
  'frog', 'horse', 'ship', 'truck'], id=None)}


Apply Transforms, Create DataLoader. We will use `Dataset.with_transform() <https://huggingface.co/docs/datasets/v2.14.5/en/package_reference/main_classes#datasets.Dataset.with_transform>`_.
It works on-the-fly, meaning the transforms you specified will be applied only when you access the data, which is also how the transforms work in the PyTorch ecosystem.
The last detail is to know that this function works on the batches of data (even if you select a single element, it is represented as a batch).
That is why we iterate over all the samples from this batch and apply our transforms::

  from torch.utils.data import DataLoader
  from torchvision.transforms import ToTensor

  transforms = ToTensor()
  def apply_transforms(batch):
    batch["img"] = [transforms(img) for img in batch["img"]]
    return batch

  partition_torch = partition.with_transform(apply_transforms)
  # Now, you can check if you didn't make any mistakes by calling partition_torch[0]
  dataloader = DataLoader(partition_torch, batch_size=64)


Alternatively, you can use the `map() <https://huggingface.co/docs/datasets/v2.14.5/en/package_reference/main_classes#datasets.Dataset.map>`_
function. Note that the operation is instant (contrary to the ``set_transform`` and ``with_transform``). Remember that the ``map``
will modify the existing dataset if the key in the dictionary you return is already present and append a new feature if
it did not exist before. Below, we modify the ``"img"`` feature of our dataset.::

  from torch.utils.data import DataLoader
  from torchvision.transforms import ToTensor

  transforms = ToTensor()
  partition_torch = partition.map(
        lambda img: {"img": transforms(img)}, input_columns="img"
    ).with_format("torch")
  dataloader = DataLoader(partition_torch, batch_size=64)

We advise you to keep the
`ToTensor() <https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html>`_ transform (especially if
you used it in your PyTorch code) because it swaps the dimensions from (H x W x C) to (C x H x W). This order is
expected by a model with a convolutional layer.

If you want to divide the dataset, you can use (at any point before passing the dataset to the DataLoader)::

  partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
  partition_train = partition_train_test["train"]
  partition_test = partition_train_test["test"]

If you want to keep the order of samples intact and need a division into 2 or more subsets, you can use::

  from flwr_datasets.utils import divide_dataset
  train, valid, test = divide_dataset(partition, [0.6, 0.2, 0.2])

Or you can simply calculate the indices yourself::

  partition_len = len(partition)
  # Split `partition` 80:20
  num_train_examples = int(0.8 * partition_len)
  # use first 80%
  partition_train = partition.select(range(num_train_examples)) )
  # use last 20%
  partition_test = partition.select(range(num_train_examples, partition_len)) )

And during the training loop, you need to apply one change. With a typical dataloader, you get a list returned for each iteration::

  for batch in all_from_pytorch_dataloader:
    images, labels = batch
    # Or alternatively:
    # images, labels = batch[0], batch[1]

With this dataset, you get a dictionary, and you access the data a little bit differently (via keys not by index)::

  for batch in dataloader:
    images, labels = batch["img"], batch["label"]

