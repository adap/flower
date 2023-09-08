Quickstart
==========

Run the Flower Datasets as fast as possible by learning only the essentials. (Scroll down to copy the whole example)

Install Federated Datasets
--------------------------
Run in command line::

  python -m pip install flwr_datasets[vision]

Install the ML framework
------------------------
TensorFlow::

  pip install tensorflow

PyTorch::

  pip install torch, torchvision

Partition the dataset
-----------------------
::

  from flwr_datasets import FederatedDataset
  mnist_fds = FederatedDataset(dataset="mnist", partitioners={"train": 100})
  partition_idx_10 = mnist_fds.load_partition(10, "train")
  centralized_dataset = mnist_fds.load_full("test")

Now you're ready to go. You have 100 partitions created from the train split of the MNIST dataset and the test split for the centralized evaluation.
Now change the type of the dataset (from datasets.Dataset - HuggingFace type of the dataset) to the one supported by your framework.

Convert
-------

Numpy
^^^^^
Often, especially for the smaller dataset, one uses Numpy as the input type of the dataset for TensorFlow::

  partition_idx_10_np = partition_idx_10.with_format("numpy")
  X_idx_10 = partition_idx_10_np["image"]
  y_idx_10 = partition_idx_10_np["label"]

PyTorch DataLoader
^^^^^^^^^^^^^^^^^^
Transform the Dataset directly into the DataLoader::

  from torch.utils.data import DataLoader
  partition_idx_10_torch = partition_idx_10.with_format("torch")
  dataloader_idx_10 = DataLoader(partition_idx_10_torch, batch_size=16)

TensorFlow Tensors
^^^^^^^^^^^^^^^^^^
It's enough to just call the method below and pass this dataset to the .fit() method::

  partition_idx_10_tf = partition_idx_10.with_format("tf")

