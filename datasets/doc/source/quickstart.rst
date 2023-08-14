Quickstart
==========

Run the Flower Datasets as fast as possible by learning only the essentials. (Scroll down to copy the whole example)

**Install the library**::

  pip install flwr_datasets[image]

**Install the ML framework of your choice**

TensorFlow::

  pip install tensorflow

PyTorch::

  pip install torch, torchvision


**Partition the dataset**::

  from flwr_dataset import FederatedDataset
  mnist_fds = FederatedDataset(dataset="mnist", partitioners={"train": 100})
  partition_idx_10 = mnist_fds.get_partition(10, "train")
  centralized_dataset = mnist_fds.get_full("test")

Now you're ready to go. You have 100 partitions created from the train split of the MNIST dataset and the test split for the centralized evaluation.
Now change the type of the dataset (from datasets.Dataset - HuggingFace type of the dataset) to the one supported by your framework.

**Change the type to Numpy**

Often, especially for the smaller dataset, one uses Numpy as the input type of the dataset for TensorFlow.

todo

**Create PyTorch DataLoader**

todo

**Create TensorFlow Dataset**

todo
