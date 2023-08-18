Use with TensorFlow
===================

You can integrate flwr-dataset with TensorFlow in two ways.

Flower Datasets used the Dataset abstraction from Hugging Face and there are a few ways to transform it to the
Let's do the standard setup::

  from flwr_datasets import FederatedDataset
  mnist_fds = FederatedDataset(dataset="mnist", partitioners={"train": 100})
  partition_idx_10 = mnist_fds.load_partition(10, "train")
  centralized_dataset = mnist_fds.load_full("test")


The simplest way (if only plan on just passing the dataset to the fit) ::

  partition_idx_10_tf = partition_idx_10.with_format("tf")
  # model defined earlier
  model.fit(partition_idx_10_tf)


If you are used to passing Numpy arrays to the fit, you can transform the Dataset object to numpy too.
Then you can also visualize the data and calculate the statistics more easily::

  partition_idx_10_np = partition_idx_10.with_format("numpy")
  X_idx_10 = partition_idx_10_np["image"]
  y_idx_10 = partition_idx_10_np["label"]


