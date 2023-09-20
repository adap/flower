Quickstart
==========

Run Flower Datasets as fast as possible by learning only the essentials.

Install Federated Datasets
--------------------------
Run on the command line::

  python -m pip install flwr_datasets[vision]

Install the ML framework
------------------------
TensorFlow::

  pip install tensorflow

PyTorch::

  pip install torch torchvision

Choose the dataset
------------------
Choose the dataset by going to Hugging Face `Datasets Hub <https://huggingface.co/datasets>`_ and searching for your
dataset by name. Note that the name is case sensitive, so make sure to pass the correct name as the `dataset` parameter
to `FederatedDataset`.

Partition the dataset
---------------------
::

  from flwr_datasets import FederatedDataset
  fds = FederatedDataset(dataset="cifar10", partitioners={"train": 10})
  partition = fds.load_partition(0, "train")
  centralized_dataset = fds.load_full("test")

Now you're ready to go. You have ten partitions created from the train split of the MNIST dataset and the test split
for the centralized evaluation. We will convert the type of the dataset from Hugging Face's Dataset type to the one
supported by your framework.

Conversion
----------
For more detailed instructions, go to :doc:`how-to`.

PyTorch DataLoader
^^^^^^^^^^^^^^^^^^
Transform the Dataset directly into the DataLoader::

  from torch.utils.data import DataLoader
  from torchvision.transforms import ToTensor

  transforms = ToTensor()
  partition_torch = partition.map(
        lambda img: {"img": transforms(img)}, input_columns="img"
    ).with_format("torch")
  dataloader = DataLoader(partition_torch, batch_size=64)

NumPy
^^^^^
NumPy can be used as input to the TensorFlow model and is very straightforward::

   partition_np = partition.with_format("numpy")
   X_train, y_train = partition_np["img"], partition_np["label"]

TensorFlow Dataset
^^^^^^^^^^^^^^^^^^
Transformation to TensorFlow Dataset is a one-liner::

  tf_dataset = partition.to_tf_dataset(columns="img", label_cols="label", batch_size=64,
                                     shuffle=True)

