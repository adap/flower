Quickstart
==========

Run Flower Datasets as fast as possible by learning only the essentials.

Install Federated Datasets
--------------------------
On the command line, run

.. code-block:: bash

  python -m pip install "flwr-datasets[vision]"

Install the ML framework
------------------------
TensorFlow

.. code-block:: bash

  pip install tensorflow

PyTorch

.. code-block:: bash

  pip install torch torchvision

Choose the dataset
------------------
Choose the dataset by going to Hugging Face `Datasets Hub <https://huggingface.co/datasets>`_ and searching for your
dataset by name that you will pass to the `dataset` parameter of `FederatedDataset`. Note that the name is case sensitive.

Partition the dataset
---------------------
To iid partition your dataset, choose the split you want to partition and the number of partitions::

  from flwr_datasets import FederatedDataset

  fds = FederatedDataset(dataset="cifar10", partitioners={"train": 10})
  partition = fds.load_partition(0, "train")
  centralized_dataset = fds.load_split("test")

Now you're ready to go. You have ten partitions created from the train split of the CIFAR10 dataset and the test split
for the centralized evaluation. We will convert the type of the dataset from Hugging Face's `Dataset` type to the one
supported by your framework.

Display the features
--------------------
Determine the names of the features of your dataset (you can alternatively do that directly on the Hugging Face
website). The names can vary along different datasets e.g. "img" or "image", "label" or "labels". You will also see
the names of label categories. Type::

  partition.features

In case of CIFAR10, you should see the following output.

.. code-block:: none

  {'img': Image(decode=True, id=None),
  'label': ClassLabel(names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
  'frog', 'horse', 'ship', 'truck'], id=None)}

Note that the image is denoted by "img" which is crucial for the next steps (conversion you the ML
framework of your choice).

Conversion
----------
For more detailed instructions, go to :doc:`how-to-use-with-pytorch`, :doc:`how-to-use-with-numpy`, or
:doc:`how-to-use-with-tensorflow`.

PyTorch DataLoader
^^^^^^^^^^^^^^^^^^
Transform the Dataset into the DataLoader, use the PyTorch transforms (`Compose` and all the others are also
possible)::

  from torch.utils.data import DataLoader
  from torchvision.transforms import ToTensor

  transforms = ToTensor()
  def apply_transforms(batch):
    batch["img"] = [transforms(img) for img in batch["img"]]
    return batch
  partition_torch = partition.with_transform(apply_transforms)
  dataloader = DataLoader(partition_torch, batch_size=64)

NumPy
^^^^^
NumPy can be used as input to the TensorFlow and scikit-learn models and it is very straightforward::

   partition_np = partition.with_format("numpy")
   X_train, y_train = partition_np["img"], partition_np["label"]

TensorFlow Dataset
^^^^^^^^^^^^^^^^^^
Transformation to TensorFlow Dataset is a one-liner::

  tf_dataset = partition.to_tf_dataset(columns="img", label_cols="label", batch_size=64,
                                     shuffle=True)

