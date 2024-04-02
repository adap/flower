Use with Local Data
===================

You can partition your local files and Python objects in
``Flower Datasets`` library using any available ``Partitioner``.

This guide details how to create a `Hugging Face <https://huggingface.co/>`_ `Dataset <https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset>`_ which is the required type of input for Partitioners.
We will cover:

* local files: CSV, JSON, image, audio,
* in-memory data : dictionary, list, pd.DataFrame, np.ndarray.


General Overview
----------------
`FederatedDataset <ref-api/flwr_datasets.FederatedDataset.html>`_ is an abstraction that performs all the steps to prepare the dataset for FL experiments: downloading, preprocessing (including resplitting), and partitioning.
However, the partitioning happens thanks to the ``partitioners`` of type ``str: Partitioner`` (meaning a dictionary mapping the split name of the dataset to the ``Partitioner``).

You can also use ``Partitioner`` s alone without relying on the ``FederatedDataset`` (skipping the data download part). A ``Partitioner`` is also not concerned if the data is downloaded from the Hugging Face Hub or created from another source (e.g., loaded locally). The only crucial point is that the dataset you assign to the ``Partitioner`` has to be of type `datasets.Dataset <https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.Dataset>`_.

The rest of this guide will explain how to create a Dataset from local files and existing (in memory) Python objects.

Local Files
-----------
CSV
^^^
.. code-block:: python

  from datasets import load_dataset
  from flwr_datasets.partitioner import ChosenPartitioner

  # Single file
  data_files = "path-to-my-file.csv"

  # Multitple Files
  data_files = [ "path-to-my-file-1.csv", "path-to-my-file-2.csv", ...]
  dataset = load_dataset("csv", data_files=data_files)

  # Divided Dataset
  data_files = {
    "train": single_train_file_or_list_of_files,
    "test": single_test_file_or_list_of_files,
    "can-have-more-splits": ...
  }
  dataset = load_dataset("csv", data_files=data_files)

  partitioner = ChosenPartitioner(...)
  partitioner.dataset = dataset
  partition = partitioner.load_partition(partition_id=0)

JSON
^^^^

.. code-block:: python

  from datasets import load_dataset
  from flwr_datasets.partitioner import ChosenPartitioner

  # Single file
  data_files = "path-to-my-file.json"

  # Multitple Files
  data_files = [ "path-to-my-file-1.json", "path-to-my-file-2.json", ...]
  dataset = load_dataset("json", data_files=data_files)

  # Divided Dataset
  data_files = {
    "train": single_train_file_or_list_of_files,
     "test": single_test_file_or_list_of_files,
     "can-have-more-splits": ...
  }
  dataset = load_dataset("json", data_files=data_files)

  partitioner = ChosenPartitioner(...)
  partitioner.dataset = dataset
  partition = partitioner.load_partition(partition_id=0)


Image
^^^^^
You can create an image dataset in tow ways:

1) give a path the the directory

.. code-block:: python

  from datasets import load_dataset
  from flwr_datasets.partitioner import ChosenPartitioner

  # Directly from a directory
  dataset = load_dataset("imagefolder", data_dir="/path/to/folder")
  partitioner = ChosenPartitioner(...)
  partitioner.dataset = dataset
  partition = partitioner.load_partition(partition_id=0)

2) create a dataset from a CSV/JSON file and cast the path column to Image.

.. code-block:: python

  from datasets import Image
  from flwr_datasets.partitioner import ChosenPartitioner

  dataset = csv_data_with_path_column.cast_column("path", Image())

  partitioner = ChosenPartitioner(...)
  partitioner.dataset = dataset
  partition = partitioner.load_partition(partition_id=0)


Audio
^^^^^
Analogously to the image datasets there are two methods here:

1) give a path the the directory

.. code-block:: python

  from datasets import load_dataset
  from flwr_datasets.partitioner import ChosenPartitioner

  dataset = load_dataset("audiofolder", data_dir="/path/to/folder")

  partitioner = ChosenPartitioner(...)
  partitioner.dataset = dataset
  partition = partitioner.load_partition(partition_id=0)

2) create a dataset from a CSV/JSON file and cast the path column to Image.

.. code-block:: python

  from datasets import Audio
  from flwr_datasets.partitioner import ChosenPartitioner

  dataset = csv_data_with_path_column.cast_column("path", Audio())

  partitioner = ChosenPartitioner(...)
  partitioner.dataset = dataset
  partition = partitioner.load_partition(partition_id=0)

In-Memory
---------

From dictionary
^^^^^^^^^^^^^^^
.. code-block:: python

  from datasets import Dataset
  data = {"features": [1, 2, 3], "labels": [0, 0, 1]}
  dataset = Dataset.from_dict(data)

  partitioner = ChosenPartitioner(...)
  partitioner.dataset = dataset
  partition = partitioner.load_partition(partition_id=0)

From list
^^^^^^^^^
.. code-block:: python

  data = [
    {"features": 1, "labels": 0},
    {"features": 2, "labels": 0},
    {"features": 3, "labels": 1}
  ]
  dataset = Dataset.from_dict(my_dict)

  partitioner = ChosenPartitioner(...)
  partitioner.dataset = dataset
  partition = partitioner.load_partition(partition_id=0)

From pd.DataFrame
^^^^^^^^^^^^^^^^^
.. code-block:: python

  data = {"features": [1, 2, 3], "labels": [0, 0, 1]}
  df = pd.DataFrame(data)
  dataset = Dataset.from_pandas(df)

  partitioner = ChosenPartitioner(...)
  partitioner.dataset = dataset
  partition = partitioner.load_partition(partition_id=0)

From np.ndarray
^^^^^^^^^^^^^^^
The np.ndarray will be first transformed to pd.DataFrame

.. code-block:: python

  data = np.array([[1, 2, 3], [0, 0, 1]).T
  # You can add the column names by passing columns=["features", "labels"]
  df = pd.DataFrame(data)
  dataset = Dataset.from_pandas(df)

  partitioner = ChosenPartitioner(...)
  partitioner.dataset = dataset
  partition = partitioner.load_partition(partition_id=0)

Partitioner Details
-------------------
Partitioning is triggered automatically during the first ``load_partition`` call.
You do not need to call any “do_partitioning” method.

Partitioner abstraction is designed to allow for a single dataset assignment.

.. code-block:: python

  partitioner.dataset = your_dataset

If you need to do the same partitioning on a different dataset, create a new Partitioner
for that, e.g.:

.. code-block:: python

  iid_partitioner_for_mnist = IidPartitioner(num_partitions=10)
  iid_partitioner_for_mnist.dataset = mnist_dataset

  iid_partitioner_for_cifar = IidPartitioner(num_partitions=10)
  iid_partitioner_for_cifar.dataset = cifar_dataset


More Resources
--------------
If you are looking for more details or you have not found the format you are looking supported please visit the `HuggingFace Datasets docs <https://huggingface.co/docs/datasets/index>`_.
This guide is based on the following ones:

* `General Information <https://huggingface.co/docs/datasets/en/loading>`_
* `Tabular Data <https://huggingface.co/docs/datasets/en/tabular_load>`_
* `Image Data <https://huggingface.co/docs/datasets/en/image_load>`_
* `Audio Data <https://huggingface.co/docs/datasets/en/audio_load>`_
