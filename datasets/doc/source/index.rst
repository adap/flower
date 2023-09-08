Flower Datasets
===============

Flower Datasets (``flwr-datasets``) is a library to quickly and easily create datasets for federated
learning/analytics/evaluation. It is created by the ``Flower Labs`` team that also created `Flower <https://flower.dev>`_ - a Friendly Federated Learning Framework.

.. toctree::
   :maxdepth: 2
   :hidden:

   get-started
   how-to
   ref-api-flwr-datasets

Main features
-------------
Flower Datasets library supports:

- **downloading datasets** - choose the dataset from Hugging Face's ``dataset``
- **partitioning datasets** - customize the partitioning scheme
- **creating centralized datasets** - leave parts of the dataset unpartitioned (e.g. for centralized evaluation)

Thanks to using Hugging Face's ``datasets`` used under the hood, Flower Datasets integrates with the following popular formats/frameworks:

- Hugging Face
- PyTorch
- TensorFlow
- Numpy
- Pandas
- Jax
- Arrow

Install
-------

The simplest install is::

  python -m pip install flwr_datasets

Or, if you plan to use the image datasets::

  python -m pip install flwr_datasets[image]

Check out the full details on the download in :doc:`get-started-installation`.

How To Use the library
----------------------
Learn how to use the ``flwr-datasets`` library from the :doc:`get-started-quickstart` examples .



Join the Flower Community
-------------------------

The Flower Community is growing quickly - we're a friendly group of researchers, engineers, students, professionals, academics, and other enthusiasts.

.. button-link:: https://flower.dev/join-slack
    :color: primary
    :shadow:

    Join us on Slack
