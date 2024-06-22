Flower Datasets
===============

Flower Datasets (``flwr-datasets``) is a library to quickly and easily create datasets for federated
learning/analytics/evaluation. It is created by the ``Flower Labs`` team that also created `Flower <https://flower.ai>`_ - a Friendly Federated Learning Framework.

Flower Datasets Framework
-------------------------

Tutorials
~~~~~~~~~

A learning-oriented series of tutorials is the best place to start.

.. toctree::
   :maxdepth: 1
   :caption: Tutorial

   tutorial-quickstart

How-to guides
~~~~~~~~~~~~~

Problem-oriented how-to guides show step-by-step how to achieve a specific goal.

.. toctree::
   :maxdepth: 1
   :caption: How-to guides

   how-to-install-flwr-datasets
   how-to-use-with-pytorch
   how-to-use-with-tensorflow
   how-to-use-with-numpy
   how-to-use-with-local-data
   how-to-visualize-label-distribution
   how-to-disable-enable-progress-bar

References
~~~~~~~~~~

Information-oriented API reference and other reference material.

.. autosummary::
   :toctree: ref-api
   :template: autosummary/module.rst
   :caption: API reference
   :recursive:

      flwr_datasets

.. toctree::
   :maxdepth: 1
   :caption: Reference docs

   ref-telemetry

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

The simplest install is

.. code-block:: bash

  python -m pip install flwr-datasets

If you plan to use the image datasets

.. code-block:: bash

  python -m pip install flwr-datasets[vision]

If you plan to use the audio datasets

.. code-block:: bash

  python -m pip install flwr-datasets[audio]

Check out the full details on the download in :doc:`how-to-install-flwr-datasets`.

How To Use the library
----------------------
Learn how to use the ``flwr-datasets`` library from the :doc:`tutorial-quickstart` examples .

Join the Flower Community
-------------------------

The Flower Community is growing quickly - we're a friendly group of researchers, engineers, students, professionals, academics, and other enthusiasts.

.. button-link:: https://flower.ai/join-slack
    :color: primary
    :shadow:

    Join us on Slack
