Flower Documentation
====================

Welcome to Flower's documentation. `Flower <https://flower.dev>`_ is a friendly federated learning framework.


Join the Flower Community
-------------------------

The Flower Community is growing quickly - we're a friendly group of researchers, engineers, students, professionals, academics, and other enthusiasts.

.. button-link:: https://flower.dev/join-slack
    :color: primary
    :shadow:

    Join us on Slack


Flower Framework
----------------

The user guide is targeted at researchers and developers who want to use Flower
to bring existing machine learning workloads into a federated setting. One of
Flower's design goals was to make this simple. Read on to learn more.

Tutorials
~~~~~~~~~

A learning-oriented series of federated learning tutorials, the best place to start.

.. toctree::
   :maxdepth: 1
   :caption: Tutorial

   tutorial/Flower-1-Intro-to-FL-PyTorch
   tutorial/Flower-2-Strategies-in-FL-PyTorch

.. toctree::
   :maxdepth: 1
   :caption: Quickstart tutorials
   :hidden:

   quickstart-pytorch
   quickstart-tensorflow
   quickstart-huggingface
   quickstart-pytorch-lightning
   quickstart-mxnet
   quickstart-scikitlearn

QUICKSTART TUTORIALS: :ref:`PyTorch <quickstart-pytorch>` | :ref:`TensorFlow <quickstart-tensorflow>` | :ref:`🤗 Transformers <quickstart-huggingface>` | :ref:`PyTorch Lightning <quickstart-pytorch-lightning>` | :ref:`MXNet <quickstart-mxnet>` | :ref:`scikit-learn <quickstart-scikitlearn>`

How-to guides
~~~~~~~~~~~~~

Problem-oriented how-to guides show step-by-step how to achieve a specific goal.

.. toctree::
   :maxdepth: 1
   :caption: How-to guides

   installation
   configuring-clients
   strategies
   implementing-strategies
   saving-progress
   saving-and-loading-pytorch-checkpoints
   ssl-enabled-connections
   example-walkthrough-pytorch-mnist
   example-pytorch-from-centralized-to-federated
   example-mxnet-walk-through
   example-jax-from-centralized-to-federated
   fedbn-example-pytorch-from-centralized-to-federated
   recommended-env-setup
   upgrade-to-flower-1.0

Explanations
~~~~~~~~~~~~

Understanding-oriented concept guides explain and discuss key topics and underlying ideas behind Flower and collaborative AI.

.. toctree::
   :maxdepth: 1
   :caption: Explanations

   evaluation
   differential-privacy-wrappers

Reference
~~~~~~~~~

Information-oriented API reference and other reference material.

.. toctree::
   :maxdepth: 2
   :caption: API reference

   flwr <apiref-flwr>

.. toctree::
   :maxdepth: 1
   :caption: Reference docs

   examples
   changelog
   faq


Flower Baselines
----------------

Flower Baselines are a collection of organised scripts used to reproduce results from well-known publications or benchmarks. You can check which baselines already exist and/or contribute your own baseline.

.. toctree::
   :maxdepth: 1
   :caption: Flower Baselines
   
   using-baselines
   contributing-baselines


Contributor Guide
-----------------

The Flower authors welcome external contributions. The following guides are
intended to help along the way.

.. toctree::
   :maxdepth: 1
   :caption: Contributor guide

   getting-started-for-contributors
   good-first-contributions
   contributor-setup
   writing-documentation
   architecture
   secagg
   release-process
   creating-new-messages
   devcontainer


.. Indices and tables
.. ------------------

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
