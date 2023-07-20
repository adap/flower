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

   tutorial/Flower-0-What-is-FL
   tutorial/Flower-1-Intro-to-FL-PyTorch
   tutorial/Flower-2-Strategies-in-FL-PyTorch
   tutorial/Flower-3-Building-a-Strategy-PyTorch
   tutorial/Flower-4-Client-and-NumPyClient-PyTorch

.. toctree::
   :maxdepth: 1
   :caption: Quickstart tutorials
   :hidden:

   quickstart-pytorch
   quickstart-tensorflow
   quickstart-huggingface
   quickstart-jax
   quickstart-pandas
   quickstart-fastai
   quickstart-pytorch-lightning
   quickstart-mxnet
   quickstart-scikitlearn
   quickstart-xgboost
   quickstart-android
   quickstart-ios

QUICKSTART TUTORIALS: :ref:`PyTorch <quickstart-pytorch>` | :ref:`TensorFlow <quickstart-tensorflow>` | :ref:`🤗 Transformers <quickstart-huggingface>` | :ref:`JAX <quickstart-jax>` | :ref:`Pandas <quickstart-pandas>` | :ref:`fastai <quickstart-fastai>` | :ref:`PyTorch Lightning <quickstart-pytorch-lightning>` | :ref:`MXNet <quickstart-mxnet>` | :ref:`scikit-learn <quickstart-scikitlearn>` | :ref:`XGBoost <quickstart-xgboost>` | :ref:`Android <quickstart-android>` | :ref:`iOS <quickstart-ios>`

..  youtube:: FGTc2TQq7VM

How-to guides
~~~~~~~~~~~~~

Problem-oriented how-to guides show step-by-step how to achieve a specific goal.

.. toctree::
   :maxdepth: 1
   :caption: How-to guides

   install-flower
   configure-clients
   strategies
   implementing-strategies
   save-progress
   logging
   saving-and-loading-pytorch-checkpoints
   monitor-simulation
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

   flwr (Python package) <apiref-flwr>
   Flower CLI commands <apiref-cli>

.. toctree::
   :maxdepth: 1
   :caption: Reference docs

   examples
   telemetry
   changelog
   faq


Contributor Guide
-----------------

The Flower authors welcome external contributions. The following guides are
intended to help along the way.

.. toctree::
   :maxdepth: 1
   :caption: Contributor guide

   first-time-contributors
   getting-started-for-contributors
   good-first-contributions
   contributor-setup
   write-documentation
   architecture
   secagg
   release-process
   creating-new-messages
   devcontainer
   fed/index


.. Indices and tables
.. ------------------

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
