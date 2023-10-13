Flower Framework Documentation
==============================

.. meta::
   :description: Check out the documentation of the main Flower Framework enabling easy Python development for Federated Learning.

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

   tutorial-series-what-is-federated-learning
   tutorial-series-get-started-with-flower-pytorch
   tutorial-series-use-a-federated-learning-strategy-pytorch
   tutorial-series-build-a-strategy-from-scratch-pytorch
   tutorial-series-customize-the-client-pytorch

.. toctree::
   :maxdepth: 1
   :caption: Quickstart tutorials
   :hidden:

   tutorial-quickstart-pytorch
   tutorial-quickstart-tensorflow
   tutorial-quickstart-huggingface
   tutorial-quickstart-jax
   tutorial-quickstart-pandas
   tutorial-quickstart-fastai
   tutorial-quickstart-pytorch-lightning
   tutorial-quickstart-mxnet
   tutorial-quickstart-scikitlearn
   tutorial-quickstart-xgboost
   tutorial-quickstart-android
   tutorial-quickstart-ios

QUICKSTART TUTORIALS: :doc:`PyTorch <tutorial-quickstart-pytorch>` | :doc:`TensorFlow <tutorial-quickstart-tensorflow>` | :doc:`🤗 Transformers <tutorial-quickstart-huggingface>` | :doc:`JAX <tutorial-quickstart-jax>` | :doc:`Pandas <tutorial-quickstart-pandas>` | :doc:`fastai <tutorial-quickstart-fastai>` | :doc:`PyTorch Lightning <tutorial-quickstart-pytorch-lightning>` | :doc:`MXNet <tutorial-quickstart-mxnet>` | :doc:`scikit-learn <tutorial-quickstart-scikitlearn>` | :doc:`XGBoost <tutorial-quickstart-xgboost>` | :doc:`Android <tutorial-quickstart-android>` | :doc:`iOS <tutorial-quickstart-ios>`

We also made video tutorials for PyTorch:

..  youtube:: jOmmuzMIQ4c
   :width: 80%

And TensorFlow:

..  youtube:: FGTc2TQq7VM
   :width: 80%

How-to guides
~~~~~~~~~~~~~

Problem-oriented how-to guides show step-by-step how to achieve a specific goal.

.. toctree::
   :maxdepth: 1
   :caption: How-to guides

   how-to-install-flower
   how-to-configure-clients
   how-to-use-strategies
   how-to-implement-strategies
   how-to-aggregate-evaluation-results
   how-to-save-and-load-model-checkpoints
   how-to-run-simulations
   how-to-monitor-simulation
   how-to-configure-logging
   how-to-enable-ssl-connections
   how-to-upgrade-to-flower-1.0

.. toctree::
   :maxdepth: 1
   :caption: Legacy example guides

   example-walkthrough-pytorch-mnist
   example-pytorch-from-centralized-to-federated
   example-mxnet-walk-through
   example-jax-from-centralized-to-federated
   example-fedbn-pytorch-from-centralized-to-federated

Explanations
~~~~~~~~~~~~

Understanding-oriented concept guides explain and discuss key topics and underlying ideas behind Flower and collaborative AI.

.. toctree::
   :maxdepth: 1
   :caption: Explanations

   explanation-federated-evaluation
   explanation-differential-privacy

References
~~~~~~~~~~

Information-oriented API reference and other reference material.

.. toctree::
   :maxdepth: 2
   :caption: API reference

   ref-api-flwr
   ref-api-cli

.. toctree::
   :maxdepth: 1
   :caption: Reference docs

   ref-example-projects
   ref-telemetry
   ref-changelog
   ref-faq


Contributor docs
----------------

The Flower community welcomes contributions. The following docs are intended to help along the way.


.. toctree::
   :maxdepth: 1
   :caption: Contributor tutorials

   contributor-tutorial-contribute-on-github
   contributor-tutorial-get-started-as-a-contributor

.. toctree::
   :maxdepth: 1
   :caption: Contributor how-to guides

   contributor-how-to-install-development-versions
   contributor-how-to-set-up-a-virtual-env
   contributor-how-to-develop-in-vscode-dev-containers
   contributor-how-to-create-new-messages
   contributor-how-to-write-documentation
   contributor-how-to-release-flower
   contributor-how-to-contribute-translations

.. toctree::
   :maxdepth: 1
   :caption: Contributor explanations

   contributor-explanation-architecture

.. toctree::
   :maxdepth: 1
   :caption: Contributor references

   fed/index
   contributor-ref-good-first-contributions
   contributor-ref-secure-aggregation-protocols


.. Indices and tables
.. ------------------

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
