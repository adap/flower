Flower Framework Documentation
==============================

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

.. toctree::
   :maxdepth: 1
   :caption: Flower Framework

   tutorials
   guides
   reference
   explanations


Be sure to also checkout our brand new Flower Tutorial series over on YouTube: 

.. grid:: 2

  .. grid-item-card::  PyTorch

    ..  youtube:: jOmmuzMIQ4c
       :width: 100%

  .. grid-item-card::  TensorFlow

    ..  youtube:: FGTc2TQq7VM
       :width: 100%

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
   recommended-env-setup
   upgrade-to-flower-1.0

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

   evaluation
   differential-privacy-wrappers

References
~~~~~~~~~~

Information-oriented API reference and other reference material.

.. toctree::
   :maxdepth: 2
   :caption: API reference

   apiref-flwr
   apiref-cli

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
   setup-contrib
   devcontainer
   framework
   baselines
   documentation
   fed/index


.. Indices and tables
.. ------------------

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
