#####################
 Flower Architecture
#####################

This page summarizes the architecture of Flower federated learning
systems.

In federated learning (FL), there is typically a server and a number of
clients that are connected to it, i.e. the hub-and-spoke topology:

.. figure:: ./_static/flower-architecture-hub-and-spoke.svg
   :align: center
   :width: 600
   :alt: Hub-and-spoke topology in federated learning
   :class: no-scaled-link

   Hub-and-spoke topology in federated learning.

In a real-world deployment, we want such a federation to be able to run
multiple workloads (different model architectures, different training or
evaluation runs, different hyperparameters, etc...). This is why in
Flower, we have long-running server called Superlink, a long-running
client called Supernode, and short-running ServerApp and ClientApp:

.. figure:: ./_static/flower-architecture-basic-architecture.svg
   :align: center
   :width: 600
   :alt: Basic Flower architecture
   :class: no-scaled-link

   The basic Flower architecture for federated learning.

.. tip::

   For more details, please refer to the |serverapp_link|_ and
   |clientapp_link|_ documentation.

With `multi-tenancy`, multiple ServerApp and ClientApps are now allowed
to run on the same long-running SuperLink and SuperNodes.

.. figure:: ./_static/flower-architecture-multi-run.svg
   :align: center
   :width: 600
   :alt: Multi-tenancy federated learning architecture
   :class: no-scaled-link

   Multi-tenancy federated learning architecture with Flower

To illustrate, consider one federated learning training run. In one run,
one ServerApp and a federation of ClientApp will take part. Note that a
SuperNode will only run the ClientApp if it is selected to participate
in the training run. For example, in ``run 1`` below, all the SuperNodes
are selected and run their corresponding ClientApps:

.. figure:: ./_static/flower-architecture-multi-run-1.svg
   :align: center
   :width: 600
   :alt: Multi-tenancy federated learning architecture - Run 1
   :class: no-scaled-link

   Run 1 in multi-tenancy federated learning architecture with Flower.
   All SuperNodes participate in the training round.

However, in ``run 2``, only the first and third SuperNodes are selected
and will participate in the training:

.. figure:: ./_static/flower-architecture-multi-run-2.svg
   :align: center
   :width: 600
   :alt: Multi-tenancy federated learning architecture - Run 2
   :class: no-scaled-link

   Run 2 in multi-tenancy federated learning architecture with Flower.
   Only SuperNodes 1 and 3 are selected and will participate in the
   training round.

To manage all of the concurrently running training runs, Flower adds one
additional long-running service called SuperExec:

.. figure:: ./_static/flower-architecture-deployment-engine.svg
   :align: center
   :width: 800
   :alt: Flower Deployment Engine with SuperExec
   :class: no-scaled-link

   The SuperExec service for managing concurrent training runs in
   Flower.

This allows many users to share the same federation and to just type
``flwr run`` to start their training.

.. important::

   As we continuously enhance Flower at a rapid pace, we'll periodically
   update this explainer document. Feel free to share any feedback with
   us!

.. |clientapp_link| replace::

   ``ClientApp``

.. |serverapp_link| replace::

   ``ServerApp``

.. _clientapp_link: ref-api/flwr.client.ClientApp.html

.. _serverapp_link: ref-api/flwr.server.ServerApp.html
