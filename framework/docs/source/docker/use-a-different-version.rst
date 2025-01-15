:og:description: Learn how to switch to a different version of Flower, including nightly builds, by changing Docker image tags for consistent federated learning environments.
.. meta::
    :description: Learn how to switch to a different version of Flower, including nightly builds, by changing Docker image tags for consistent federated learning environments.

Use a Different Flower Version
==============================

If you want to use a different version of Flower, for example Flower nightly, you can do
so by changing the tag. All available versions are on `Docker Hub
<https://hub.docker.com/u/flwr>`__.

.. important::

    When using Flower nightly, the SuperLink nightly image must be paired with the
    corresponding SuperNode and ServerApp nightly images released on the same day. To
    ensure the versions are in sync, using the concrete tag, e.g.,
    ``1.10.0.dev20240610`` instead of ``nightly`` is recommended.
