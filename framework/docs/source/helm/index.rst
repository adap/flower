:og:description: Learn how to run Flower for federated learning using Helm charts.
.. meta::
    :description: Learn how to run Flower for federated learning using Helm charts.

Run Flower using Helm
=====================

.. note::

    Flower Helm charts are a Flower Enterprise feature. See `Flower Enterprise
    <https://flower.ai/enterprise>`_ for details.

This guide provides the step-by-step instructions to deploy Flower using Helm, which
simplifies the deployment and management of the Flower framework on Kubernetes. For
instance, to deploy SuperLink and SuperNode services via command line, simply run the
|helm_install_link|_ command like so:

.. code-block:: sh

    # Provision and deploy SuperLink service
    $ helm install flower-superlink path/to/folder/containing/chart

    # Provision and deploy SuperNode service
    $ helm install flower-supernode path/to/folder/containing/chart

Then to tear down the deployment, run the |helm_uninstall_link|_ command:

.. code-block:: sh

    # Uninstall the `flower-superlink` release
    $ helm uninstall flower-superlink

    # Uninstall the `flower-supernode` release
    $ helm uninstall flower-supernode

Running in Production
---------------------

.. toctree::
    :maxdepth: 1

    Deploy SuperLink <how-to-deploy-superlink-using-helm.md>
    Deploy SuperNode <how-to-deploy-supernode-using-helm.md>

.. |helm_install_link| replace:: ``helm install``

.. |helm_uninstall_link| replace:: ``helm uninstall``

.. _helm_install_link: https://helm.sh/docs/helm/helm_install/

.. _helm_uninstall_link: https://helm.sh/docs/helm/helm_uninstall/
