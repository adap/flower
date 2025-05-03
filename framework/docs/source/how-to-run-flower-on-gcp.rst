

Run Flower on GCP
=================

A step-by-step guide to learn how to create, deploy and run a Flower application on the Google Cloud Platform (GCP)
using the Google Kubernetes Engine (GKE).

Create a GCP Cluster
--------------------
Here, we will discuss the necessary steps to create an account and a Kubernetes cluster in `GCP <https://console.cloud.google.com>`_ through the GCP interface. Before proceeding, please make sure you have created an account on `GCP <https://console.cloud.google.com>`_.

1. **Create GCP Project**: Once you have created the account, please create a new project, by pressing on the top button with the project name. This will open a new window from where you can press the ``NEW PROJECT`` button and create the new project and assign a name, e.g., ``flower-gcp``.

2. **Enable Kubernetes API**: After the project creation in the search bar at the top of the GCP page type ``Kubernetes Engine API``. This will
redirect you to the ``Kubernetes Engine API`` Product page. From there you need to select ``Enable``. After the you enable it you should see a green mark
in the ``Kubernetes Engine API`` saying ``API Enabled``.

3. **Create Kubernetes Cluster**: in the home page of the GCP project, under the ``Products`` section, look for the a tab that is called ``Create a Kubernetes Cluster``.
This will redirect you to a page where you will see an overview of the existing Kubernetes clusters. At the top of the page you should see a button called ``Create``.
By default, the Kubernetes clusters are deployed using the ``Autopilot`` mode. For the current guide, we will use the ``Autopilot`` mode; in an advanced version of this guide, we
shall deploy a cluster using the ``Standard`` mode.

4. **Configure Kubernetes Cluster**: in the page that is shown, we will assign a name to the new cluster, e.g., ``flower-numpy-example`` and we will select the region, e.g., ``us-central1``.
For the rest of the configuration settings, such as ``Cluster Tier``, ``Fleet Registration``, ``Networking``, and other settings we shall use the default values. Now, press the ``Create`` button.

.. note::

    Please wait for a couple of minutes until the cluster is ready and fully deployed.



Configure GCP & GKE CLI Tools
-----------------------------



Create a Google Artifact Repository
-----------------------------------

Configure Flower App Docker Images
----------------------------------

Create Images
~~~~~~~~~~~~~

Tag Images
~~~~~~~~~~

Push Images
~~~~~~~~~~~

Deploy Flower Application
--------------------------

Deploy Pods
~~~~~~~~~~~

Run Application
~~~~~~~~~~~~~~~


Helpful Commands
----------------
