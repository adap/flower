Run Flower on Multiple OpenShift Clusters
=========================================

In this guide, you will learn how to deploy Flower in multiple `Red Hat OpenShift (RHOS)
<https://www.redhat.com/en/technologies/cloud-computing/openshift>`_ application
platforms. This deployment pattern is useful when connecting multiple OpenShift clusters
deployed in environments that run critical workloads, such as in secure research
environments (also known as trusted research environments, or `secure data environments
<https://digital.nhs.uk/services/secure-data-environment-service>`_), datacenters,
high-performance computing clusters, or on-premises servers.

.. note::

    This guide assumes you have a working knowledge of OpenShift and have deployed
    Flower in a single OpenShift cluster before. If you are new to OpenShift or Flower,
    please refer to our guide on `how to run Flower on OpenShift
    <how-to-run-flower-on-red-hat-openshift.rst>`_ before proceeding with this
    multi-cluster deployment.

To connect multiple OpenShift clusters, we will use Red Hat Service Interconnect (RHSI).
Based on the open source `Skupper <https://skupper.io/>`_ project, RHSI simplifies
connections between sites. Applications and services, such as SuperLink and SuperNodes,
can communicate with each other using RHSI as if they were in the same network.

Skupper works by creating a site in each cluster, with a router (or routers) that
connect to other sites over the application network. The application network is the
network that connects multiple K8s clusters. So, the cluster where SuperLink is deployed
requires a Skupper site to accept links and route incoming traffic and the cluster where
SuperNode is deployed must have a Skupper site to join the application network and route
outgoing requests via the interconnect.

Pre-requisites
--------------

Install the Skupper CLI on your local system by following the instructions `on the
Skupper website
<https://skupper.io/docs/install/index.html#installing-the-skupper-cli>`_.

Create Red Hat OpenShift Clusters on AWS
----------------------------------------

For this guide, we will create two OpenShift clusters using the ``Red Hat OpenShift
Service on AWS (ROSA)``. Follow the steps in our companion guide `here
<how-to-run-flower-on-red-hat-openshift.rst#create-a-red-hat-openshift-cluster-on-aws>`_
on deploying the clusters. In this guide, we will assume that your clusters are deployed
in different AWS regions or availability zones.

Next, deploy SuperLink in one cluster and SuperNode in the other cluster. For reference,
you can follow the `deployment steps
<how-to-run-flower-on-red-hat-openshift.rst#deploy-flower-superlink-and-supernodes-on-openshift>`_
in our companion guide.

Install Red Hat Service Interconnect Operator
---------------------------------------------

In each OpenShift cluster, install the Red Hat Service Interconnect Operator from the
OperatorHub:

.. figure:: ../_static/images/rhos_install_service_interconnect_operator.png
    :align: center
    :width: 90%
    :alt: Red Hat Service Interconnect Operator

    Install Red Hat Service Interconnect Operator from OperatorHub.

Create Skupper Sites
--------------------

From your local system, you will now create a Skupper site in each OpenShift cluster and
connect the sites to form an application network.

First, set the namespace in your CLI by logging in to your first OpenShift cluster (the
one with SuperLink deployed):

.. code-block:: shell

    oc login --server=<your-openshift-api-endpoint> --web

If successful, you should see a message similar to this:

.. code-block:: shell

    Opening login URL in the default browser: [...]
    Login successful.

    You have access to 81 projects, the list has been suppressed. You can list all projects with 'oc projects'

    Using project "default".

Now, switch the project to the project name (i.e. namespace) where you deployed
SuperLink:

.. code-block:: shell

    oc project <your-namespace>

In our `previous guide <how-to-run-flower-on-red-hat-openshift>`_, we used the project
name ``flower-openshift-demo``, so let's do that:

.. code-block:: shell

    ➜ oc project flower-openshift-demo
    Now using project "flower-openshift-demo" on server "<your-openshift-api-endpoint>".

.. tip::

    If you are unsure of the project name, you can list all projects you have access to
    using the command ``oc projects``. You can also check that you are in the correct
    context by running ``oc whoami --show-context``.

With the correct namespace set, create a Skupper site in this cluster:

.. code-block:: shell

    skupper site create superlink-interconnect --enable-link-access

This command creates a Skupper site named ``superlink-interconnect`` and the
``--enable-link-access`` option enables external access for links *from* remote sites;
this option is necessary for the site where SuperLink is deployed so that SuperNodes in
other clusters can connect to it. You should see output similar to this:

.. code-block:: shell

    ➜ skupper site create superlink-interconnect --enable-link-access
    Waiting for status...
    Site "superlink-interconnect" is ready.

References
----------

To learn more about Red Hat Service Interconnect and Skupper concepts, please refer to
the following resources: - `Red Hat Service Interconnect
<https://www.redhat.com/en/technologies/cloud-computing/service-interconnect>`_ -
`Skupper concepts <https://skupperproject.github.io/refdog/concepts/>`_
