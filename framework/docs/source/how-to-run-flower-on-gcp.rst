:og:description: A step-by-step guide to learn how to create, deploy and run a Flower application on the Google Cloud Platform (GCP) using the Google Kubernetes Engine (GKE).
.. meta::
    :description: A step-by-step guide to learn how to create, deploy and run a Flower application on the Google Cloud Platform (GCP) using the Google Kubernetes Engine (GKE).

Run Flower on GCP
=================

A step-by-step guide to learn how to create, deploy and run a Flower application on the
Google Cloud Platform (GCP) using the Google Kubernetes Engine (GKE). The figure below
presents an overview of the architecture of the Flower components we will deploy on GCP
using GKE through the current guide.

.. figure:: ./_static/flower-gke-architecture.png
    :align: center
    :width: 600
    :alt: Running Flower on GCP using GKE Architecture
    :class: no-scaled-link

    Running Flower on GCP using GKE Architecture

Part of this guide has also been presented during the `Flower AI Summit 2025
<https://flower.ai/events/flower-ai-summit-2025/>`_, by Prashant Kulkarni, GenAI
Security Engineer at Google Cloud.

.. raw:: html

    <a href="https://www.youtube.com/watch?v=DoklGCdtrrc" target="_blank" style="display: block; text-align: center;">
        <img src="https://img.youtube.com/vi/DoklGCdtrrc/0.jpg" alt="Introduction" width="300"/>
    </a>

Create a Kubernetes Cluster
---------------------------

Here, we will discuss the necessary steps to create an account and a Kubernetes cluster
in `GCP <https://console.cloud.google.com>`_ through the GCP interface. Before
proceeding, please make sure you have created an account on `GCP
<https://console.cloud.google.com>`_.

1. **Create GCP Project**: Once you have created the account, please create a new
   project, by pressing on the top button with the project name. This will open a new
   window from where you can press the ``NEW PROJECT`` button and create the new project
   and assign a name, e.g., ``flower-gcp``.

2. **Enable Kubernetes API**: After the project creation in the search bar at the top of
the GCP page type ``Kubernetes Engine API``. This will redirect you to the ``Kubernetes
Engine API`` Product page. From there you need to select ``Enable``. After the you
enable it you should see a green mark in the ``Kubernetes Engine API`` saying ``API
Enabled``.

3. **Create Kubernetes Cluster**: in the home page of the GCP project, under the
``Products`` section, look for the a tab that is called ``Create a Kubernetes Cluster``.
This will redirect you to a page where you will see an overview of the existing
Kubernetes clusters. At the top of the page you should see a button called ``Create``.
By default, the Kubernetes clusters are deployed using the ``Autopilot`` mode. For the
current guide, we use the ``Autopilot`` mode; in an advanced version of this guide, we
shall deploy a cluster using the ``Standard`` mode.

4. **Configure Kubernetes Cluster**: in the page that is shown, we assign a name to the
new cluster, e.g., ``flower-numpy-example`` and we select the region, e.g.,
``us-central1``. For the rest of the configuration settings, such as ``Cluster Tier``,
``Fleet Registration``, ``Networking``, and other settings we use the default values.
Now, press the ``Create`` button.

.. note::

    Please wait for a couple of minutes until the cluster is ready and fully deployed.

Configure Google Cloud SDK
--------------------------

To start interacting with the our newly deployed Kubernetes cluster we need to configure
locally the Google Cloud SDK. The SDK will allow us to interact with the Google Cloud
Platform and in turn with out recently deployed Kubernetes cluster.

To install the Google Cloud SDK, we first need to install and configure the ``gcloud``
CLI:

.. code-block:: bash

    # macOS
    curl https://sdk.cloud.google.com | bash  # and then follow on-screen prompts

    # macOS (w/ Homebrew)
    brew install --cask google-cloud-sdk

    # Windows
    # Download the Windows installer from the Google Cloud SDK page
    # https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe
    # Run the .exe installer and follow the on-screen instructions

    # Once the package is installed, we initialize gcloud as follows:
    gcloud init  # initialize with gcloud init.
    source ~/.bashrc  # update PATH
    gcloud version  # verify installation

.. note::

    For more detailed installation instructions and for installing ``gcloud`` for
    different operating systems, please take look at the official ``gcloud`` CLI
    installation page https://cloud.google.com/sdk/docs/install

Once ``gcloud`` is installed we need to install ``kubectl``:

.. code-block:: bash

    gcloud components install kubectl
    kubectl version --client  # this will show the installed versions of the Client and Kustomize

Now you need to configure ``kubectl`` to point to the GKE cluster you created in the
previous steps by using the name of the cluster, e.g., ``flower-numpy-example``, and the
name of the region where the cluster was created:

.. code-block:: bash

    gcloud container clusters get-credentials flower-numpy-example --region us-central1

This will configure the required metadata and fetch the necessary credentials to allow
your local ``kubectl`` to communicate with the GKE cluster. To verify that ``kubectl``
was able to connect to the cluster and get the necessary information, you can run the
following command:

.. code-block:: bash

    kubectl config current-context  # this should return the Kubernetes cluster you connected

.. note::

    For more information on how ``kubectl`` works, please have a look at the following
    official quick-reference guide:
    https://kubernetes.io/docs/reference/kubectl/quick-reference/

Create a Google Artifact Repository
-----------------------------------

The Google Cloud Artifact Registry is a fully managed, scalable, and private service for
storing and managing software build artifacts and dependencies. Consequently, to run our
Flower application on the GKE cluster, we need to store the application's specific
Flower Docker images within the registry, i.e., ``ClientApp`` and ``ServerApp``, which
we discuss in the next section. This step is crucial as it enables the cluster, and
subsequently the pods, to download the built Docker images and deploy the necessary
Flower components.

There are two ways to create the required registry, one through the UI and another
through the CLI., below we discuss both approaches.

**Create through UI**:

- we need to go to the ``APIs & Services`` and then look for ``Library``.
- we search for ``Artifact Registry API`` and we enable the API (if it's not already
  enabled).
- once the ``Artifact Registry API`` is enabled, we navigate to the `Artifact Registry
  page <https://console.cloud.google.com/artifacts>`_ and we select ``Create
  Repository``.
- we enter the name of the new repository, e.g., ``flower-gcp-example-artifacts``, we
  then choose ``Docker`` as format, ``Standard`` as mode and we pick a location type and
  region, e.g., ``Region: us-central``.

For all the rest of the fields, such as ``Encryption``, ``Immutable image tags``,
``Cleanup Policies``, and ``Artifact Analysis`` we leave the default values. Finally, we
press ``Create``.

**Create through CLI**:

.. code-block:: bash

    # Enable the Artifact Registry API service
    gcloud services enable artifactregistry.googleapis.com

    # Create the repository
    # gcloud artifacts repositories create YOUR_REPOSITORY_NAME
    gcloud artifacts repositories create flower-gcp-example-artifacts \
    --repository-format=docker \
    --location=us-central1

    # Configure Docker to Authenticate with Artifact Registry:
    # gcloud auth configure-docker YOUR_REGION-docker.pkg.dev
    gcloud auth configure-docker us-central1-docker.pkg.dev  # we use us-central1 as our region

Configure Flower Application Docker Images
------------------------------------------

In order to proceed with this next step, first, we create a local Flower application,
and then create a dedicated Dockerfile for the ServerApp and the ClientApp Docker
images. Once we build the images, we tag them and push them to the newly created Google
registry. Most of the steps on how to build Docker images discussed below are based on
the `Flower Quickstart with Docker Tutorial
<https://flower.ai/docs/framework/docker/tutorial-quickstart-docker.html>`_.

.. note::

    We do not create a Dockerfile for the SuperLink or the SuperNode components, since
    we only need to use the default provided by the official `Flower DockerHub
    repository <https://hub.docker.com/u/flwr>`_.

We create the Flower NumPy application as follows:

.. code-block:: bash

    # flwr new YOUR_APP_NAME --framework YOUR_ML_FRAMEWORK --username YOUR_USERNAME
    flwr new flower-numpy-example --framework NumPy --username flower

Create Docker Images
~~~~~~~~~~~~~~~~~~~~

Once the application is created, we navigate inside the parent directory and create two
``Dockerfiles`` one for the ``ClientApp`` component, named ``clientapp.Dockerfile`` and
one for the ``ServerApp`` component, named as ``serverapp.Dockerfile``. We will use both
files to build locally the necessary Docker images.

.. dropdown:: clientapp.Dockerfile

    .. code-block:: bash

        # clientapp.Dockerfile
        ARG FLWR_VERSION
        FROM flwr/clientapp:${FLWR_VERSION}  # set the Flower version, e.g., 1.18.0

        WORKDIR /app

        COPY pyproject.toml .
        RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
            && python -m pip install -U --no-cache-dir .

        ENTRYPOINT ["flwr-clientapp"]

.. dropdown:: serverapp.Dockerfile

    .. code-block:: bash

        # serverapp.Dockerfile
        ARG FLWR_VERSION
        FROM flwr/serverapp:${FLWR_VERSION}  # set the Flower version, e.g., 1.18.0

        WORKDIR /app

        COPY pyproject.toml .
        RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
           && python -m pip install -U --no-cache-dir .

        ENTRYPOINT ["flwr-serverapp"]

Once we have created the required Dockerfiles, we build the Docker Images as follows:

.. important::

    - Depending on which Flower version you used to create the Flower application, make
      sure you use the same version while building the ``ClientApp`` and ``ServerApp``
      components, and either update the value of the ``FLWR_VERSION`` variable directly
      in the two Dockerfiles, or pass ``--build-arg FLWR_VERSION=<FLWR_VERSION>``
      argument as shown below.
    - Before running the commands below, make sure ``Docker`` is installed and it is up
      running. The ``--platform`` type is set to ``linux/amd64``, because when using the
      ``Autopilot`` mode, all ``Pods`` in the Kubernetes cluster (by default) are
      deployed with an ``amd64``-based architecture.

.. code-block:: bash

    # ServerApp
    # with build-arg
    docker build --build-arg FLWR_VERSION=1.18.0 --platform linux/amd64 -f serverapp.Dockerfile -t flower_numpy_example_serverapp:0.0.1 .
    # without build-arg
    docker build --platform linux/amd64 -f serverapp.Dockerfile -t flower_numpy_example_serverapp:0.0.1 .

    # ClientApp
    # with build-arg
    docker build --build-arg FLWR_VERSION=1.18.0 --platform linux/amd64 -f clientapp.Dockerfile -t flower_numpy_example_clientapp:0.0.1 .
    # without build-arg
    docker build --platform linux/amd64 -f clientapp.Dockerfile -t flower_numpy_example_clientapp:0.0.1 .

Tag Docker Images
~~~~~~~~~~~~~~~~~

Before we are able to push our two newly locally created Docker images, we need to tag
them with the Google Artifact Registry repository name and image name we created during
the previous steps. If you have followed the earlier naming suggestions, then the
project ID is ``flower-gcp``, the repository name is ``flower-gcp-example-artifacts``,
the local Docker images names are ``flower_numpy_example_serverapp:0.0.1`` and
``flower_numpy_example_numpy:0.0.1``, and the region is ``us-central1``. Putting all
this together, the final commands you need to run to tag the ``ServerApp`` and
``ClientApp`` Docker images are:

.. code-block:: bash

    # docker tag YOUR_IMAGE_NAME YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/YOUR_REPOSITORY_NAME/YOUR_IMAGE_NAME:YOUR_TAG

    # ServerApp
    docker tag flower_numpy_example_serverapp:0.0.1 us-central1-docker.pkg.dev/flower-gcp/flower-gcp-example-artifacts/flower_numpy_example_serverapp:0.0.1

    # ClientApp
    docker tag flower_numpy_example_clientapp:0.0.1 us-central1-docker.pkg.dev/flower-gcp/flower-gcp-example-artifacts/flower_numpy_example_clientapp:0.0.1

Push Docker Images
~~~~~~~~~~~~~~~~~~

Once our images are tagged correctly, you can push them to your ``Artifact Registry``
repository using the ``docker push`` command with the tagged name:

.. code-block:: bash

    # docker push YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/YOUR_REPOSITORY_NAME/YOUR_IMAGE_NAME:YOUR_TAG

    # ServerApp
    docker push us-central1-docker.pkg.dev/flower-gcp/flower-gcp-example-artifacts/flower_numpy_example_serverapp:0.0.1

    # ClientApp
    docker push us-central1-docker.pkg.dev/flower-gcp/flower-gcp-example-artifacts/flower_numpy_example_clientapp:0.0.1

Deploy Flower Application
-------------------------

To be able to deploy our Flower application, the final step is to deploy our ``Pods`` on
the Kubernetes cluster.

In this step, we shall deploy six ``Pods``: 1x ``SuperLink``, 2x ``SuperNode``, 2x
``ClientApp``, and 1x ``ServerApp``. To achieve this, below we provide the definition of
the six ``yaml`` files that are necessary to deploy the ``Pods`` on the cluster and
which are passed to ``kubectl``, and a helper ``k8s-deploy.sh`` script, which will
deploy the ``Pods``. To define the Flower version for ``SuperLink`` and ``SuperNodes``,
you can either change directly the the value of the ``${FLWR_VERSION}`` within each
respective ``.yaml`` file or modify the value of the ``FLWR_VERSION=<FLWR_VERSION>``
directly in the helper ``k8s-deploy.sh`` script.

.. dropdown:: superlink-deployment.yaml

    .. code-block:: bash

        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: superlink
        spec:
          replicas: 1
          selector:
            matchLabels:
              app: superlink
          template:
            metadata:
              labels:
                app: superlink
            spec:
              containers:
              - name: superlink
                image: flwr/superlink:${FLWR_VERSION}  # set the Flower version, e.g., 1.18.0
                args:
                  - "--insecure"
                  - "--isolation"
                  - "process"
                ports:  # which ports to expose/available
                - containerPort: 9091
                - containerPort: 9092
                - containerPort: 9093
        ---
        apiVersion: v1
        kind: Service
        metadata:
          name: superlink-service
        spec:
          selector:
            app: superlink
          ports:  # like a dynamic IP routing table/mapping that routes traffic to the designated ports
          - protocol: TCP
            port: 9091   # Port for ServerApp connection
            targetPort: 9091  # the SuperLink container port
            name: superlink-serverappioapi
          - protocol: TCP
            port: 9092   # Port for SuperNode connection
            targetPort: 9092  # the SuperLink container port
            name: superlink-fleetapi
          - protocol: TCP
            port: 9093   # Port for Flower applications
            targetPort: 9093  # the SuperLink container port
            name: superlink-execapi
          type: LoadBalancer  # balances workload, makes the service publicly available

.. dropdown:: supernode-1-deployment.yaml

    .. code-block:: bash

        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: supernode-1
        spec:
          replicas: 1
          selector:
            matchLabels:
              app: supernode-1
          template:
            metadata:
              labels:
                app: supernode-1
            spec:
              containers:
              - name: supernode
                image: flwr/supernode:${FLWR_VERSION}  # set the Flower version, e.g., 1.18.0
                args:
                  - "--insecure"
                  - "--superlink"
                  - "superlink-service:9092"
                  - "--clientappio-api-address"
                  - "0.0.0.0:9094"
                  - "--isolation"
                  - "process"
                  - "--node-config"
                  - "partition-id=0 num-partitions=2"
                ports:
                - containerPort: 9094
        ---
        apiVersion: v1
        kind: Service
        metadata:
          name: supernode-1-service
        spec:
          selector:
            app: supernode-1
          ports:
          - protocol: TCP
            port: 9094
            targetPort: 9094

.. dropdown:: supernode-2-deployment.yaml

    .. code-block:: bash

        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: supernode-2
        spec:
          replicas: 1
          selector:
            matchLabels:
              app: supernode-2
          template:
            metadata:
              labels:
                app: supernode-2
            spec:
              containers:
              - name: supernode
                image: flwr/supernode:${FLWR_VERSION}  # set the Flower version, e.g., 1.18.0
                args:
                  - "--insecure"
                  - "--superlink"
                  - "superlink-service:9092"
                  - "--clientappio-api-address"
                  - "0.0.0.0:9094"
                  - "--isolation"
                  - "process"
                  - "--node-config"
                  - "partition-id=1 num-partitions=2"
                ports:
                - containerPort: 9094
        ---
        apiVersion: v1
        kind: Service
        metadata:
          name: supernode-2-service
        spec:
          selector:
            app: supernode-2
          ports:
          - protocol: TCP
            port: 9094
            targetPort: 9094

.. dropdown:: serverapp-1-deployment.yaml

    .. code-block:: bash

        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: serverapp
        spec:
          replicas: 1
          selector:
            matchLabels:
              app: serverapp
          template:
            metadata:
              labels:
                app: serverapp
            spec:
              containers:
              - name: serverapp
                image: us-central1-docker.pkg.dev/flower-gcp/flower-gcp-example-artifacts/flower_numpy_example_serverapp:0.0.1
                args:
                  - "--insecure"
                  - "--serverappio-api-address"
                  - "superlink-service:9091"

.. dropdown:: clientapp-1-deployment.yaml

    .. code-block:: bash

        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: clientapp-1
        spec:
          replicas: 1
          selector:
            matchLabels:
              app: clientapp-1
          template:
            metadata:
              labels:
                app: clientapp-1
            spec:
              containers:
              - name: clientapp
                image: us-central1-docker.pkg.dev/flower-gcp/flower-gcp-example-artifacts/flower_numpy_example_clientapp:0.0.1
                args:
                  - "--insecure"
                  - "--clientappio-api-address"
                  - "supernode-1-service:9094"

.. dropdown:: clientapp-2-deployment.yaml

    .. code-block:: bash

        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: clientapp-2
        spec:
          replicas: 1
          selector:
            matchLabels:
              app: clientapp-2
          template:
            metadata:
              labels:
                app: clientapp-2
            spec:
              containers:
              - name: clientapp
                image: us-central1-docker.pkg.dev/flower-gcp/flower-gcp-example-artifacts/flower_numpy_example_clientapp:0.0.1
                args:
                  - "--insecure"
                  - "--clientappio-api-address"
                  - "supernode-2-service:9094"

Once you have created the required files, you can use the following ``k8s-deploy.sh``
helper script to deploy all the ``Pods``.

.. important::

    Please note that you need to define the Flower version. The version needs to match
    the Flower version you used when you created the Flower Application.

.. dropdown:: k8s-deploy.sh

    .. code-block:: bash

        #! /bin/bash -l

        export FLWR_VERSION=<FLWR_VERSION>  # set the Flower version, e.g., 1.18.0

        # Change directory to the yaml files directory
        cd "$(dirname "${BASH_SOURCE[0]}")"

        envsubst < superlink-deployment.yaml > ./_rendered.yaml && \
        kubectl apply -f ./_rendered.yaml
        sleep 0.1

        envsubst < supernode-1-deployment.yaml > ./_rendered.yaml && \
        kubectl apply -f ./_rendered.yaml
        sleep 0.1

        envsubst < supernode-2-deployment.yaml > ./_rendered.yaml && \
        kubectl apply -f ./_rendered.yaml
        sleep 0.1

        kubectl apply -f ./serverapp-deployment.yaml
        sleep 0.1

        kubectl apply -f ./clientapp-1-deployment.yaml
        sleep 0.1

        kubectl apply -f ./clientapp-2-deployment.yaml
        sleep 0.1

To see that your ``Pods`` are deployed, please go to the ``Navigation Menu`` on the
Google Console, select ``Kubernetes Engine`` and then the ``Workloads`` page. The new
window that appears will show the status of the pods under deployment.

.. caution::

    Please wait for a couple of minutes (3' to 5' minutes should be enough) before the
    ``Pods`` are up and running. While ``Pods`` resources are being provisioned, some
    warnings are expected.

Run Flower Application
----------------------

Once all ``Pods`` are up and running, we need to get the ``EXTERNAL_IP`` of the
``superlink-service`` and point our Flower application to use the Kubernetes cluster to
submit and execute the job.

To get the ``EXTERNAL-IP`` of the ``superlink-service`` we run the following command,
which will show the ``NAME``, ``TYPE``, ``CLUSTER-IP``, ``EXTERNAL-IP`` and ``PORTS`` of
the service:

.. code-block:: bash

    kubectl get service superlink-service

After we get the ``EXTERNAL-IP`` , we go to the directory of the Flower example, we open
the ``pyproject.toml`` and then add the following section at the end of the file:

.. code-block:: bash

    [tool.flwr.federations.gcp-deployment]
    address = "<EXTERNAL_IP>:9093" # replace the EXTERNAL_IP with the correct value
    insecure = true

Then we can execute the example on the GCP cluster by running:

.. code-block:: bash

    flwr run . gcp-deployment --stream

If the job is successfully submitted, and executed, then in your console you should see
the ``fit`` and ``evaluate`` configuration and execution execution per round, and in the
end a ``Summary`` of the performance per round.

.. note::

    Please note that if you terminate or shut down the cluster, and create a new one,
    the value of the ``EXTERNAL_IP`` changes. In that case, you will have to update the
    ``pyproject.toml``.
