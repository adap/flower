.. |authenticate_supernodes| replace:: Authenticate Supernodes

.. _authenticate_supernodes: how-to-authenticate-supernodes.html

.. |enable_tls_connections| replace:: Enable TLS Connections

.. _enable_tls_connections: how-to-enable-tls-connections.html

.. |flower_architecture_link| replace:: Flower Architecture

.. _flower_architecture_link: explanation-flower-architecture.html

Run Flower on Azure
===================

.. note::

    There are many ways to deploy Flower on Microst Azure. The instructions provided in
    this guide is just a basic walkthrough, step-by-step guide on how to quickly setup
    and run a Flower application on a Federated Learning environment on Microst Azure.

In this how-to guide, we want to create a Federated Learning environment on Microst
Azure using three Virtual Machines (VMs). From the three machines, one machine will be
used as the Federation server and two as the Federation clients. Our goal is to create a
Flower federation on Microst Azure where we can run Flower apps from our local machine,
e.g., laptop.

On the Federation server VM we will deploy the long-running Flower server
(``SuperLink``) and on the two Federation client VMs we will deploy the long-running
Flower client (``SuperNode``). For more details For more details regarding the
``SuperLink`` and ``SuperNode`` concepts, please see the |flower_architecture_link|_ .

Azure VMs
---------

First we need to create the three VMs configure their Python environments, and inbound
networking rules to allow cross-VM communication.

VM Create
~~~~~~~~~

Assuming we are already inside the Microst Azure portal, we navigate to the ``Create``
page and we select ``Azure virtual machine``. In the new page that opens up, for each VM
we fill in the following fields:

- **Virtual machine name**: for instance, for the server machine we can use
  ``flower-server`` and for the clients, ``flower-client-1``, ``flower-client-2``,
  respectively.
- **Image**: for this guide, we will use ``Ubuntu Server 24.04 - x64 Gen2 LTS``
- **Size**: for this guide, we will use ``Standard_D2s_v3 - 2 vcpus, 8GiB memory``

.. tip::

    For resource group, we can create a new group and assign it to all VMs.

When each VM instance has been created the portal will allow you to download the public
key (.pem) of each instance. Make sure you save this key in safe place and change its
permissions to user read only, i.e., run the ``chmod 400 <PATH_TO_PEM_FILE>`` command
for every .pem file.

Once all three VMs are created then navigate to the overview page where all three VMs
are listed and open every other VM, and copy its Public IP address. Using the Public IP
address and the public key (after changing the permissions), login to the instances from
our local machine by running the following command (by default Azure creates the
``azureuser``):

.. code-block:: bash

    ssh -i <PATH_TO_PEM_FILE> azureuser@<PUBLIC_IP>

VM Networking
~~~~~~~~~~~~~

During the execution of the Flower application, the server VM (``SuperLink``) will be
responsible to orchestrate the execution of the application across the client VMs
(``SuperNode``). When the SuperLink server starts, by default, it listens to the
following ports: ``{9092, 9093}``. Port `9092` is used to communicate with the
Federation clients (``SuperNode``) and port ``9093`` to receive and execute Flower
applications.

Therefore, to enable this communication we need to allow inbound traffic to the server
VM instance. To achieve this, we need to navigate to the Networking page of the server
VM in the Microsoft Azure portal. There, we will click the ``Add inbound port rule``. In
the new window that pops up we perform the following steps:

- **Source**: we select ``IP Addresses``
- **Source IP addresses/CIDR ranges**: we add the two Public IPs of our client VMs,
  separated by comma
- **Destination**: ``Any``
- **Service**: ``custom``
- **Destination port ranges**: ``9092``
- **Protocol**: ``TCP``

The rest of the fields can be left at their default values.

Finally, we need to also open port 9093 to allow receiving and executing incoming
application requests. To enable this we just need to repeat the steps above, i.e.,
create a new inbound rule, where for port range we assign port 9093. If we already know
the Public IP from which our local machine (e.g., laptop) will be submitting
applications to the Azure cluster, then we just need to specify the Source IP
address/CIDR range. However, if we want to keep the port widely open we simply need to
change source to ``Any``.

To be more precise, if we know the Public IP of our machine, then we need to change the
following:

- **Source IP addresses/CIDR ranges**: add machine's Public IP
- **Destination port ranges**: ``9093``

Otherwise, we just need to change the following:

- **Source**: ``Any``
- **Destination port ranges**: ``9093``

Flower Environment
------------------

Configuration
~~~~~~~~~~~~~

Assuming we have been able to login to each VM, we need to make sure that the Flower
environment is installed. To accomplish this, we first install Python in each machine
and then create a path, referred to as ``<INSTANCE_PATH>`` in the code snippet below,
which we will use to create inside a virtual environment, (e.g.,
``<INSTANCE_PATH>/flower_venv-3.11``) and the Flower application project.

To create the Flower application project, we need to run ``flwr run`` and then give a
name to the project, e.g., ``flwr_azure_test``, give the name of the author and finally
select the type of the Flower Framework we want to run, e.g., ``numpy``.

.. code-block:: bash

    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install python3.11
    sudo apt install python3.11-venv
    python3.11 -m venv <INSTANCE_PATH>/flower-venv-3.11
    source activate <INSTANCE_PATH>/flower-venv-3.11/bin/activate
    pip install flwr
    cd <INSTANCE_PATH>
    flwr new  # <-- give project name, user name, and select framework type
    # Edit the files under the ``flwr_azure_test`` project, e.g., define server and client logic.

Server Initialization
~~~~~~~~~~~~~~~~~~~~~

After configuring the Flower environment, we can proceed by starting the long-running
processes at each VM instance. In particular we need to run the following commands,
first in the server (``SuperLink``) and then at each client (``SuperNode``); assuming
the virtual environment is enabled and we are inside the <INSTANCE_PATH> we created
earlier.

.. code-block:: bash

    # Server VM (SuperLink)
    flower-superlink --insecure

    # Client-1 VM (SuperNode-1)
    flower-supernode \
      --insecure \
      --clientappio-api-address="0.0.0.0:9094" \  # SuperNode listening port
      --superlink="SUPERLINK_PUBLIC_IP:9092"  # SuperLink public ip and port

    # Client-2 VM (SuperNode-2)
    flower-supernode \
      --insecure \
      --clientappio-api-address="0.0.0.0:9095" \  # SuperNode listening port
      --superlink="SUPERLINK_PUBLIC_IP:9092"  # SuperLink public ip and port

Run Flower App
~~~~~~~~~~~~~~

Finally, after all running servers have been initialized on the Microsoft Azure cluster,
in our local machine, we first need to install Flower and can create a project with a
similar structure as the one we have in the server and the clients, or copy the project
structure from one of them. Once we have the project locally, we can open the
``pyproject.toml`` file, and then add the following sections:

.. code-block:: python

    [tool.flwr.federations]
    default = "my_federation"  # replaced the default value with "my_federation"

    [tool.flwr.federations.my_federation]  # replaced name with "my_federation"
    address = "SUPERLINK_PUBLIC_IP:9093"  # Address of the SuperLink Exec API
    insecure = true

Then from our local machine we need to run ``flwr run . my_federation``.

Additional Resources
--------------------

If we want to enable authentication and encrypted communication throughout the lifecycle
of the Flower application, please have a look at the following resources:

- |authenticate_supernodes|_
- |enable_tls_connections|_
