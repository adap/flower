########################
 Flower Architecture
########################

This page explains the decoupling concept in the Flower framework and
how it benefits Flower users.

**********************
 Infrastructure layer
**********************

Federated learning typically relies on a system that relays messages
during training. Beyond sending and receiving messages, this backbone
system may also handle a variety of other tasks that are non-learning
related, like syncing with federation nodes or temporarily storing
messages. However, as the range and complexity of applications
benefiting from federated learning grows, reliability and
reproducibility of such a system become key challenges. These challenges
are especially pertinent to users wanting to deploy their federated
learning workflows to production.

With Flower, we set out to address these crucial challenges by
decoupling and standardizing the infrastructure layer for federated
learning (see the following diagram).

.. figure:: ./_static/flower-infrastructure.png
   :align: center
   :width: 800
   :alt: Flower infrastructure layer
   :class: no-scaled-link

   Flower infrastructure layer consisting of the SuperExec, SuperLink, and SuperNodes.

This layer is dedicated to tasks such as transmitting data and
maintaining connections. It is separate from the application layer (more
on that below) which is the user-interaction layer [#f1]_. Because of
this decoupling, the Flower federated learning system becomes truly
infrastructure agnostic, meaning that Flower users can seamlessly move
projects between simulation and deployment without code changes.

.. important::

   In Flower, the infrastructure layer consists of the SuperExec,
   SuperLink, and SuperNode.

SuperExec
=========

The SuperExec the first long-running service in the Flower federated
learning system that receives instructions from the user. Using the
`Flower CLI <ref-api-cli.html#flwr-cli>`_,
users send instruction to launch a federated learning training using
``flwr run``. The SuperExec then launches the ``ServerApp`` which
contain the code for performing server-side processes such as
aggregating model weights.

SuperLink
=========

The SuperLink relays messages in a Flower federated learning system. It
acts like a hub that receives the model and training instructions from
the server, then passes them along to the training nodes.

More concretely, the SuperLink relays |message_link|_\s between the
SuperNodes and the |serverapp_link|_. Let's look at an example for a
federated learning workflow: a SuperLink receives a model to be
federated and a corresponding set of instructions in the form of a
``Message`` from the ``ServerApp``. Selected SuperNodes that a
participating in the federated learning round then pull that ``Message``
from the SuperLink and respectively process their ``Message``\s by
running their |clientapp_link|_\s. In this case, the SuperNodes launch
their ``ClientApp``\s to train a local model on each of their local
data. Once trainings are complete, the ``ClientApp``\s return their
models and any corresponding metrics to the SuperNodes also in a
``Message`` format. In turn, each SuperNode relay their ``Message`` via
the SuperLink back to the ``ServerApp`` for aggregation.

..
   TODO: Add section labels where appropriate: https://docs.readthedocs.io/en/stable/guides/cross-referencing-with-sphinx.html#automatically-label-sections

In the "Application Layer" section, we will define what we mean by the
``ServerApp``, ``ClientApp``, and ``Message``. For now, it is important
to know that as the main relay hub, the SuperLink is always running in
the background to handle any communication needs.

SuperNode
=========

The SuperNode is the next component of the infrastructure layer. Just
like SuperLink, the SuperNode continuously runs in the background. It
can run where the data is gathered, like on smartphones, IoT devices, or
servers belonging to organizations. It can also be run independently on
a dedicated server as a lightweight service, separate from where the
``ClientApp`` will be executed, for example on physical or virtual
compute with larger CPU/GPU resources. All connected SuperNodes ping the
SuperLink regularly. When a ``Message`` is available for the SuperNode,
it will pull the ``Message`` (that wwas first pushed by a ``ServerApp``)
from the SuperLink, process the ``Message`` by launching a
``ClientApp``, and then push the results back to the SuperLink.

Together, SuperLink and SuperNodes make up the infrastructure layer of a
Flower federated learning system.

*******************
 Application Layer
*******************

On the application layer, we have the |serverapp_link|_ and
|clientapp_link|_. These are the applications - or packaged code - that
runs on the server and clients, respectively. Flower users need only to
interact with the ``ServerApp`` to start tasks on federated nodes, such
as initiating federated learning. ``ClientApp``\s will be launched on
demand by the long-running SuperNodes. The graphic below illustrates the
workflow.

.. figure:: ./_static/flower-infrastructure-applications.png
   :align: center
   :width: 800
   :alt: Flower infrastructure and application layers
   :class: no-scaled-link

   Flower application layer (``ServerApp`` and ``ClientApp``\s) interacting with the infrastructure layer.

ServerApp
=========

Let's start with the |serverapp_link|_. Typically, in federated
learning, only a handful of connected nodes are involved in training.
The ``ServerApp`` plays a crucial role in this. It is responsible for
sampling SuperNodes that are connected to the SuperLink, pushing
messages to the SuperLink, and pulling messages from it. It would
normally process messages that get pulled, for example, when performing
model aggregation. The ``ServerApp`` is also ephemeral, meaning that it
is temporarily spun up and executed for a task, for instance one
complete federated learning run. This allows an on-demand utilization of
server-side compute resources.

ClientApp
=========

Now, onto its counterpart, the |clientapp_link|_. Like the
``ServerApp``, the ``ClientApp`` is ephemeral - it is launched on-demand
by the SuperNode to process a ``Message`` (sent by the ``ServerApp``).
When the ``ClientApp`` is launched, it receives a message from the
SuperNode, executes the instructions in the message, returns results
back to the SuperNode, and finally terminates.

.. note::

   As of Flower 1.11, we have introduced the concept of multi-app
   support. This means that multiple ``ClientApp``\s can be connected to
   and launched from a single SuperNode. This allows multiple users of
   the same federation to execute different tasks on the same SuperNode,
   bringing greater freedom for building and using task-specific apps,
   all while using the same infrastructure.

The reason for employing the mechanisms for launching apps is to enable
code reusability: In previous versions (<1.8), clients are inherently
linked to a server for an FL project and additional code needs to be
written to move a simulation-based FL setup to a real-world deployment.
This can sometimes prohibitively impede the speed of deploying
production-grade federated learning workflows. Now, with the decoupling
of the layers in Flower, the same ``ServerApp`` and ``ClientApp`` code
can be easily run in a simulation setup or deployment setup *without any
code changes*.

**********************
 Information Exchange
**********************

``Message``
===========

|message_link|_ is an object that the Flower framework uses to carry
information between ``ServerApp`` and ``ClientApp``. This information
can be a model the ``ServerApp`` wants to federate or metrics the
``ClientApp`` is pushing back to the ``ServerApp`` via the SuperLink. In
fact, we have designed the ``Message`` class to be sufficiently generic
such that it can carry any datatypes that is required for any federated
learning workflow. Since only a ``Messsage`` type object is communicated
within the workflow, you only need to work with the ``Message`` object,
thereby ensuring a overall more unified and smoother user experience.

``Context``
===========

|context_link|_ is another useful object that we introduced in the
Flower framework. For every execution of ``ServerApp``, i.e. a run, the
``Context`` carries any record and messages that are unique to the run.
Each time a SuperNode runs a ``ClientApp`` within the run, the same
``Context`` object is made available to the ``ClientApp``, allowing the
``ClientApp`` to process persistent data throughout the duration of the
run. ``Context`` can be used as a temporary buffer for executing
additional tasks, like |mods_link|_.

*********
 Summary
*********

In this explainer, you've learnt the essential components of federated
learning with Flower, divided neatly into infrastructure and application
layers.

At the infrastructure layer, we've the backbone: the SuperLink and
SuperNode, ensuring standardized and persistent communication between
nodes. On the application layer, we've seen the ``ServerApp`` and
``ClientApp`` in action, handling tasks on the server and client sides,
respectively.

The benefit of this setup lies in decoupling—data scientists and
researchers can focus on building and using the apps while making use of
pre-existing infrastructure. Importantly, users can seamlessly move
projects between simulation and real-world setting without code changes,
thereby easily making meaningful progress in production based on the
success on simulated problems. Under the hood, ``Message`` and
``Context`` objects standardize the mechanisms of relaying and
persisting information between ``ServerApp`` and ``ClientApp``\s. It's a
win-win scenario, enabling smoother development experience and
flexibility to experiment and build federated learning systems.

.. admonition:: Important
   :class: important

   As we continuously enhance Flower at a rapid pace, we'll be
   periodically updating this explainer document. Feel free to share any
   feedback with us!

.. rubric:: Footnotes

.. [#f1]

   This concept of layers is broadly based on the `Open Systems
   Interconnection (OSI) model <https://en.wikipedia.org/wiki/OSI_model>`_.

.. |clientapp_link| replace::

   ``ClientApp``

.. |serverapp_link| replace::

   ``ServerApp``

.. |message_link| replace::

   ``Message``

.. |context_link| replace::

   ``Context``

.. |mods_link| replace::

   ``mods``

.. _clientapp_link: ref-api/flwr.client.ClientApp.html

.. _context_link: ref-api/flwr.common.Context.html

.. _message_link: ref-api/flwr.common.Message.html

.. _mods_link: how-to-use-built-in-mods.html

.. _serverapp_link: ref-api/flwr.server.ServerApp.html
