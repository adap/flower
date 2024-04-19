Flower Next
===========

Infrastructure Layer
--------------------
[graphic-outlining-infrastructure-layer]

Federated learning relies on a system that relays messages between all involved applications during training. This backbone system handles tasks like sending and receiving messages, syncing with federation nodes, and storing messages temporarily. But how do we ensure this backbone fits different situations? That's where SuperLink and SuperNode step in! We call these the infrastructure layer.

SuperLink
~~~~~~~~~
The SuperLink relays messages in a Flower federated learning system. It's like a hub that receives the model and training instructions from the server, then passes them along to the training nodes.

More concretely, the SuperLink relays messages between the SuperNodes and the ServerApp. Let's give an example for a federated learning workflow: a SuperLink receives a model to be federated in a message from the ServerApp. Selected SuperNodes then pull that message from the SuperLink and respectively process their messages by running their ClientApps. In this case, the SuperNodes launch their ClientApps to train a model each on their local data. Once trainings are complete, the ClientApps return their models to the SuperNodes, which in turn relay messages via the SuperLink back to the ServerApp for aggregation.

(Don't worry about the definitions of the ServerApp, ClientApp, and Message for now. We will explain them [below](#application-layer).)

As the main relay hub, the SuperLink is always running in the background, ready to handle any communication needs. 

SuperNode
~~~~~~~~~
Just like SuperLink, the SuperNode continuously runs in the background as a lightweight service. It runs where the data is gathered, like on smartphones, IoT devices, or servers belonging to organizations. All connected SuperNodes check in with the SuperLink regularly. They pull messages (that were first pushed by a ServerApp) from the SuperLink, process the messages by launching a ClientApp, and then push the results back to the SuperLink.

As a simple analogy, imagine SuperLink as the main internet router. It gets data packets from a central server and sends them to connected devices. Each SuperNode acts like a device, talking to SuperLink for data, processing it, and sending back results.

Together, SuperLink and SuperNodes make up the infrastructure layer of a Flower federated learning system.

Application Layer
-----------------
[graphic-outlining-application-layer]

On the application layer, we have the ServerApp and ClientApp. These are essentially applications or packaged code that runs, you guessed it, on the server and client, respectively.

ServerApp
~~~~~~~~~
Let's start with the [ServerApp][serverapplink]. Remember from our previous tutorial that only a handful of connected nodes are involved in training? The ServerApp plays a crucial role in this. It's responsible for sampling SuperNodes that are connected to the SuperLink, pushing messages to the SuperLink and pulling messages from it. It would normally process messages that get pulled (e.g. when performing model aggregation). What's neat is that the ServerApp is ephemeral — it's triggered just once for a complete training cycle.

ClientApp
~~~~~~~~~
Now, onto its counterpart, the [ClientApp][clientapplink]. Like the ServerApp, the ClientApp is ephemeral - it is spun up on-demand by the SuperNode to process a message (sent by the ServerApp). When the ClientApp is launched, it receives a message from the SuperNode, executes the instructions in the message, returns results back to the SuperNode, and finally terminates. 

..
    <div class="alert alert-info">

    Note

    In the coming weeks, we will introduce the concept of multi-app support. This means that multiple ClientApps can be connected to a single SuperNode. This allows multiple users of the same federation to execute different tasks on the same SuperNode, bringing greater freedom for building and using task-specific apps, all while using the same infrastructure! 

    </div>

Others
------
Messages
~~~~~~~~
[Message][messagelink]s is a Python dataclass that Flower uses to carry information between ServerApp and ClientApp. This information can be a model the ServerApp wants to federate, metrics the ClientApp is pushing back to the ServerApp via the SuperLink, and anything in between. The design of Messages and how they are handled by Flower ensures that a Message sent by the ServerApp looks exactly the same when received by the ClientApp (and vice versa). This ensures a more unified and smoother developer experience for you!

Context
~~~~~~~
[Context][contextlink] is another useful Python dataclass that we introduced in Flower Next. At a high level, it carries the record and messages for a specific execution of `ServerApp`, i.e. a run. Each time a SuperNode runs a ClientApp in the same run, the same Context object is exposed to the ClientApp, allowing the ClientApp to persist throughout the duration of the run.

Conclusion
----------
To wrap up, you've learnt the essential components of federated learning with Flower Next, divided neatly into infrastructure and application layers.

At the infrastructure layer, we've the backbone: the SuperLink and SuperNode, ensuring standardized and persistent communication between nodes. On the application layer, we've seen the ServerApp and ClientApp in action, handling tasks on the server and client sides, respectively. The benefit of this setup lies in decoupling—data scientists and ML researchers can focus on building and using the apps while making use of pre-existing infrastructure. Under the hood, Messages and Context standardize the mechanisms of relaying and persisting information between ServerApp and ClientApps. It's a win-win scenario, enabling smoother development experience and flexibility to experiment with federated learning systems.

.. 
    [clientapp_link]: ref-api/flwr.client.ClientApp.rst
    [serverapp_link]: ref-api/flwr.server.ServerApp.rst
    [builtinmods_link]: how-to-use-built-in-mods.rst
    [message_link]: ref-api/flwr.common.Message.rst
    [context_link]: ref-api/flwr.common.Context.rst