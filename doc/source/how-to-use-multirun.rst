Use Multi-run
=============

In federated learning projects, we may encounter situations where we need to run more than one federated learning server task. Executing such tasks sequentially is possible although not time- and resource-efficient. In such cases, we can leverage the *multi-run* feature of the Flower framework!

Multi-run, a.k.a. concurrent run, is a feature where more than one :code:`ServerApp` can be launched in parallel to run tasks on nodes in the federation. In this way, each :code:`ServerApp` is no longer bound to the same :code:`ClientApp`. This dramatically increases the flexibility of federated learning workloads since we can now launch different :code:`ServerApps` (or even multiple instances of the same :code:`ServerApp`) for the same :code:`ClientApp`, while sharing a common infrastructure under the hood. More crucially, workloads are isolated for each launched :code:`ServerApp`. 

With this concept in mind, let’s walk through an example of how we can use the Flower multi-run feature.

Setup ClientApp
---------------

First, we setup the following :code:`ClientApp` structure in a module :code:`client.py`:

.. code-block:: python

    # file: `client.py`

    from flwr.client import ClientApp
    from flwr.common import Message, Context, log
    from logging import INFO

    # Create a ClientApp (Run via `flower-client-app client:app`)
    app = ClientApp()


    @app.train()
    def train(msg: Message, ctx: Context):
        log(INFO, "`train` not implemented, echoing original message")
        return msg.create_reply(msg.content)

Note that we have intentionally written a minimalistic :code:`ClientApp` containing only one :code:`train()` function which is initialized with :code:`@app.train()`. (This is the :code:`ClientApp` function that is executed when a message of type :code:`MessageType.TRAIN` is pushed by the :code:`ServerApp`). In federated learning projects, this :code:`train()` function will typically contain the full training workflow, e.g. instantiating a model, loading data, getting global model from server, getting the training configs from the server, etc … . 

Now, we launch 5 :code:`SuperNodes` with one :code:`SuperNode` linked to one :code:`ClientApp`. An analogy to the 5 nodes is 5 hospitals, which respectively have 1 :code:`ClientApp` each. In 5 separate terminals, we run the following:

.. code-block:: shell

    $ flower-client-app client:app --insecure

.. admonition:: Note
    :class: important

    In this example, the :code:`--insecure` command line argument starts Flower without HTTPS and is only used for prototyping. To run with HTTPS, we instead use the argument :code:`--certificates` and pass the paths to the certificates. Please refer to `Flower CLI reference <ref-api-cli.html>`_ for implementation details.

Setup ServerApp
---------------

Now for the :code:`ServerApp`. First, in a file :code:`server.py`, we import the necessary libraries, create a :code:`ServerApp`, and initialize :code:`main()` with :code:`@app.main()`:

.. code-block:: python

    # file: `server.py`

    import random
    import time
    from logging import INFO
    from typing import List

    from flwr.common import Context, Message, MessageType, RecordSet, log
    from flwr.server import Driver, ServerApp

    app = ServerApp()


    @app.main()
    def main(driver: Driver, context: Context) -> None:
        num_nodes_per_round = 2
        num_rounds = 3

        # [...]

Inside :code:`main()`, we loop over all :code:`num_rounds` (each loop emulating one federated learning round on the local hospital data in the example above). In each loop, we implement the steps to select nodes, push messages to nodes, and pull messages from nodes:

.. code-block:: python

        # [...]		
        # List of sampled node IDs in this round
        node_ids: List[int] = []

        while True:
            all_node_ids = driver.get_node_ids()
            if len(all_node_ids) >= num_nodes_per_round:
                # sample client nodes
                node_ids = random.sample(all_node_ids, num_nodes_per_round)
                break
            time.sleep(1)

        # Create a RecordSet
        recordset = RecordSet()

				# Create messages using the RecordSet
        messages = []
        for node_id in node_ids:
            message = driver.create_message(
                content=recordset,
                message_type=MessageType.TRAIN,
                dst_node_id=node_id,
                group_id=str(server_round),
            )
            messages.append(message)

        # Push messages
        message_ids = driver.push_messages(messages)

        # Pull messages
        message_ids = [message_id for message_id in message_ids if message_id != ""]
        all_replies: list[Message] = []
        while True:
            replies = driver.pull_messages(message_ids=message_ids)
            log(INFO, f"Pulled {len(replies)} result(s)")
            all_replies += replies
            if len(all_replies) == len(message_ids):
                break
            time.sleep(1)

        # Ignore messages with Error
        all_replies = [msg for msg in all_replies if msg.has_content()]

        # Print connected node IDs. Each connected node ID is stored
        # in the `metadata.src_node_id` attribute.
        src_node_ids = [reply.metadata.src_node_id for reply in all_replies]
        log(INFO, f"Received replies from node IDs: {src_node_ids}")

As you can tell, the steps above are relatively straightforward and in fact is similar to the code contained in the how-to guide for using Flower framework’s Driver API. The only nuanced differences here is that (i) we are not training any PyTorch models, and (ii) we implemented a :code:`time.sleep(1)` when pulling the messages to emulate an elapsed time due to a task being executed on the node. On the last line, we print the two connected node IDs for each round and for each :code:`ServerApp` .

Launch multi-run
----------------

Now to see multi-run in action! We simultaneously launch the :code:`ServerApp` in two new separate terminals using the following command:

.. code-block:: shell

    flower-server-app server:app --insecure

In the first terminal, you will see the following output:

.. code-block:: shell

    [...]
    INFO :      Starting server round 1
    INFO :      Pulled 0 result(s)
    INFO :      Pulled 0 result(s)
    INFO :      Pulled 0 result(s)
    INFO :      Pulled 2 result(s)
    INFO :      Received replies from node IDs: [-1465525346594594034, 8810599964921873562]
    INFO :      Starting server round 2
    INFO :      Pulled 0 result(s)
    INFO :      Pulled 0 result(s)
    INFO :      Pulled 0 result(s)
    INFO :      Pulled 2 result(s)
    INFO :      Received replies from node IDs: [-1465525346594594034, 8810599964921873562]
    INFO :      Starting server round 3
    INFO :      Pulled 0 result(s)
    INFO :      Pulled 0 result(s)
    INFO :      Pulled 1 result(s)
    INFO :      Pulled 1 result(s)
    INFO :      Received replies from node IDs: [-1038433217907319669, -623142144126604011]
    INFO :      Multi-run example complete!

And in the second terminal, we see the following output:

.. code-block:: shell

    [...]
    INFO :      Starting server round 1
    INFO :      Pulled 0 result(s)
    INFO :      Pulled 2 result(s)
    INFO :      Received replies from node IDs: [-623142144126604011, 8810599964921873562]
    INFO :      Starting server round 2
    INFO :      Pulled 0 result(s)
    INFO :      Pulled 0 result(s)
    INFO :      Pulled 0 result(s)
    INFO :      Pulled 2 result(s)
    INFO :      Received replies from node IDs: [-623142144126604011, -1465525346594594034]
    INFO :      Starting server round 3
    INFO :      Pulled 0 result(s)
    INFO :      Pulled 0 result(s)
    INFO :      Pulled 1 result(s)
    INFO :      Pulled 1 result(s)
    INFO :      Received replies from node IDs: [-1038433217907319669, -1465525346594594034]
    INFO :      Multi-run example complete!

Congratulations! You have successfully executed :code:`ServerApps` in multi-run mode, using the same :code:`server.py` and :code:`client.py` modules. Note that the node IDs may differ from one round to another between the two :code:`ServerApps`. Under the hood, this translates to the server sending messages to the first two available connected nodes.