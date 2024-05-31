Use Driver API
==============

This guide explains how you can use Flower framework's |driverapi_link|_ API to customize the :code:`ServerApp` for *any* server workloads in your federated learning system. The Driver APIs are a specific set of server-side APIs that provide a robust way to communicate with a :code:`SuperNode` in a Flower federated learning system. They include methods to:

* `Select nodes`_
* `Push messages to nodes`_
* `Pull messages from nodes`_

..
    Generate link text as literal.

.. |driverapi_link| replace:: ``Driver()``
.. |recordset_link| replace:: ``RecordSet()``
.. |flower_serverapp_link| replace:: ``flower-server-app``
.. _driverapi_link: ref-api/flwr.server.Driver.html
.. _recordset_link: ref-api/flwr.common.RecordSet.html
.. _flower_serverapp_link: ref-api-cli.html#flower-server-app

Setup
-----

Before diving into the :code:`Driver` API, we need to setup your :code:`server.py` module. In a file :code:`server.py`, import :code:`flwr` and create a :code:`ServerApp`:

.. code-block:: python

    import flwr
    from flwr.server import ServerApp, Driver
    from flwr.common import Context, log

    # Run via `flower-server-app server:app`
    app = ServerApp()

Next, we initialize the :code:`main()` function using :code:`@app.main()`. Within :code:`main()`, we define the workload that we want. For example, in federated learning workloads, we define a number of server rounds (:code:`server_round`) and loop over them throughout training:

.. code-block:: python

    # File: server.py

    @app.main()
    def main(driver: Driver, context: Context):
        num_rounds = 3
    
        for server_round in range(num_rounds):
            print(f"Commencing server round {server_round + 1}")
            
            # Do something using Driver APIs

Now that we have completed the setup, let’s walk through how we can use the Driver APIs for a federated learning example.


Select nodes
------------

The first stage of a federated learning workflow is to select client nodes. For simplicity, we will refer to client nodes as nodes throughout this guide. 

To select nodes in each server round, we run :code:`Driver.get_node_ids()` to retrieve all connected node IDs. Occasionally, the Driver API may not immediately return enough node IDs (e.g. due to real-world network issues), so we loop and wait until the minimum number of nodes are available. If we have a large pool of nodes which we want to sample from - as is commonly the case for federated learning systems - we can randomly sample node IDs using, for instance, :code:`random.sample(all_node_ids, num_client_nodes_per_round)`. One example is shown below:

.. code-block:: python

    # File: server.py
    
    num_client_nodes_per_round = 3

    for server_round in range(num_rounds):
        log(INFO, f"Commencing server round {server_round + 1}")

        # List of sampled node IDs in this round
        sampled_nodes: List[int] = []

        # Loop and wait until enough nodes are available.
        while True:
            all_node_ids = driver.get_node_ids()
            log(INFO,f"Got {len(all_node_ids)} client nodes: {all_node_ids}")
            if len(all_node_ids) >= num_client_nodes_per_round:
                # Sample client nodes
                sampled_nodes = random.sample(all_node_ids, num_client_nodes_per_round)
                break
            time.sleep(3)

        # Log sampled node IDs
        log(INFO,f"Sampled {len(sampled_nodes)} node IDs: {sampled_nodes}")

        # Do something else
    ...


Push messages to nodes
----------------------

Now that we have a list of node IDs, we can push information to them to execute a task such as training or evaluating a model. To achieve this, we will first use :code:`Driver.create_message()` to create a :code:`Message` and then use :code:`Driver.push_message()` to push the :code:`Messages` to the list of node IDs. 

.. admonition:: Note
    :class: note

    Each :code:`Message` contains a |recordset_link|_ object. It contains :code:`parameters_records`, :code:`metrics_records`, and :code:`configs_records` attributes, which are - unsurprisingly - parameters, metrics, and configurations that are used by a node to execute a task. 

Here is an example of how to push a PyTorch model and instructions to a set of client node IDs to train the model. First, we define a utility function to convert a PyTorch model into a :code:`ParametersRecord` and create a PyTorch model in :code:`main()` (we’ve omitted the implementation details for :code:`Net()`, but you can refer to :doc:`this quickstart tutorial <tutorial-quickstart-pytorch>` for an example):

.. code-block:: python

    # File: server.py

    def pytorch_to_parameter_record(pytorch_module: torch.nn.Module):
        state_dict = pytorch_module.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = _ndarray_to_array(v.numpy())
        return ParametersRecord(state_dict)


    @app.main()
    def main(driver: Driver, context: Context) -> None:
    	global_model = Net()

Then, we create a :code:`RecordSet` and add parameters and configurations to :code:`parameters_record` and :code:`configs_record`, respectively. Note that the dictionary keys used here are customizable, so you have a great degree of flexibility to assign and use the dictionaries in a :code:`RecordSet`:

.. code-block:: python

    # File: server.py
    # In the for-loop in `main()`

    # Create a RecordSet
    recordset = RecordSet()

    # Add model parameters to the RecordSet
    recordset.parameters_records["my_model"] = pytorch_to_parameter_record(global_model)

    # Add a training configuration for 1 epoch only to the RecordSet
    recordset.configs_records["my_config"] = ConfigsRecord({"epochs": 1})

Next, we create a list of :code:`Messages`, one for each node ID. To do so, we loop over all node IDs and run :code:`Driver.create_message()` with the :code:`recordset` as the content of the message: 

.. code-block:: python

    # File: server.py

    messages = []
    for node_id in node_ids:
        message = driver.create_message(
            content=recordset,
            message_type=MessageType.TRAIN,
            dst_node_id=node_id,
            group_id=str(server_round),
            ttl=DEFAULT_TTL,
        )
        messages.append(message)

Finally, we use :code:`Driver.push_messages()` to push the list of :code:`Messages` containing the encapsulated parameters and configurations to the nodes.

.. code-block:: python

    # File: server.py

    message_ids = driver.push_messages(messages)

:code:`Driver.push_messages()` yields an iterable list of message IDs. In some real-world scenarios, you may encounter situations where only some :code:`Messages` can be pushed, so it is good practice to filter out empty message IDs:

.. code-block:: python

    # File: server.py

    # Wait for results, ignore empty message_ids
    message_ids = [message_id for message_id in message_ids if message_id != ""]


Pull messages from nodes
------------------------

Once messages are successfully sent to the nodes, we can use the associated message IDs to get results from these nodes. To do so, we continuously run :code:`Driver.pull_messages()` with the list of :code:`message_ids` until all of the :code:`Messages` from the nodes are received. 

.. code-block:: python

    # File: server.py

    all_replies: List[Message] = []
    while True:
        replies = driver.pull_messages(message_ids=message_ids)
        for res in replies:
            print(f"Got 1 {'result' if res.has_content() else 'error'}")
        all_replies += replies
        if len(all_replies) == len(message_ids):
            break
        print("Pulling messages...")
        time.sleep(3)

To only keep :code:`Messages` with content, we apply a simple filter on the results:

.. code-block:: python

    # File: server.py

    # Filter correct results
    all_replies = [
        msg
        for msg in all_replies
        if msg.has_content()
    ]
    print(f"Received {len(all_replies)} results")


Process results from nodes
--------------------------

Now that we have a results from the nodes in the form of :code:`Messages`, we can access their content and use them for any subsequent server-side tasks. Here is how we print the :code:`metrics_records` for each node in a for-loop:

.. code-block:: python

    # File: server.py

    # Print metrics from nodes
    for reply in all_replies:
    	print(reply.content.metrics_records)

And here is how we can retrieve :code:`parameters_records` from the contents and convert them to PyTorch `state_dict`:

.. code-block:: python

    # File: server.py

    # Convert received parameters_records to state_dicts
    received_state_dicts = [
        parameters_to_pytorch_state_dict(
            reply.content.parameters_records["fancy_model_returned"]
        )
        for reply in all_replies
    ]

Run :code:`ServerApp`
---------------------

To use the :code:`Driver` API, we run the :code:`ServerApp` from CLI using the |flower_serverapp_link|_ command. Pass the :code:`<module>:<attribute>` to the command, where :code:`module` is the filename (:code:`server.py`) and :code:`attribute` is the instantiated :code:`ServerApp` in the :code:`module`:

.. code-block:: shell

    $ flower-server-app server:app  --insecure

.. admonition:: Note
    :class: note

    In this example, the :code:`--insecure` command line argument starts Flower without HTTPS and is only used for prototyping. To run with HTTPS, we instead use the argument :code:`--root-certificates` and pass the paths to the certificate. Please refer to `Flower CLI reference <ref-api-cli.html#flower-server-app>`_ for implementation details.

Conclusion
----------

Congratulations! You now know how to use the :code:`Driver` API to query training nodes and send/receive messages from them.

A full example on the :code:`Driver` API is coming soon, so stay tuned!

.. admonition:: Important
    :class: important

    As we continuously enhance Flower at a rapid pace, we'll periodically update the functionality and this how-to document. Please feel free to share any feedback with us!

If there are further questions, `join the Flower Slack <https://flower.ai/join-slack/>`_ and use the channel ``#questions``. You can also `participate in Flower Discuss <https://discuss.flower.ai/>`_ where you can find us answering questions, or share and learn from others about migrating to Flower Next.
