Configure logging
=================

The Flower logger keeps track of all core events that take place in federated learning workloads.
It presents information by default following a standard message format:

.. code-block:: python

    DEFAULT_FORMATTER = logging.Formatter(
    "%(levelname)s %(name)s %(asctime)s | %(filename)s:%(lineno)d | %(message)s"
    )

containing relevant information including: log message level (e.g. :code:`INFO`, :code:`DEBUG`), a timestamp, the line where the logging took place from, as well as the log message itself.
In this way, the logger would typically display information on your terminal as follows:

.. code-block:: bash

    ...
    INFO flwr 2023-07-15 15:32:30,935 | server.py:125 | fit progress: (3, 392.5575705766678, {'accuracy': 0.2898}, 13.781953627998519)
    DEBUG flwr 2023-07-15 15:32:30,935 | server.py:173 | evaluate_round 3: strategy sampled 25 clients (out of 100)
    DEBUG flwr 2023-07-15 15:32:31,388 | server.py:187 | evaluate_round 3 received 25 results and 0 failures
    DEBUG flwr 2023-07-15 15:32:31,388 | server.py:222 | fit_round 4: strategy sampled 10 clients (out of 100)
    DEBUG flwr 2023-07-15 15:32:32,429 | server.py:236 | fit_round 4 received 10 results and 0 failures
    INFO flwr 2023-07-15 15:32:33,516 | server.py:125 | fit progress: (4, 370.3378576040268, {'accuracy': 0.3294}, 16.36216809399957)
    DEBUG flwr 2023-07-15 15:32:33,516 | server.py:173 | evaluate_round 4: strategy sampled 25 clients (out of 100)
    DEBUG flwr 2023-07-15 15:32:33,966 | server.py:187 | evaluate_round 4 received 25 results and 0 failures
    DEBUG flwr 2023-07-15 15:32:33,966 | server.py:222 | fit_round 5: strategy sampled 10 clients (out of 100)
    DEBUG flwr 2023-07-15 15:32:34,997 | server.py:236 | fit_round 5 received 10 results and 0 failures
    INFO flwr 2023-07-15 15:32:36,118 | server.py:125 | fit progress: (5, 358.6936808824539, {'accuracy': 0.3467}, 18.964264554999318)
    ...


Saving log to file
-------------------

By default, the Flower log is outputted to the terminal where you launch your Federated Learning workload from. This applies for both gRPC-based federation (i.e. when you do :code:`fl.server.start_server`) and when using the :code:`VirtualClientEngine` (i.e. when you do :code:`fl.simulation.start_simulation`).
In some situations you might want to save this log to disk. You can do so by calling the `fl.common.logger.configure() <https://github.com/adap/flower/blob/main/src/py/flwr/common/logger.py>`_ function. For example:

.. code-block:: python
        
        import flwr as fl
        
        ...

        # in your main file and before launching your experiment
        # add an identifier to your logger
        # then specify the name of the file where the log should be outputted to
        fl.common.logger.configure(identifier="myFlowerExperiment", filename="log.txt")

        # then start your workload
        fl.simulation.start_simulation(...) # or fl.server.start_server(...)

With the above, Flower will record the log you see on your terminal to :code:`log.txt`. This file will be created in the same directory as were you are running the code from. 
If we inspect we see the log above is also recorded but prefixing with :code:`identifier` each line:

.. code-block:: bash

    ...
    myFlowerExperiment | INFO flwr 2023-07-15 15:32:30,935 | server.py:125 | fit progress: (3, 392.5575705766678, {'accuracy': 0.2898}, 13.781953627998519)
    myFlowerExperiment | DEBUG flwr 2023-07-15 15:32:30,935 | server.py:173 | evaluate_round 3: strategy sampled 25 clients (out of 100)
    myFlowerExperiment | DEBUG flwr 2023-07-15 15:32:31,388 | server.py:187 | evaluate_round 3 received 25 results and 0 failures
    myFlowerExperiment | DEBUG flwr 2023-07-15 15:32:31,388 | server.py:222 | fit_round 4: strategy sampled 10 clients (out of 100)
    myFlowerExperiment | DEBUG flwr 2023-07-15 15:32:32,429 | server.py:236 | fit_round 4 received 10 results and 0 failures
    myFlowerExperiment | INFO flwr 2023-07-15 15:32:33,516 | server.py:125 | fit progress: (4, 370.3378576040268, {'accuracy': 0.3294}, 16.36216809399957)
    myFlowerExperiment | DEBUG flwr 2023-07-15 15:32:33,516 | server.py:173 | evaluate_round 4: strategy sampled 25 clients (out of 100)
    myFlowerExperiment | DEBUG flwr 2023-07-15 15:32:33,966 | server.py:187 | evaluate_round 4 received 25 results and 0 failures
    myFlowerExperiment | DEBUG flwr 2023-07-15 15:32:33,966 | server.py:222 | fit_round 5: strategy sampled 10 clients (out of 100)
    myFlowerExperiment | DEBUG flwr 2023-07-15 15:32:34,997 | server.py:236 | fit_round 5 received 10 results and 0 failures
    myFlowerExperiment | INFO flwr 2023-07-15 15:32:36,118 | server.py:125 | fit progress: (5, 358.6936808824539, {'accuracy': 0.3467}, 18.964264554999318)
    ...


Log your own messages
---------------------

You might expand the information shown by default with the Flower logger by adding more messages relevant to your application.
You can achieve this easily as follows.

.. code-block:: python

    # in the python file you want to add custom messages to the Flower log
    from logging import INFO, DEBUG
    from flwr.common.logger import log

    # For example, let's say you want to add to the log some info about the training on your client for debugging purposes

    class FlowerClient(fl.client.NumPyClient):
        def __init__(self, cid: int ...):
            self.cid = cid
            self.net = ...
            ...

        def fit(self, parameters, config):
            log(INFO, f"Printing a custom INFO message at the start of fit() :)")
            
            set_params(self.net, parameters)

            log(DEBUG, f"Client {self.cid} is doing fit() with config: {config}")

            ...

In this way your logger will show, in addition to the default messages, the ones introduced by the clients as specified above.

.. code-block:: bash
    
    ...
    INFO flwr 2023-07-15 16:18:21,726 | server.py:89 | Initializing global parameters
    INFO flwr 2023-07-15 16:18:21,726 | server.py:276 | Requesting initial parameters from one random client
    INFO flwr 2023-07-15 16:18:22,511 | server.py:280 | Received initial parameters from one random client
    INFO flwr 2023-07-15 16:18:22,511 | server.py:91 | Evaluating initial parameters
    INFO flwr 2023-07-15 16:18:25,200 | server.py:94 | initial parameters (loss, other metrics): 461.2934241294861, {'accuracy': 0.0998}
    INFO flwr 2023-07-15 16:18:25,200 | server.py:104 | FL starting
    DEBUG flwr 2023-07-15 16:18:25,200 | server.py:222 | fit_round 1: strategy sampled 10 clients (out of 100)
    INFO flwr 2023-07-15 16:18:26,391 | main.py:64 | Printing a custom INFO message :)
    DEBUG flwr 2023-07-15 16:18:26,391 | main.py:63 | Client 44 is doing fit() with config: {'epochs': 5, 'batch_size': 64}
    INFO flwr 2023-07-15 16:18:26,391 | main.py:64 | Printing a custom INFO message :)
    DEBUG flwr 2023-07-15 16:18:28,464 | main.py:63 | Client 99 is doing fit() with config: {'epochs': 5, 'batch_size': 64}
    INFO flwr 2023-07-15 16:18:28,465 | main.py:64 | Printing a custom INFO message :)
    DEBUG flwr 2023-07-15 16:18:28,519 | main.py:63 | Client 67 is doing fit() with config: {'epochs': 5, 'batch_size': 64}
    INFO flwr 2023-07-15 16:18:28,519 | main.py:64 | Printing a custom INFO message :)
    DEBUG flwr 2023-07-15 16:18:28,615 | main.py:63 | Client 11 is doing fit() with config: {'epochs': 5, 'batch_size': 64}
    INFO flwr 2023-07-15 16:18:28,615 | main.py:64 | Printing a custom INFO message :)
    DEBUG flwr 2023-07-15 16:18:28,617 | main.py:63 | Client 13 is doing fit() with config: {'epochs': 5, 'batch_size': 64}
    ...


Log to a remote service
-----------------------

The :code:`fl.common.logger.configure` function, also allows specifying a host to which logs can be pushed (via :code:`POST`) through a native Python :code:`logging.handler.HTTPHandler`.
This is a particularly useful feature in :code:`gRPC`-based Federated Learning workloads where otherwise gathering logs from all entities (i.e. the server and the clients) might be cumbersome.
Note that in Flower simulation, the server automatically displays all logs. You can still specify a :code:`HTTPHandler` should you wish to backup or analyze the logs somewhere else.
