Working with stateful clients
=============================

Flower clients are ephemeral objects that are instantiated to perform a specific workload when sampled by the strategy. This is a desirable behavior since in this way only clients that do require compute or memory resources at a given point in time made use of them. For example, if client _c_ is sampled at round _r_ to perform `fit() <ref-api-flwr.html#flwr.client.Client.fit>`_, the object (e.g. of type `client.NumPyClient <ref-api-flwr.html#flwr.client.NumPyClient>`_) is destroyed once :code:`fit()` is completed. This means that all internal variables defined in the client as well as those that might have been created during the client's workload (in our example :code:`fit()`) will be lost and therefore not present next time client _c_ participates in the FL process. These types of clients are often referred to as *stateless*. In some settings (e.g. federated personalization) having clients whose internal state persist across FL rounds is desirable. Flower comes with built-in support for keeping track of clients' state while they remain ephemeral. The diagram below illustrates how a client lifecycle looks like and how its state is managed. This applies to both simulation and non-simulation FL workloads with Flower.


.. TODO: Client lifecycle and state

.. warning:: Built-in support for stateful clients is a new feature of Flower. While we aim to keep the core functionality unchanged, please be aware some API elements will change. Changes will be reflected in future versions of this documentation page. If you'd like to contribute and shape how stateful clients are implemented, reach out to us on Slack or GitHub.

You can interact with your client's state as follow.

.. code-block:: python

    import flwr as fl

    class FlowerClient(fl.client.NumPyClient):

        ...

        def fit(self, parameters, config):
           ...

            # The state is a simple Python dataclass
            # You can add/set your own variables directly
            self.state.my_important_variable = 1234

            # Then the next time this client participates,
            # its state will contain the variable `my_important_variable` set to 1234

            # Your clients can make use of elements in their state
            # like if they wer defined in the client constructor
            # Note in this example variable "something" is assumed
            # to be contained in the state already.
            local_variable = 2 * self.state.something

            ...


Stateless clients
-----------------

For those forms of FL where clients have all their internal variables defined upon object construction (i.e. in :code:`__init__()`) and that do not change thereafter, using stateless clients is fine and therefore no interaction with the client's internal state is needed (despite it being there always at your disposal). This is arguably the predominant type of client in FL research currently. For applications involving a very large number of devices/clients (more typical in `cross-device` settings), clients are often assumed to be stateless since the expectation of them being sampled more than once is extremely low (read more in `Kairouz et al. (2019) <https://arxiv.org/abs/1912.04977>`_)


Stateful Clients
----------------

As mentioned, some FL settings do require clients to define and update certain internal variables over the course of multiple rounds, and even over the entire duration of the FL process. Flower transparently keeps track of the client's state so when a client object is destroyed (e.g. at the end of :code:`fit()`) its state is preserved in-memory. Then, when the client is instantiated again (i.e. when sampled by the strategy to perform some action) its state is injected so it is ready to be used. This means that when implementing your own Flower client classes, either `flwr.client.Client <ref-api-flwr.html#flwr.client.Client>`_ or `flwr.client.NumPyClient <ref-api-flwr.html#flwr.client.NumPyClient>`_, as long as you follow the steps described below on how to interact with the client's state, your clients would look like if they were stateful.


Using your client state
~~~~~~~~~~~~~~~~~~~~~~~

All Flower clients have a companion object, a `flwr.client.ClientState <ref-api-flwr.html#flwr.client.ClientState>`_ object, stored outside the client. The :code:`ClientState`, which is implemented as a standard Python dataclass (read more about them in the `Python documentation <https://docs.python.org/3/library/dataclasses.html>`_), can keep track of individual states, one for each workload the clients are designed to perform. Normally there will be a single workload in each FL experiment. But in some settings, the same client might be involved in several FL pipelines, each training a different model. When that's the case, the state object the client can access to internally (e.g. during :code:`fit()`) is actually of type `flwr.client.WorkloadState <ref-api-flwr.html#flwr.client.WorkloadState>`_. 

Flower clients get injected a `WorkloadState` object right after they are instantiated and **before** they run the job they were sampled for (e.g. :code:`fit()`, :code:`evaluate()`, etc). The `WorkloadState` is also a Python dataclass with two member variables: one to perform workload-to-state mapping and the other (optional) to use internally to keep track of which client is running (this is useful for simulation) and act accordingly. The latter is useful also for logging purposes during experimentation/prototyping.


Just like normal dataclasses, you can add new attributes:

.. code-block:: python

    class FlowerClient(flwr.client.NumPyClient):

        ...

        def fit(self, parameters, config):
           ...
           self.state.my_fancy_variable = 1234
           # Here our variable is just and integer, but you can add more complex data structures

           print(self.state)
           # WorkloadState(cid:<>, workload: <>): {'cid': '<>', 'workload_id': '<>', 'my_fancy_variable': 1234}


You can update the variables that your client state (type `WorkloadState`) stores:

.. code-block:: python

    class FlowerClient(flwr.client.NumPyClient):

        ...

        def fit(self, parameters, config):
            ...
            # This wil double it's value and assumes it was defined earlier
            self.state.my_fancy_variable *= 2

            # You could check if an attribute is present in your state
            # before updating it's value, else you can initialize it
            if hasattr(self.state, 'number_fit_called'):
                # It is present, so update it
                self.state.number_fit_called += 1
            else:
                # Let's initialize it
                self.state.number_fit_called = 1



Considerations and best practices
---------------------------------

Implementing the client's state as a Python dataclass brings a fair amount of versatility to your Flower clients. This sections outlines several best practices and considerations.

* **The states of all your clients are kept in-memory.** This means that if your clients store large objects (e.g. entire ML models) in their state, you can run into memory issues on your system, specially if you have a large number of clients in your FL setup.
* **Be mindful of appending to a list** inside your state, as this will make your client state grow over time. 
* **The client state works in all settings**: with non-simulated clients, in single-machine simulation and in multi-node simulation.
* **Communication costs for multi-node settings:** bear in mind that the client state object needs to be transferred to each client from the central node (i.e. where you start the simulation from). This can slow down your simulation if the state of the clients is large.
* **The elements you add to your state need to be serializable** objects if you are running simulations. Standard Python types and data structures (e.g. `int`, `List`, `Dict`... ) and the usual ML data containers (e.g. `NumPy` arrays, `PyTorch` models and tensors as well as those from `TensorFlow`) are. For more information about serialization, please refer to the `Ray documentation <https://docs.ray.io/en/latest/ray-core/objects/serialization.html#serialization>`_.
* **Prior to Flower 1.6 non-virtual clients were stateful** but now are ephemeral, just like virtual clients. This was done to support a wider range of workloads and so the same client class you implement can be used in simulation without restrictions. You can achieve the exact same behavior with recent versions of Flower as in Flower 1.5 or earlier, but you'll need to store those elements that should persist in the client's `self.state`.


