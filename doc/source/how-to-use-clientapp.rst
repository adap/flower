Use :code:`ClientApp`
=====================

With Flower 1.8, we introduced the :code:`ClientApp` and :code:`ServerApp`. These are new features in Flower that allow more flexibility for you to run your federated learning projects, such as adding intermediate operations in your workflows :doc:`using Mods <how-to-use-built-in-mods>`, maximizing compute usage by :doc:`running multiple projects concurrently <how-to-use-multirun>`, and speeding up your project deployment process by :doc:`deploying from a simulation setup <how-to-deploy-to-production>`.

.. |clientapp_link| replace:: ``ClientApp()``
.. |serverapp_link| replace:: ``ServerApp()``
.. |message_link| replace:: ``Message()``
.. |context_link| replace:: ``Context()``
.. |recordset_link| replace:: ``RecordSet()``
.. |flower_clientapp_link| replace:: ``flower-client-app``
.. _clientapp_link: ref-api/flwr.client.ClientApp.html
.. _serverapp_link: ref-api/flwr.server.ServerApp.html
.. _message_link: ref-api/flwr.common.Message.html
.. _context_link: ref-api/flwr.common.Context.html
.. _flower_clientapp_link: ref-api-cli.html#flower-client-app
.. _recordset_link: ref-api/flwr.common.RecordSet.html#recordset

What is a :code:`ClientApp`?
----------------------------

The |clientapp_link|_ is an application that can run various tasks on clients, such as model training. It is launched on demand by the :code:`SuperNode`. The communication pattern between the :code:`SuperNode` and :code:`ClientApp` uses :code:`Message` and :code:`Context` objects containing all the information needed to run the tasks. These objects will be elaborated further below.

There are two ways to use the :code:`ClientApp`:

1. :ref:`Using high-level ClientApp API<Using high-level API>`
2. :ref:`Using low-level ClientApp API<Using low-level API>`

Using high-level API
--------------------

This approach is the compatibility mode. It allows us to reuse any client-code that has been implemented using :code:`NumpyClient` or :code:`Client` classes with a few additional lines of code.

.. admonition:: Tip
    :class: important

    Your existing Flower projects and simulations are compatible with both :code:`ClientApp` and :code:`ServerApp`. To migrate your code, please refer to `How to Upgrade to Flower Next (§Required Changes) <how-to-upgrade-to-flower-next.html#required-changes>`_.

Let’s take the :doc:`PyTorch Quickstart <tutorial-quickstart-pytorch>` code for example. In that example, we have included a :code:`CifarClient` class that inherits from the :code:`NumpyClient` convenience class:

.. code-block:: python

    # File: client.py

    class CifarClient(NumPyClient):
        def get_parameters(self, config):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            train(net, trainloader, epochs=1)
            return self.get_parameters(config={}), num_examples["trainset"], {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            return float(loss), num_examples["testset"], {"accuracy": float(accuracy)}

The above is a familiar structure when implementing a custom client module using :code:`NumpyClient` (this has less boilerplate code to implement than :code:`Client`). 

To use :code:`CifarClient()` with the high-level :code:`ClientApp` API, we need to do several things:

#. Wrap :code:`CifarClient` in a function, such as :code:`client_fn`. This allows multiple instances of :code:`CifarClient` to be created when :code:`client_fn` is called.
#. Convert :code:`CifarClient` to a :code:`Client` implementation using the :code:`to_client()` method.

.. code-block:: python

    # File: client.py

    def client_fn(cid: str) -> Client:
        return CifarClient().to_client()

Then, we create the :code:`ClientApp` as follows:

.. code-block:: python

    # File: client.py

    app = ClientApp(client_fn=client_fn)

Finally, your :code:`ClientApp` is ready to be executed!

To run :code:`ClientApp` from CLI, use the |flower_clientapp_link|_ command. Pass the :code:`<module>:<attribute>` to the command, where :code:`module` is the filename (:code:`client.py`) and :code:`attribute` is the instantiated :code:`ClientApp` in the :code:`module`:

.. code-block:: shell

    $ flower-client-app client:app  --insecure

.. admonition:: Note
    :class: note

    In this example, the :code:`--insecure` command line argument starts Flower without HTTPS and is only used for prototyping. To run with HTTPS, we instead use the argument :code:`--root-certificates` and pass the paths to the certificates. Please refer to `Flower CLI reference <ref-api-cli.html#flower-client-app>`_ for implementation details.

As you can see, we can easily reuse existing Flower clients with the :code:`ClientApp` by adding 3 lines of code! Let’s now walk through how to use the low-level API for greater implementation flexibility in our projects.

Using low-level API
-------------------

With Flower 1.8, we provide a set of low-level APIs to allow more versatile ways to implement any functionality that we like in the :code:`ClientApp`. The two functions that can be registered in a :code:`ClientApp` are:

* :code:`app.train()`
* :code:`app.evaluate()`

which runs the training and evaluation on client-side data. 

Next, we have a key part in the low-level API, which is the message passing format. In Flower 1.8, we introduced |message_link|_ and |context_link|_ objects. The :code:`Message` is an abstraction that unifies the message formats that are relayed back and forth in a Flower project. When compared to the high-level example above, objects such as :code:`config` and :code:`parameter` need to be separately configured and tracked in each method (such as :code:`get_parameters()`, :code:`set_parameters()`, :code:`fit()`, and :code:`evaluate()`) which can make the workflow rigid. To simplify the usage, we unify the data format via the :code:`Message` abstractions. In contrast, :code:`Context` is a local object accessible to the :code:`SuperNode` - it persists until the :code:`SuperNode` is shut down, at which time it is deallocated.

A :code:`Message` contains different information depending on whether it is received or returned by the :code:`ClientApp`: A received :code:`Message` contains information required to run the tasks within each function, such as training configurations and parameters. A returned :code:`Message` on the other hand, contains results of the computation, such as loss or accuracy metrics. While a :code:`Message` wraps information that is consumed within a round, a :code:`Context` object wraps information for a :code:`SuperNode` that is persisted across each round of federated learning. One example where :code:`Context` can be used is to track the metrics from an earlier round (or rounds), which is handy, such as when early stopping strategies are implemented.

.. admonition:: Tip
    :class: note

    The :code:`Message` and :code:`Context` objects contain data such as parameters, metrics, configs, and identifiers of the current run. All of these information is used by the :code:`ClientApp` to determine what and how to run the task. By default, :code:`Context` is empty, unless a user saves value(s) to it via the  :code:`ClientApp` or :code:`mod`.

In short:

#. :code:`Message` is a simplified messaging pattern between :code:`ClientApp` and :code:`ServerApp`, and is consumed within a round.
#. :code:`Context`, which is a |recordset_link|_ object, persists information across rounds.

With this brief explanation of Flower’s messaging pattern, let’s walk through how to use it in a :code:`ClientApp` .

:code:`app.train()`
~~~~~~~~~~~~~~~~~~~

The :code:`app.train()` decorator registers a single training task on a client, e.g. for FL, it executes one FL round of model training on a client. The following steps outline how to implement it.

First, import the necessary modules and register the function :code:`train()` using the decorator :code:`@app.train()`:

.. code-block:: python

    # File: client.py

    from flwr.client import ClientApp
    from flwr.common import Message, Context

    app = ClientApp()

    @app.train()
    def train(msg: Message, ctx: Context):
        ...

Now, we implement the steps to train a model on the client, which generally follow the pattern:

#. Instantiate model
#. Load local data
#. Get model and configs (e.g. from the :code:`ServerApp`)
#. Train model
#. Return results

The first two steps are straightforward and we can implement as follows:

.. code-block:: python

    # File: client.py

    # instantiate model
    model = Net()

    # load local training and validation data
    train_loader, val_loader, _ = load_data()

For simplicity, we have omitted the implementations for :code:`Net()` and :code:`load_data()`, but you can refer to our :code:`quickstart-pytorch` `code <https://github.com/adap/flower/tree/main/examples/quickstart-pytorch>`_ for similar implementations.

Next, to get the model parameters from the :code:`ServerApp`, we access the :code:`parameters_records` dictionary in the :code:`content` attribute of :code:`Message`, assign value of the :code:`'my_model'` key to a variable, then deserialize it so that we can load the model’s parameter dictionary with the deserialized :code:`state_dict`:

.. code-block:: python

    # File: client.py

    my_parameters = msg.content.parameters_records['my_model']
    state_dict = parameters_to_pytorch_state_dict(my_parameters)
    model.load_state_dict(state_dict=state_dict, strict=True)

Note that in this example, the server sends initial model parameters and training configs for federated learning. It is based on the :doc:`How-to guide for using Flower Driver APIs <how-to-use-driver-api>`. After loading the parameters, we load the configs by accessing the :code:`configs_records` dictionary in the content of the :code:`Message`:

.. code-block:: python

    # File: client.py

    my_config = msg.content.configs_records['my_config']

For this simple example, we now have enough information to train the model, so let’s do it!

.. code-block:: python

    # File: client.py

    train_metrics = train_fn(
        model,
        train_loader,
        val_loader,
        epochs=my_config['epochs'],
        device='cpu',
    )

Next, we prepare the updated model parameters and local metrics to be sent to the :code:`ServerApp` for aggregation. To do so, we create a :code:`RecordSet()` and assign to it two things: the serialized parameters to the :code:`parameters_records` attribute dictionary, and the metrics (which are converted to a :code:`MetricsRecord` object) to the :code:`metrics_records` attribute dictionary:

.. code-block:: python

    # File: client.py

    # Construct reply message carrying updated model parameters and generated metrics
    reply_content = RecordSet()
    reply_content.parameters_records['my_model_returned'] = pytorch_to_parameter_record(model)
    reply_content.metrics_records['train_metrics'] = MetricsRecord(train_metrics)


Finally, we return :code:`reply_content` to the :code:`ServerApp` using the :code:`Message.create_reply()` method:

.. code-block:: python

    # File: client.py

    return msg.create_reply(reply_content)

That’s it! You now have a working Flower :code:`ClientApp` that initializes the model, loads local data, trains the model, and returns the updated model and metrics to the :code:`ServerApp`. Note that the training workflow is dramatically simplified and can be intuitively implemented end-to-end.

For completeness, you can implement the utility functions referenced in the code snippets above as follows:

.. admonition:: Important
    :class: important

    To ease converting any model to a :code:`ParametersRecord` and back, we'll soon include these utility functions natively in the Flower framework. So, stay tuned for updates on this page! 

.. code-block:: python

    # File: utils.py

    import torch
    import numpy as np
    from flwr.common.typing import NDArray
    from flwr.common.record import RecordSet, ParametersRecord, Array

    def _ndarray_to_array(ndarray: NDArray) -> Array:
        """Represent NumPy ndarray as Array."""
        return Array(
            data=ndarray.tobytes(),
            dtype=str(ndarray.dtype),
            stype="numpy.ndarray.tobytes",
            shape=list(ndarray.shape),
        )

    def _basic_array_deserialization(array: Array) -> NDArray:
        return np.frombuffer(buffer=array.data, dtype=array.dtype).reshape(array.shape)

    def pytorch_to_parameter_record(pytorch_module: torch.nn.Module):
        """Serialize your PyTorch model."""
        state_dict = pytorch_module.state_dict()

        for k, v in state_dict.items():
            state_dict[k] = _ndarray_to_array(v.numpy())

        return ParametersRecord(state_dict)

    def parameters_to_pytorch_state_dict(params_record: ParametersRecord):
        """Reconstruct PyTorch state_dict from its serialized representation."""
        state_dict = {}
        for k, v in params_record.items():
            state_dict[k] = torch.tensor(_basic_array_deserialization(v))

        return state_dict

:code:`app.evaluate()`
~~~~~~~~~~~~~~~~~~~~~~

Now that we’ve implemented client training, let’s walk through how to register an evaluation function in the :code:`ClientApp`.

The structure of :code:`app.evaluate()` is the same as as :code:`app.train()`:

1. Instantiate model
2. Load local test data
3. Get aggregated model parameters (e.g. from the :code:`ServerApp`)
4. Evaluate aggregated model
5. Return results

Putting it together, our code is implemented as follows:

.. code-block:: python

    # File: client.py

    @app.evaluate()
    def eval(msg: Message, ctx: Context):
        # 1. Instantiate model
        model = Net()

        # 2. Load local test data
        _, _, test_loader = load_data()

        # 3. Get sent aggregated model
        my_aggregated_parameters = msg.content.parameters_records['my_model']
        state_dict = parameters_to_pytorch_state_dict(my_aggregated_parameters)
        model.load_state_dict(state_dict=state_dict, strict=True)

        # 4. Run local testing
        loss, accuracy = test_fn(model, test_loader)
        test_metrics = {
            "test_loss": loss,
            "test_accuracy": accuracy,
        }

        # 5. Construct reply message carrying test metrics
        reply_content = RecordSet()
        reply_content.metrics_records['test_metrics'] = MetricsRecord(test_metrics)

        return msg.create_reply(reply_content)

The only difference with :code:`app.train()` is that here, we get the :code:`test_loader` and evaluate the test dataset in :code:`test_fn` using the aggregated model.

Finally, with both functions registered, we execute the :code:`ClientApp` to train a model on local data and then test the aggregated model on a test data, as before:

.. code-block:: shell

    $ flower-client-app client:app  --insecure

Conclusion
----------

Congratulations! You now know how to register the :code:`@app.train` and :code:`@app.evaluate` functions for the :code:`ClientApp` . As you can see, the structure is similar for both functions. More importantly, the sequence follow a natural workflow for typical machine learning projects, making it easier and more versatile for you to implement your own projects. 

A full example on the low-level :code:`ClientApp` is coming soon, so stay tuned!

.. admonition:: Important
    :class: important

    As we continuously enhance Flower at a rapid pace, we'll periodically update the functionality and this how-to document. Please feel free to share any feedback with us!

If there are further questions, `join the Flower Slack <https://flower.ai/join-slack/>`_ and use the channel ``#questions``. You can also `participate in Flower Discuss <https://discuss.flower.ai/>`_ where you can find us answering questions, or share and learn from others about migrating to Flower Next.
