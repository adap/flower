:og:description: Build stateful ClientApps in Flower with context objects, enabling efficient simulations and deployments.
.. meta::
    :description: Build stateful ClientApps in Flower with context objects, enabling efficient simulations and deployments.

Design stateful ClientApps
==========================

.. _array: ref-api/flwr.common.Array.html

.. _clientapp: ref-api/flwr.client.ClientApp.html

.. _configrecord: ref-api/flwr.common.ConfigRecord.html

.. _context: ref-api/flwr.common.Context.html

.. _metricrecord: ref-api/flwr.common.MetricRecord.html

.. _numpyclient: ref-api/flwr.client.NumPyClient.html

.. _parametersrecord: ref-api/flwr.common.ParametersRecord.html

.. _recorddict: ref-api/flwr.common.RecordDict.html#recorddict

By design, ClientApp_ objects are stateless. This means that the ``ClientApp`` object is
recreated each time a new ``Message`` is to be processed. This behaviour is identical
with Flower's Simulation Engine and Deployment Engine. For the former, it allows us to
simulate the running of a large number of nodes on a single machine or across multiple
machines. For the latter, it enables each ``SuperNode`` to be part of multiple runs,
each running a different ``ClientApp``.

When a ``ClientApp`` is executed it receives a Context_. This context is unique for each
``ClientApp``, meaning that subsequent executions of the same ``ClientApp`` from the
same node will receive the same ``Context`` object. In the ``Context``, the ``.state``
attribute can be used to store information that you would like the ``ClientApp`` to have
access to for the duration of the run. This could be anything from intermediate results
such as the history of training losses (e.g. as a list of `float` values with a new
entry appended each time the ``ClientApp`` is executed), certain parts of the model that
should persist on the client side, or some other arbitrary Python objects. These items
would need to be serialized before saving them into the context.

Saving metrics to the context
-----------------------------

This section will demonstrate how to save metrics such as accuracy/loss values to the
Context_ so they can be used in subsequent executions of the ``ClientApp``. If your
``ClientApp`` makes use of NumPyClient_ then entire object is also re-created for each
call to methods like ``fit()`` or ``evaluate()``.

Let's begin with a simple setting in which ``ClientApp`` is defined as follows. The
``evaluate()`` method only generates a random number and prints it.

.. tip::

    You can create a PyTorch project with ready-to-use ``ClientApp`` and other
    components by running ``flwr new``.

.. code-block:: python

    import random
    from flwr.common import Context, ConfigRecord
    from flwr.client import ClientApp, NumPyClient


    class SimpleClient(NumPyClient):

        def __init__(self):
            self.n_val = []

        def evaluate(self, parameters, config):
            n = random.randint(0, 10)  # Generate a random integer between 0 and 10
            self.n_val.append(n)
            # Even though in this line `n_val` has the value returned in the line
            # above, self.n_val will be re-initialized to an empty list the next time
            # this `ClientApp` runs
            return float(0.0), 1, {}


    def client_fn(context: Context):
        return SimpleClient().to_client()


    # Finally, construct the ClientApp instance by means of the `client_fn` callback
    app = ClientApp(client_fn=client_fn)

Let's say we want to save that randomly generated integer and append it to a list that
persists in the context. To do that, you'll need to do two key things:

1. Make the ``context.state`` reachable within your client class
2. Initialise the appropriate record type (in this example we use ConfigRecord_) and
   save/read your entry when required.

.. code-block:: python

    def SimpleClient(NumPyClient):

        def __init__(self, context: Context):
            self.client_state = (
                context.state
            )  # add a reference to the state of your ClientApp
            if "eval_metrics" not in self.client_state.config_records:
                self.client_state.config_records["eval_metrics"] = ConfigRecord()

            # Print content of the state
            # You'll see it persists previous entries of `n_val`
            print(self.client_state.config_records)

        def evaluate(self, parameters, config):
            n = random.randint(0, 10)  # Generate a random integer between 0 and 10
            # Add results into a `ConfigRecord` object under the "n_val" key
            # Note a `ConfigRecord` is a special type of python Dictionary
            eval_metrics = self.client_state.config_records["eval_metrics"]
            if "n_val" not in eval_metrics:
                eval_metrics["n_val"] = [n]
            else:
                eval_metrics["n_val"].append(n)

            return float(0.0), 1, {}


    def client_fn(context: Context):
        return SimpleClient(context).to_client()  # Note we pass the context


    # Finally, construct the ClientApp instance by means of the `client_fn` callback
    app = ClientApp(client_fn=client_fn)

If you run the app, you'll see an output similar to the one below. See how after each
round the `n_val` entry in the context gets one additional integer ? Note that the order
in which the `ClientApp` logs these messages might differ slightly between rounds.

.. code-block:: shell

    # round 1 (.evaluate() hasn't been executed yet, so that's why it's empty)
    config_records={'eval_metrics': {}}
    config_records={'eval_metrics': {}}

    # round 2 (note `eval_metrics` has results added in round 1)
    config_records={'eval_metrics': {'n_val': [2]}}
    config_records={'eval_metrics': {'n_val': [8]}}

    # round 3 (note `eval_metrics` has results added in round 1&2)
    config_records={'eval_metrics': {'n_val': [8, 2]}}
    config_records={'eval_metrics': {'n_val': [2, 9]}}

    # round 4 (note `eval_metrics` has results added in round 1&2&3)
    config_records={'eval_metrics': {'n_val': [2, 9, 4]}}
    config_records={'eval_metrics': {'n_val': [8, 2, 5]}}

Saving model parameters to the context
--------------------------------------

Using ConfigRecord_ or MetricRecord_ to save "simple" components is fine (e.g., float,
integer, boolean, string, bytes, and lists of these types. Note that MetricRecord_ only
supports float, integer, and lists of these types) Flower has a specific type of record,
a ParametersRecord_, for storing model parameters or more generally data arrays.

Let's see a couple of examples of how to save NumPy arrays first and then how to save
parameters of PyTorch and TensorFlow models.

.. note::

    The examples below omit the definition of a ``ClientApp`` to keep the code blocks
    concise. To make use of ``ParametersRecord`` objects in your ``ClientApp`` you can
    follow the same principles as outlined earlier.

Saving NumPy arrays to the context
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Elements stored in a `ParametersRecord` are of type Array_, which is a data structure
that holds ``bytes`` and metadata that can be used for deserialization. Let's see how to
create an ``Array`` from a NumPy array and insert it into a ``ParametersRecord``. Here
we will make use of the built-in serialization and deserialization mechanisms in Flower,
namely the ``flwr.common.array_from_numpy`` function and the `numpy()` method of an
Array_ object.

.. note::

    Array_ objects carry bytes as their main payload and additional metadata to use for
    deserialization. You can implement your own serialization/deserialization if the
    provided ``array_from_numpy`` doesn't fit your usecase.

Let's see how to use those functions to store a NumPy array into the context.

.. code-block:: python

    import numpy as np
    from flwr.common import Context, ParametersRecord, array_from_numpy


    # Let's create a simple NumPy array
    arr_np = np.random.randn(3, 3)

    # If we print it
    # array([[-1.84242409, -1.01539537, -0.46528405],
    #        [ 0.32991896,  0.55540414,  0.44085534],
    #        [-0.10758364,  1.97619858, -0.37120501]])

    # Now, let's serialize it and construct an Array
    arr = array_from_numpy(arr_np)

    # If we print it (note the binary data)
    # Array(dtype='float64', shape=[3, 3], stype='numpy.ndarray', data=b'\x93NUMPY\x01\x00v\x00...)

    # It can be inserted in a ParametersRecord like this
    p_record = ParametersRecord({"my_array": arr})

    # Then, it can be added to the state in the context
    context.state.parameters_records["some_parameters"] = p_record

To extract the data in a ``ParametersRecord``, you just need to deserialize the array if
interest. For example, following the example above:

.. code-block:: python

    # Get Array from context
    arr = context.state.parameters_records["some_parameters"]["my_array"]

    # Deserialize it
    arr_deserialized = arr.numpy()

    # If we print it (it should show the exact same values as earlier)
    # array([[-1.84242409, -1.01539537, -0.46528405],
    #        [ 0.32991896,  0.55540414,  0.44085534],
    #        [-0.10758364,  1.97619858, -0.37120501]])

Saving PyTorch parameters to the context
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Following the NumPy example above, to save parameters of a PyTorch model a
straightforward way of doing so is to transform the parameters into their NumPy
representation and then proceed as shown earlier. Below is a simple self-contained
example for how to do this.

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from flwr.common import Array, ParametersRecord, array_from_numpy


    class Net(nn.Module):
        """A very simple model"""

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 32, 5)
            self.fc = nn.Linear(1024, 10)

        def forward(self, x):
            x = F.relu(self.conv(x))
            return self.fc(x)


    # Instantiate model as usual
    model = Net()

    # Save all elements of the state_dict into a single RecordDict
    p_record = ParametersRecord()
    for k, v in model.state_dict().items():
        # Convert to NumPy, then to Array. Add to record
        p_record[k] = array_from_numpy(v.detach().cpu().numpy())

    # Add to a context
    context.state.parameters_records["net_parameters"] = p_record

Let say now you want to apply the parameters stored in your context to a new instance of
the model (as it happens each time a ``ClientApp`` is executed). You will need to:

1. Deserialize each element in your specific ``ParametersRecord``
2. Construct a ``state_dict`` and load it

.. code-block:: python

    state_dict = {}
    # Extract record from context
    p_record = context.state.parameters_records["net_parameters"]

    # Deserialize arrays
    for k, v in p_record.items():
        state_dict[k] = torch.from_numpy(v.numpy())

    # Apply state dict to a new model instance
    model_ = Net()
    model_.load_state_dict(state_dict)
    # now this model has the exact same parameters as the one created earlier
    # You can verify this by doing
    for p, p_ in zip(model.state_dict().values(), model_.state_dict().values()):
        assert torch.allclose(p, p_), "`state_dict`s do not match"

And that's it! Recall that even though this example shows how to store the entire
``state_dict`` in a ``ParametersRecord``, you can just save part of it. The process
would be identical, but you might need to adjust how it is loaded into an existing model
using PyTorch APIs.

Saving Tensorflow/Keras parameters to the context
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Follow the same steps as done above but replace the ``state_dict`` logic with simply
`get_weights() <https://www.tensorflow.org/api_docs/python/tf/keras/Layer#get_weights>`_
to convert the model parameters to a list of NumPy arrays that can then be serialized
into an ``Array``. Then, after deserialization, use `set_weights()
<https://www.tensorflow.org/api_docs/python/tf/keras/Layer#set_weights>`_ to apply the
new parameters to a model.
