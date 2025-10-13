:og:description: Learn how to train a linear regression using federated learning with Flower and JAX in this step-by-step tutorial.
.. meta::
    :description: Learn how to train a linear regression using federated learning with Flower and JAX in this step-by-step tutorial.

.. _quickstart-jax:

Quickstart JAX
==============

In this federated learning tutorial we will learn how to train a linear regression model
using Flower and `JAX <https://jax.readthedocs.io/en/latest/>`_. It is recommended to
create a virtual environment and run everything within a :doc:`virtualenv
<contributor-how-to-set-up-a-virtual-env>`.

Let's use ``flwr new`` to create a complete Flower+JAX project. It will generate all the
files needed to run, by default with the Flower Simulation Engine, a federation of 10
nodes using |fedavg|_. A random regression dataset will be loaded from scikit-learn's
|makeregression|_ function.

Now that we have a rough idea of what this example is about, let's get started. First,
install Flower in your new environment:

.. code-block:: shell

    # In a new Python environment
    $ pip install flwr

Then, run the command below. You will be prompted to select one of the available
templates (choose ``JAX``), give a name to your project, and type in your developer
name:

.. code-block:: shell

    $ flwr new

After running it you'll notice a new directory with your project name has been created.
It should have the following structure:

.. code-block:: shell

    <your-project-name>
    ├── <your-project-name>
    │   ├── __init__.py
    │   ├── client_app.py   # Defines your ClientApp
    │   ├── server_app.py   # Defines your ServerApp
    │   └── task.py         # Defines your model, training and data loading
    ├── pyproject.toml      # Project metadata like dependencies and configs
    └── README.md

If you haven't yet installed the project and its dependencies, you can do so by:

.. code-block:: shell

    # From the directory where your pyproject.toml is
    $ pip install -e .

To run the project, do:

.. code-block:: shell

    # Run with default arguments
    $ flwr run .

With default arguments you will see an output like this one:

.. code-block:: shell

    Loading project configuration...
    Success
    INFO :      Starting FedAvg strategy:
    INFO :          ├── Number of rounds: 3
    INFO :          ├── ArrayRecord (0.00 MB)
    INFO :          ├── ConfigRecord (train): (empty!)
    INFO :          ├── ConfigRecord (evaluate): (empty!)
    INFO :          ├──> Sampling:
    INFO :          │       ├──Fraction: train (1.00) | evaluate ( 1.00)
    INFO :          │       ├──Minimum nodes: train (2) | evaluate (2)
    INFO :          │       └──Minimum available nodes: 2
    INFO :          └──> Keys in records:
    INFO :                  ├── Weighted by: 'num-examples'
    INFO :                  ├── ArrayRecord key: 'arrays'
    INFO :                  └── ConfigRecord key: 'config'
    INFO :
    INFO :
    INFO :      [ROUND 1/3]
    INFO :      configure_train: Sampled 10 nodes (out of 10)
    INFO :      aggregate_train: Received 10 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'train_loss': 1.2003}
    INFO :      configure_evaluate: Sampled 10 nodes (out of 10)
    INFO :      aggregate_evaluate: Received 10 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'test_loss': 1.5446}
    INFO :
    INFO :      [ROUND 2/3]
    INFO :      configure_train: Sampled 10 nodes (out of 10)
    INFO :      aggregate_train: Received 10 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'train_loss': 0.0005}
    INFO :      configure_evaluate: Sampled 10 nodes (out of 10)
    INFO :      aggregate_evaluate: Received 10 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'test_loss': 2.2913e-07}
    INFO :
    INFO :      [ROUND 3/3]
    INFO :      configure_train: Sampled 10 nodes (out of 10)
    INFO :      aggregate_train: Received 10 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'train_loss': 2.1887e-07}
    INFO :      configure_evaluate: Sampled 10 nodes (out of 10)
    INFO :      aggregate_evaluate: Received 10 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'test_loss': 5.3860e-14}
    INFO :
    INFO :      Strategy execution finished in 10.16s
    INFO :
    INFO :      Final results:
    INFO :
    INFO :          Global Arrays:
    INFO :                  ArrayRecord (0.000 MB)
    INFO :
    INFO :          Aggregated ClientApp-side Train Metrics:
    INFO :          { 1: {'train_loss': '1.2003e+00'},
    INFO :            2: {'train_loss': '5.4981e-04'},
    INFO :            3: {'train_loss': '2.1888e-07'}}
    INFO :
    INFO :          Aggregated ClientApp-side Evaluate Metrics:
    INFO :          { 1: {'test_loss': '1.5446e+00'},
    INFO :            2: {'test_loss': '2.2914e-07'},
    INFO :            3: {'test_loss': '5.3860e-14'}}
    INFO :
    INFO :          ServerApp-side Evaluate Metrics:
    INFO :          {}
    INFO :

    Saving final model to disk...

You can also override the parameters defined in the ``[tool.flwr.app.config]`` section
in ``pyproject.toml`` like this:

.. code-block:: shell

    # Override some arguments
    $ flwr run . --run-config "num-server-rounds=5 input-dim=5"

What follows is an explanation of each component in the project you just created:
dataset partition, the model, defining the ``ClientApp`` and defining the ``ServerApp``.

The Data
--------

This tutorial uses scikit-learn's |makeregression|_ function to generate a random
regression problem.

.. code-block:: python

    def load_data():
        # Load dataset
        X, y = make_regression(n_features=3, random_state=0)
        X, X_test, y, y_test = train_test_split(X, y)
        return X, y, X_test, y_test

The Model
---------

We defined a simple linear regression model to demonstrate how to create a JAX model,
but feel free to replace it with a more sophisticated JAX model if you'd like, (such as
with NN-based `Flax <https://flax.readthedocs.io/en/latest/index.html>`_):

.. code-block:: python

    def load_model(model_shape):
        # Extract model parameters
        params = {"b": jax.random.uniform(key), "w": jax.random.uniform(key, model_shape)}
        return params

In addition to defining the model architecture, we also include two utility functions to
perform both training (i.e. ``train()``) and evaluation (i.e. ``evaluation()``) using
the above model.

.. code-block:: python

    def loss_fn(params, X, y):
        # Return MSE as loss
        err = jnp.dot(X, params["w"]) + params["b"] - y
        return jnp.mean(jnp.square(err))


    def train(params, grad_fn, X, y):
        loss = 1_000_000
        num_examples = X.shape[0]
        for epochs in range(50):
            grads = grad_fn(params, X, y)
            params = jax.tree.map(lambda p, g: p - 0.05 * g, params, grads)
            loss = loss_fn(params, X, y)
        return params, loss, num_examples


    def evaluation(params, grad_fn, X_test, y_test):
        num_examples = X_test.shape[0]
        err_test = loss_fn(params, X_test, y_test)
        loss_test = jnp.mean(jnp.square(err_test))
        return loss_test, num_examples

The ClientApp
-------------

The main changes we have to make to use JAX with Flower have to do with converting the
|arrayrecord_link|_ received in the |message_link|_ into NumPy arrays and vice versa
when generating the reply ``Message`` from the ClientApp. We also have to introduce the
``get_params()`` and ``set_params()`` functions for setting parameter values for the JAX
model. In ``get_params()``, JAX model parameters are extracted and represented as a list
of NumPy arrays. The ``set_params()`` function is the opposite: given a list of NumPy
arrays it applies them to an existing JAX model. We will combine these functions with
the built-in methods in the ``ArrayRecord`` to make these conversions:

.. code-block:: python

    def get_params(params):
        parameters = []
        for _, val in params.items():
            parameters.append(np.array(val))
        return parameters


    def set_params(local_params, global_params):
        for key, value in list(zip(local_params.keys(), global_params)):
            local_params[key] = value

.. code-block:: python

    # Load the model
    model = load_model((input_dim,))

    # Extract ArrayRecord from Message and convert to NumPy arrays
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    # Set JAX model parameters using the converted NumPy arrays
    set_params(model, ndarrays)

    # ... do some training

    # Extract NumPy arrays from the JAX model and convert back into an ArrayRecord
    model_record = ArrayRecord(get_params(model))

The rest of the functionality is directly inspired by the centralized case. The
|clientapp_link|_ comes with three core methods (``train``, ``evaluate``, and ``query``)
that we can implement for different purposes. For example: ``train`` to train the
received model using the local data; ``evaluate`` to assess its performance of the
received model on a validation set; and ``query`` to retrieve information about the node
executing the ``ClientApp``. In this tutorial we will only make use of ``train`` and
``evaluate``.

Let's see how the ``train`` method can be implemented. It receives as input arguments a
|message_link|_ from the ``ServerApp``. By default it carries:

- an ``ArrayRecord`` with the arrays of the model to federate. By default they can be
  retrieved with key ``"arrays"`` when accessing the message content.
- a ``ConfigRecord`` with the configuration sent from the ``ServerApp``. By default it
  can be retrieved with key ``"config"`` when accessing the message content.

The ``train`` method also receives the ``Context``, giving access to configs for your
run and node. The run config hyperparameters are defined in the ``pyproject.toml`` of
your Flower App. The node config can only be set when running Flower with the Deployment
Runtime and is not directly configurable during simulations.

.. code-block:: python

    # Flower ClientApp
    app = ClientApp()


    @app.train()
    def train(msg: Message, context: Context):
        """Train the model on local data."""

        # Read from config
        input_dim = context.run_config["input-dim"]

        # Load data and model
        train_x, train_y, _, _ = load_data()
        model = load_model((input_dim,))
        grad_fn = jax.grad(loss_fn)

        # Set model parameters
        ndarrays = msg.content["arrays"].to_numpy_ndarrays()
        set_params(model, ndarrays)

        # Train the model on local data
        model, loss, num_examples = train_fn(model, grad_fn, train_x, train_y)

        # Construct and return reply Message
        model_record = ArrayRecord(get_params(model))
        metrics = {
            "train_loss": float(loss),
            "num-examples": num_examples,
        }
        metric_record = MetricRecord(metrics)
        content = RecordDict({"arrays": model_record, "metrics": metric_record})
        return Message(content=content, reply_to=msg)

The ``@app.evaluate()`` method would be near identical with two exceptions: (1) the
model is not locally trained, instead it is used to evaluate its performance on the
locally held-out validation set; (2) including the model in the reply Message is no
longer needed because it is not locally modified.

The ServerApp
-------------

To construct a |serverapp_link|_ we define its ``@app.main()`` method. This method
receive as input arguments:

- a ``Grid`` object that will be used to interface with the nodes running the
  ``ClientApp`` to involve them in a round of train/evaluate/query or other.
- a ``Context`` object that provides access to the run configuration.

In this example we use the |fedavg|_ and configure it with a specific value of
``input_dim`` which is read from the run config. You can find the default value defined
in the ``pyproject.toml``. Then, the execution of the strategy is launched when invoking
its |strategy_start_link|_ method. To it we pass:

- the ``Grid`` object.
- an ``ArrayRecord`` carrying a randomly initialized model that will serve as the global
  model to be federated.
- the ``num_rounds`` parameter specifying how many rounds of ``FedAvg`` to perform.

You may also pass a ``ConfigRecord`` with the training hyperparameters to be sent to the
clients. The strategy will also insert the current round number in this config before
sending it to the participating nodes. An example where a ``ConfigRecord`` is passed can
be found in the :doc:`Quickstart PyTorch <tutorial-quickstart-pytorch>` tutorial.

.. code-block:: python

    # Create ServerApp
    app = ServerApp()


    @app.main()
    def main(grid: Grid, context: Context) -> None:
        """Main entry point for the ServerApp."""

        # Read from config
        num_rounds = context.run_config["num-server-rounds"]
        input_dim = context.run_config["input-dim"]

        # Load global model
        model = load_model((input_dim,))
        arrays = ArrayRecord(get_params(model))

        # Initialize FedAvg strategy
        strategy = FedAvg()

        # Start strategy, run FedAvg for `num_rounds`
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            num_rounds=num_rounds,
        )

        # Save final model to disk
        print("\nSaving final model to disk...")
        ndarrays = result.arrays.to_numpy_ndarrays()
        np.savez("final_model.npz", *ndarrays)

Note the ``start`` method of the strategy returns a result object. This object contains
all the relevant information about the FL process, including the final model weights as
an ``ArrayRecord``, and federated training and evaluation metrics as ``MetricRecords``.
You can easily log the metrics using Python's `pprint
<https://docs.python.org/3/library/pprint.html>`_ and save the global model NumPy arrays
using ``np.savez()`` as shown above.

Congratulations! You've successfully built and run your first federated learning system
for JAX with Flower!

.. note::

    Check the source code of the extended version of this tutorial in
    |quickstart_jax_link|_ in the Flower GitHub repository.

.. |fedavg| replace:: ``FedAvg``

.. _fedavg: ref-api/flwr.serverapp.strategy.FedAvg.html

.. |makeregression| replace:: ``make_regression()``

.. _makeregression: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html

.. |quickstart_jax_link| replace:: ``examples/quickstart-jax``

.. _quickstart_jax_link: https://github.com/adap/flower/tree/main/examples/quickstart-jax

.. |message_link| replace:: ``Message``

.. _message_link: ref-api/flwr.app.Message.html

.. |arrayrecord_link| replace:: ``ArrayRecord``

.. _arrayrecord_link: ref-api/flwr.app.ArrayRecord.html

.. |clientapp_link| replace:: ``ClientApp``

.. _clientapp_link: ref-api/flwr.clientapp.ClientApp.html

.. |serverapp_link| replace:: ``ServerApp``

.. _serverapp_link: ref-api/flwr.serverapp.ServerApp.html

.. |strategy_start_link| replace:: ``start``

.. _strategy_start_link: ref-api/flwr.serverapp.strategy.Strategy.html#flwr.serverapp.strategy.Strategy.start

.. |strategy_link| replace:: ``Strategy``

.. _strategy_link: ref-api/flwr.serverapp.Strategy.html
