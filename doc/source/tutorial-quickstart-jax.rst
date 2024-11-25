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
    INFO :      Starting Flower ServerApp, config: num_rounds=3, no round_timeout
    INFO :
    INFO :      [INIT]
    INFO :      Requesting initial parameters from one random client
    INFO :      Received initial parameters from one random client
    INFO :      Starting evaluation of initial global parameters
    INFO :      Evaluation returned no results (`None`)
    INFO :
    INFO :      [ROUND 1]
    INFO :      configure_fit: strategy sampled 10 clients (out of 10)
    INFO :      aggregate_fit: received 10 results and 0 failures
    WARNING :   No fit_metrics_aggregation_fn provided
    INFO :      configure_evaluate: strategy sampled 10 clients (out of 10)
    INFO :      aggregate_evaluate: received 10 results and 0 failures
    WARNING :   No evaluate_metrics_aggregation_fn provided
    INFO :
    INFO :      [ROUND 2]
    INFO :      configure_fit: strategy sampled 10 clients (out of 10)
    INFO :      aggregate_fit: received 10 results and 0 failures
    INFO :      configure_evaluate: strategy sampled 10 clients (out of 10)
    INFO :      aggregate_evaluate: received 10 results and 0 failures
    INFO :
    INFO :      [ROUND 3]
    INFO :      configure_fit: strategy sampled 10 clients (out of 10)
    INFO :      aggregate_fit: received 10 results and 0 failures
    INFO :      configure_evaluate: strategy sampled 10 clients (out of 10)
    INFO :      aggregate_evaluate: received 10 results and 0 failures
    INFO :
    INFO :      [SUMMARY]
    INFO :      Run finished 3 round(s) in 6.07s
    INFO :          History (loss, distributed):
    INFO :                  round 1: 0.29372873306274416
    INFO :                  round 2: 5.820648354415425e-08
    INFO :                  round 3: 1.526226667528834e-14
    INFO :

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

The main changes we have to make to use JAX with Flower will be found in the
``get_params()`` and ``set_params()`` functions. In ``get_params()``, JAX model
parameters are extracted and represented as a list of NumPy arrays. The ``set_params()``
function is the opposite: given a list of NumPy arrays it applies them to an existing
JAX model.

.. note::

    The ``get_params()`` and ``set_params()`` functions here are conceptually similar to
    the ``get_weights()`` and ``set_weights()`` functions that we defined in the
    :doc:`QuickStart PyTorch <tutorial-quickstart-pytorch>` tutorial.

.. code-block:: python

    def get_params(params):
        parameters = []
        for _, val in params.items():
            parameters.append(np.array(val))
        return parameters


    def set_params(local_params, global_params):
        for key, value in list(zip(local_params.keys(), global_params)):
            local_params[key] = value

The rest of the functionality is directly inspired by the centralized case. The
``fit()`` method in the client trains the model using the local dataset. Similarly, the
``evaluate()`` method is used to evaluate the model received on a held-out validation
set that the client might have:

.. code-block:: python

    class FlowerClient(NumPyClient):
        def __init__(self, input_dim):
            self.train_x, self.train_y, self.test_x, self.test_y = load_data()
            self.grad_fn = jax.grad(loss_fn)
            model_shape = self.train_x.shape[1:]

            self.params = load_model(model_shape)

        def fit(self, parameters, config):
            set_params(self.params, parameters)
            self.params, loss, num_examples = train(
                self.params, self.grad_fn, self.train_x, self.train_y
            )
            parameters = get_params({})
            return parameters, num_examples, {"loss": float(loss)}

        def evaluate(self, parameters, config):
            set_params(self.params, parameters)
            loss, num_examples = evaluation(
                self.params, self.grad_fn, self.test_x, self.test_y
            )
            return float(loss), num_examples, {"loss": float(loss)}

Finally, we can construct a ``ClientApp`` using the ``FlowerClient`` defined above by
means of a ``client_fn()`` callback. Note that the `context` enables you to get access
to hyperparemeters defined in your ``pyproject.toml`` to configure the run. In this
tutorial we access the ``local-epochs`` setting to control the number of epochs a
``ClientApp`` will perform when running the ``fit()`` method. You could define
additioinal hyperparameters in ``pyproject.toml`` and access them here.

.. code-block:: python

    def client_fn(context: Context):
        input_dim = context.run_config["input-dim"]
        # Return Client instance
        return FlowerClient(input_dim).to_client()


    # Flower ClientApp
    app = ClientApp(client_fn)

The ServerApp
-------------

To construct a ``ServerApp`` we define a ``server_fn()`` callback with an identical
signature to that of ``client_fn()`` but the return type is |serverappcomponents|_ as
opposed to a |client|_ In this example we use the ``FedAvg`` strategy. To it we pass a
randomly initialized model that will server as the global model to federated. Note that
the value of ``input_dim`` is read from the run config. You can find the default value
defined in the ``pyproject.toml``.

.. code-block:: python

    def server_fn(context: Context):
        # Read from config
        num_rounds = context.run_config["num-server-rounds"]
        input_dim = context.run_config["input-dim"]

        # Initialize global model
        params = get_params(load_model((input_dim,)))
        initial_parameters = ndarrays_to_parameters(params)

        # Define strategy
        strategy = FedAvg(initial_parameters=initial_parameters)
        config = ServerConfig(num_rounds=num_rounds)

        return ServerAppComponents(strategy=strategy, config=config)


    # Create ServerApp
    app = ServerApp(server_fn=server_fn)

Congratulations! You've successfully built and run your first federated learning system
for JAX with Flower!

.. note::

    Check the source code of the extended version of this tutorial in
    |quickstart_jax_link|_ in the Flower GitHub repository.

.. |client| replace:: ``Client``

.. |fedavg| replace:: ``FedAvg``

.. |makeregression| replace:: ``make_regression()``

.. |quickstart_jax_link| replace:: ``examples/quickstart-jax``

.. |serverappcomponents| replace:: ``ServerAppComponents``

.. _client: ref-api/flwr.client.Client.html#client

.. _fedavg: ref-api/flwr.server.strategy.FedAvg.html#flwr.server.strategy.FedAvg

.. _makeregression: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html

.. _quickstart_jax_link: https://github.com/adap/flower/tree/main/examples/quickstart-jax

.. _serverappcomponents: ref-api/flwr.server.ServerAppComponents.html#serverappcomponents

.. meta::
    :description: Check out this Federated Learning quickstart tutorial for using Flower with Jax to train a linear regression model on a scikit-learn dataset.
