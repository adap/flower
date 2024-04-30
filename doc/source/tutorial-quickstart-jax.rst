.. _quickstart-jax:


Quickstart JAX
==============

.. meta::
   :description: Check out this Federated Learning quickstart tutorial for using Flower with Jax to train a linear regression model on a scikit-learn dataset.

Let's build a federated learning system using JAX and the Flower framework!

We will leverage JAX to train a linear regression model on a scikit-learn dataset.
We will structure the example similar to our `PyTorch - From Centralized To Federated <https://github.com/adap/flower/blob/main/examples/pytorch-from-centralized-to-federated>`_ walkthrough.
First, we build a centralized training approach based on the `Linear Regression with JAX <https://coax.readthedocs.io/en/latest/examples/linear_regression/jax.html>`_ tutorial.
Then, we build upon the centralized training code to run the training in a federated fashion over multiple clients using Flower.


Dependencies
------------

First of all, it is recommended to create a virtual environment and run everything within a :doc:`virtualenv <contributor-how-to-set-up-a-virtual-env>`.

To follow along this tutorial you will need to install the following packages: :code:`jax`, :code:`jaxlib`, :code:`scikit-learn`, and :code:`flwr`. This can be done using :code:`pip`:

.. code-block:: shell

  $ pip install flwr jax jaxlib scikit-learn


Linear Regression with JAX
--------------------------

We begin with a brief description of the centralized training code based on a :code:`Linear Regression` model.
If you want a more in-depth explanation of what's going on then have a look at the official `JAX documentation <https://jax.readthedocs.io/>`_.

Let's create a new file called :code:`jax_training.py` with all the components required for a traditional (centralized) linear regression training. 
First, the JAX packages :code:`jax` and :code:`jaxlib` need to be imported. In addition, we need to import :code:`sklearn` since we use :code:`make_regression` for the dataset and :code:`train_test_split` to split the dataset into a training and test set. 
You can see that we do not yet import the :code:`flwr` package for federated learning. This will be done later. 

.. code-block:: python

    from typing import Dict, List, Tuple, Callable
    import jax
    import jax.numpy as jnp
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    key = jax.random.PRNGKey(0)

The :code:`load_data()` function loads the mentioned training and test sets.

.. code-block:: python

    def load_data() -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        # create our dataset and start with similar datasets for different clients
        X, y = make_regression(n_features=3, random_state=0)
        X, X_test, y, y_test = train_test_split(X, y)
        return X, y, X_test, y_test

The model architecture (a very simple :code:`Linear Regression` model) is defined in :code:`load_model()`.

.. code-block:: python

    def load_model(model_shape) -> Dict:
        # model weights
        params = {
            'b' : jax.random.uniform(key),
            'w' : jax.random.uniform(key, model_shape)
        }
        return params

We now need to define the training (function :code:`train()`), which loops over the training set and measures the loss (function :code:`loss_fn()`) for each batch of training examples. The loss function is separate since JAX takes derivatives with a :code:`grad()` function (defined in the :code:`main()` function and called in :code:`train()`). 

.. code-block:: python

    def loss_fn(params, X, y) -> Callable:
        err = jnp.dot(X, params['w']) + params['b'] - y
        return jnp.mean(jnp.square(err))  # mse

    def train(params, grad_fn, X, y) -> Tuple[np.array, float, int]:
        num_examples = X.shape[0]
        for epochs in range(10):
            grads = grad_fn(params, X, y)
            params = jax.tree_multimap(lambda p, g: p - 0.05 * g, params, grads)
            loss = loss_fn(params,X, y)
            # if epochs % 10 == 9:
            #     print(f'For Epoch {epochs} loss {loss}')
        return params, loss, num_examples

The evaluation of the model is defined in the function :code:`evaluation()`. The function takes all test examples and measures the loss of the linear regression model. 

.. code-block:: python

    def evaluation(params, grad_fn, X_test, y_test) -> Tuple[float, int]:
        num_examples = X_test.shape[0]
        err_test = loss_fn(params, X_test, y_test)
        loss_test = jnp.mean(jnp.square(err_test))
        # print(f'Test loss {loss_test}')
        return loss_test, num_examples

Having defined the data loading, model architecture, training, and evaluation we can put everything together and train our model using JAX. As already mentioned, the :code:`jax.grad()` function is defined in :code:`main()` and passed to :code:`train()`.

.. code-block:: python

    def main():
        X, y, X_test, y_test = load_data()
        model_shape = X.shape[1:]
        grad_fn = jax.grad(loss_fn)
        print("Model Shape", model_shape)
        params = load_model(model_shape)   
        params, loss, num_examples = train(params, grad_fn, X, y)
        evaluation(params, grad_fn, X_test, y_test)


    if __name__ == "__main__":
        main()

You can now run your (centralized) JAX linear regression workload:

.. code-block:: shell

    python3 jax_training.py

So far this should all look fairly familiar if you've used JAX before.
Let's take the next step and use what we've built to create a simple federated learning system consisting of one server and two clients.

JAX meets Flower
----------------

The concept of federating an existing workload is always the same and easy to understand.
We have to define the Flower interface for the *clients* using the code in :code:`jax_training.py`. We also start a *server* for the *clients* to connect to.
The *server* sends model parameters to the clients. The *clients* run the training and update the parameters.
The updated parameters are sent back to the *server*, which averages all received parameter updates.
This describes one round of the federated learning process, and we repeat this for multiple rounds.

Our example consists of one *server* and two *clients*.

Flower Client
^^^^^^^^^^^^^

First, we set up our *client* logic in :code:`client.py` by building upon the previously defined JAX training in :code:`jax_training.py`.
Our *client* needs to import :code:`flwr` and :code:`jax` to update the parameters on our JAX model:

.. code-block:: python

    from typing import Dict, List, Tuple

    import flwr as fl
    import jax
    import numpy as np
    from flwr.client import ClientApp

    import jax_training

Now we load the data, set the loss function, and set the model shape:

.. code-block:: python

    train_x, train_y, test_x, test_y = jax_training.load_data()
    grad_fn = jax.grad(jax_training.loss_fn)
    model_shape = train_x.shape[1:]

After preparing the data and model, we define the Flower interface.

Implementing a Flower *client* basically means implementing a subclass of either :code:`flwr.client.Client` or :code:`flwr.client.NumPyClient`.
Our implementation will be based on :code:`flwr.client.NumPyClient` and we'll call it :code:`FlowerClient`.
:code:`NumPyClient` is slightly easier to implement than :code:`Client` if you use a framework with good NumPy interoperability (like JAX) because it avoids some of the boilerplate that would otherwise be necessary.
:code:`FlowerClient` needs to implement four methods, two methods for getting/setting model parameters, one method for training the model, and one method for testing the model:

#. :code:`set_parameters (optional)`
    * set the model parameters on the local model that are received from the server
    * transform parameters to NumPy :code:`ndarray`'s
    * loop over the list of model parameters received as NumPy :code:`ndarray`'s (think list of neural network layers)
#. :code:`get_parameters`
    * get the model parameters and return them as a list of NumPy :code:`ndarray`'s (which is what :code:`flwr.client.NumPyClient` expects)
#. :code:`fit`
    * update the parameters of the local model with the parameters received from the server
    * train the model on the local training set
    * get the updated local model parameters and return them to the server
#. :code:`evaluate`
    * update the parameters of the local model with the parameters received from the server
    * evaluate the updated model on the local test set
    * return the local loss to the server

The challenging part is to transform the JAX model parameters from :code:`DeviceArray` to :code:`NumPy ndarray` to make them compatible with `NumPyClient`. 

The two :code:`NumPyClient` methods :code:`fit` and :code:`evaluate` make use of the functions :code:`train()` and :code:`evaluate()` previously defined in :code:`jax_training.py`.
So what we really do here is we tell Flower through our :code:`NumPyClient` subclass which of our already defined functions to call for training and evaluation.
We included type annotations to give you a better understanding of the data types that get passed around.

.. code-block:: python

    class FlowerClient(fl.client.NumPyClient):
        """Flower client implementing using linear regression and JAX."""

        def __init__(
            self,
            params: Dict,
            grad_fn: Callable,
            train_x: List[np.ndarray],
            train_y: List[np.ndarray],
            test_x: List[np.ndarray],
            test_y: List[np.ndarray],
        ) -> None:
            self.params= params
            self.grad_fn = grad_fn
            self.train_x = train_x
            self.train_y = train_y
            self.test_x = test_x
            self.test_y = test_y

        def get_parameters(self, config) -> Dict:
            # Return model parameters as a list of NumPy ndarrays
            parameter_value = []
            for _, val in self.params.items():
                parameter_value.append(np.array(val))
            return parameter_value
        
        def set_parameters(self, parameters: List[np.ndarray]) -> Dict:
            # Collect model parameters and update the parameters of the local model
            value=jnp.ndarray
            params_item = list(zip(self.params.keys(),parameters))
            for item in params_item:
                key = item[0]
                value = item[1]
                self.params[key] = value
            return self.params
        
        def fit(
            self, parameters: List[np.ndarray], config: Dict
        ) -> Tuple[List[np.ndarray], int, Dict]:
            # Set model parameters, train model, return updated model parameters
            print("Start local training")
            self.params = self.set_parameters(parameters)
            self.params, loss, num_examples = jax_training.train(self.params, self.grad_fn, self.train_x, self.train_y)
            results = {"loss": float(loss)}
            print("Training results", results)
            return self.get_parameters(config={}), num_examples, results

        def evaluate(
            self, parameters: List[np.ndarray], config: Dict
        ) -> Tuple[float, int, Dict]:
            # Set model parameters, evaluate the model on a local test dataset, return result
            print("Start evaluation")
            self.params = self.set_parameters(parameters)
            loss, num_examples = jax_training.evaluation(self.params,self.grad_fn, self.test_x, self.test_y)
            print("Evaluation accuracy & loss", loss)
            return (
                float(loss),
                num_examples,
                {"loss": float(loss)},
            )

Next, we create a client function that returns instances of :code:`FlowerClient` on-demand when called:

.. code-block:: python

    def client_fn(cid: str):
        return FlowerClient().to_client()

Finally, we create a :code:`ClientApp()` object that uses this client function:

.. code-block:: python

    app = ClientApp(client_fn=client_fn)

That's it for the client. We only have to implement :code:`Client` or :code:`NumPyClient`, create a :code:`ClientApp`, and pass the client function to it. If we implement a client of type :code:`NumPyClient` we'll need to first call its :code:`to_client()` method.


Flower Server
^^^^^^^^^^^^^

For simple workloads, we create a :code:`ServerApp` and leave all the
configuration possibilities at their default values. In a file named
:code:`server.py`, import Flower and create a :code:`ServerApp`:

.. code-block:: python

    from flwr.server import ServerApp

    app = ServerApp()


Train the model, federated!
---------------------------

With both :code:`ClientApps` and :code:`ServerApp` ready, we can now run everything and see federated
learning in action. First, we run the :code:`flower-superlink` command in one terminal to start the infrastructure. This step only needs to be run once.

.. admonition:: Note
    :class: note

    In this example, the :code:`--insecure` command line argument starts Flower without HTTPS and is only used for prototyping. To run with HTTPS, we instead use the argument :code:`--certificates` and pass the paths to the certificates. Please refer to `Flower CLI reference <ref-api-cli.html>`_ for implementation details.

.. code-block:: shell

    $ flower-superlink --insecure

FL systems usually have a server and multiple clients. We therefore need to start multiple `SuperNodes`, one for each client, respectively. First, we open a new terminal and start the first `SuperNode` using the :code:`flower-client-app` command.

.. code-block:: shell

    $ flower-client-app client:app --insecure

In the above, we launch the :code:`app` object in the :code:`client.py` module.
Open another terminal and start the second `SuperNode`:

.. code-block:: shell

    $ flower-client-app client:app --insecure

Finally, in another terminal window, we run the `ServerApp`. This starts the actual training run:

.. code-block:: shell

    $ flower-server-app server:app --insecure

We should now see how the training does in the last terminal (the one that started the :code:`ServerApp`):

.. code-block:: shell

    WARNING :   Option `--insecure` was set. Starting insecure HTTP client connected to 0.0.0.0:9091.
    INFO :      Starting Flower ServerApp, config: num_rounds=1, no round_timeout
    INFO :
    INFO :      [INIT]
    INFO :      Requesting initial parameters from one random client
    INFO :      Received initial parameters from one random client
    INFO :      Evaluating initial global parameters
    INFO :
    INFO :      [ROUND 1]
    INFO :      configure_fit: strategy sampled 2 clients (out of 2)
    INFO :      aggregate_fit: received 2 results and 0 failures
    WARNING :   No fit_metrics_aggregation_fn provided
    INFO :      configure_evaluate: strategy sampled 2 clients (out of 2)
    INFO :      aggregate_evaluate: received 2 results and 0 failures
    WARNING :   No evaluate_metrics_aggregation_fn provided
    INFO :
    INFO :      [SUMMARY]
    INFO :      Run finished 1 rounds in 7.06s
    INFO :      History (loss, distributed):
    INFO :          '\tround 1: 0.15034367516636848\n'

Congratulations!
You've successfully built and run your first federated learning system with JAX.
The full source code for this example can be found in |quickstart_jax_link|_.

.. |quickstart_jax_link| replace:: :code:`examples/quickstart-jax`
.. _quickstart_jax_link: https://github.com/adap/flower/blob/main/examples/quickstart-jax

Of course, this is a very basic example, and a lot can be added or modified.
How about using a more sophisticated model or using a different dataset? How about adding more clients?