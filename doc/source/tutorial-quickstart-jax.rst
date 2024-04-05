.. _quickstart-jax:


Quickstart JAX
==============

.. meta::
   :description: Check out this Federated Learning quickstart tutorial for using Flower with Jax to train a linear regression model on a scikit-learn dataset.

This tutorial will show you how to use Flower to build a federated version of an existing JAX workload.
We are using JAX to train a linear regression model on a scikit-learn dataset.
We will structure the example similar to our `PyTorch - From Centralized To Federated <https://github.com/adap/flower/blob/main/examples/pytorch-from-centralized-to-federated>`_ walkthrough.
First, we build a centralized training approach based on the `Linear Regression with JAX <https://coax.readthedocs.io/en/latest/examples/linear_regression/jax.html>`_ tutorial`.
Then, we build upon the centralized training code to run the training in a federated fashion.

Before we start building our JAX example, we need install the packages :code:`jax`, :code:`jaxlib`, :code:`scikit-learn`, and :code:`flwr`:

.. code-block:: shell

  $ pip install jax jaxlib scikit-learn flwr


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

.. code-block:: python

    python3 jax_training.py

So far this should all look fairly familiar if you've used JAX before.
Let's take the next step and use what we've built to create a simple federated learning system consisting of one server and two clients.

JAX meets Flower
----------------

The concept of federating an existing workload is always the same and easy to understand.
We have to start a *server* and then use the code in :code:`jax_training.py` for the *clients* that are connected to the *server*.
The *server* sends model parameters to the clients. The *clients* run the training and update the parameters.
The updated parameters are sent back to the *server*, which averages all received parameter updates.
This describes one round of the federated learning process, and we repeat this for multiple rounds.

Our example consists of one *server* and two *clients*. Let's set up :code:`server.py` first. The *server* needs to import the Flower package :code:`flwr`.
Next, we use the :code:`start_server` function to start a server and tell it to perform three rounds of federated learning.

.. code-block:: python

    import flwr as fl

    if __name__ == "__main__":
        fl.server.start_server(server_address="0.0.0.0:8080", config=fl.server.ServerConfig(num_rounds=3))

We can already start the *server*:

.. code-block:: python

    python3 server.py

Finally, we will define our *client* logic in :code:`client.py` and build upon the previously defined JAX training in :code:`jax_training.py`.
Our *client* needs to import :code:`flwr`, but also :code:`jax` and :code:`jaxlib` to update the parameters on our JAX model:

.. code-block:: python

    from typing import Dict, List, Callable, Tuple

    import flwr as fl
    import numpy as np
    import jax
    import jax.numpy as jnp

    import jax_training


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

Having defined the federation process, we can run it.

.. code-block:: python

    def main() -> None:
        """Load data, start MNISTClient."""

        # Load data
        train_x, train_y, test_x, test_y = jax_training.load_data()
        grad_fn = jax.grad(jax_training.loss_fn)

        # Load model (from centralized training) and initialize parameters
        model_shape = train_x.shape[1:]
        params = jax_training.load_model(model_shape)

        # Start Flower client
        client = FlowerClient(params, grad_fn, train_x, train_y, test_x, test_y)
        fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())

    if __name__ == "__main__":
        main()


And that's it. You can now open two additional terminal windows and run

.. code-block:: python

    python3 client.py

in each window (make sure that the server is still running before you do so) and see your JAX project run federated learning across two clients. Congratulations!

Next Steps
----------

The source code of this example was improved over time and can be found here: `Quickstart JAX <https://github.com/adap/flower/blob/main/examples/quickstart-jax>`_.
Our example is somewhat over-simplified because both clients load the same dataset.

You're now prepared to explore this topic further. How about using a more sophisticated model or using a different dataset? How about adding more clients?
