"""$project_name: A Flower / JAX app."""

import jax
import jax.numpy as jnp
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np

key = jax.random.PRNGKey(0)


def load_data():
    # Load dataset
    X, y = make_regression(n_features=3, random_state=0)
    X, X_test, y, y_test = train_test_split(X, y)
    return X, y, X_test, y_test


def load_model(model_shape):
    # Extract model parameters
    params = {"b": jax.random.uniform(key), "w": jax.random.uniform(key, model_shape)}
    return params


def loss_fn(params, X, y):
    # Return MSE as loss
    err = jnp.dot(X, params["w"]) + params["b"] - y
    return jnp.mean(jnp.square(err))


def train(params, grad_fn, X, y):
    loss = 1_000_000
    num_examples = X.shape[0]
    for epochs in range(50):
        grads = grad_fn(params, X, y)
        params = jax.tree_map(lambda p, g: p - 0.05 * g, params, grads)
        loss = loss_fn(params, X, y)
    return params, loss, num_examples


def evaluation(params, grad_fn, X_test, y_test):
    num_examples = X_test.shape[0]
    err_test = loss_fn(params, X_test, y_test)
    loss_test = jnp.mean(jnp.square(err_test))
    return loss_test, num_examples


def get_params(params):
    parameters = []
    for _, val in params.items():
        parameters.append(np.array(val))
    return parameters


def set_params(local_params, global_params):
    for key, value in list(zip(local_params.keys(), global_params)):
        local_params[key] = value
