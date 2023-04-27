Early stopping
==============

Early stopping allows to save time and computational cost during training.
Indeed, it allows to stop the training after a certain number of consecutive rounds have not shown any improvement.

In Flower, this can be implemented using the :code:`evaluate_fn` that can be passed to a strategy.

Strating in Flower 1.5, this :code:`evaluate_fn` may now return a boolean which will stop the training if :code:`True`.

Example
-------

Built-In Strategies
~~~~~~~~~~~~~~~~~~~

.. All built-in strategies support centralized evaluation by providing
.. an evaluation function during initialization.
.. An evaluation function is any function that can take the current
.. global model parameters as input and return evaluation results:

.. code-block:: python
    
    from flwr.common import NDArrays, Scalar
    
    from typing import Dict, Optional, Tuple

    # The newly defined class for early stopping
    class EarlyStop:

        def __init__(self, patience: int):
            self.best_parameters = None
            self.best_accuracy = 0
            self.count = 0
            self.patience = patience

        def callback(
            self,
            parameters: NDArrays,
            res_cen: Tuple[float, float],
        ) -> bool:
            curr_accuracy = res_cen[1]
            if curr_accuracy > self.best_accuracy:
                self.count = 0
                self.best_parameters = parameters
                self.best_accuracy = curr_accuracy
            else:
                self.count += 1
            
            if self.count > self.patience:
                return True
            else:
                return False

    # The patience will define the number of rounds without improvement we will tolerate
    early_stopping = EarlyStop(patience=5)

    def get_evaluate_fn(model):
        """Return an evaluation function for server-side evaluation."""

        # Load data and model here to avoid the overhead of doing it in `evaluate` itself
        (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

        # Use the last 5k training examples as a validation set
        x_val, y_val = x_train[45000:50000], y_train[45000:50000]

        # The `evaluate` function will be called after every round
        def evaluate(
            server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            model.set_weights(parameters)  # Update model with the latest parameters
            loss, accuracy = model.evaluate(x_val, y_val)

            # This is the line that will be responsible for the early stopping
            stop_training = early_stopping.callback(parameters, (loss, accuracy))

            return loss, {"accuracy": accuracy}, stop_training

        return evaluate

    # Load and compile model for server-side parameter evaluation
    model = tf.keras.applications.EfficientNetB0(
        input_shape=(32, 32, 3), weights=None, classes=10
    )
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])


    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        # ... other FedAvg arguments 
        evaluate_fn=get_evaluate_fn(model),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(server_address="[::]:8080", strategy=strategy)

Custom Strategies
~~~~~~~~~~~~~~~~~

For using early stopping with a custom :code:`Strategy`, 
it is needed to add an :code:`evalute_fn` attribute to the :code:`Strategy` subclass
(similar to what is done with :code:`FedAvg`).
Then this :code:`evaluate_fn` needs to be called in the :code:`evaluate` function of the :code:`Strategy`
(which will therefore return a boolean alongside the loss and the metrics dictionary).
