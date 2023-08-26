"""Define the Flower Server and function to instantiate it."""

from keras.utils import to_categorical
from omegaconf import DictConfig

from fedavgm.models import create_model


def get_on_fit_config(config: DictConfig):
    """Generate the function for config."""

    def fit_config_fn(server_round: int):
        # option to use scheduling of learning rate based on round
        # if server_round > 50:
        #     lr = config.lr / 10
        print(f">>> server.py: fit_config_fn | {config}")
        return {
            "local_epochs": config.local_epochs,
            "batch_size": config.batch_size,
        }

    return fit_config_fn


def get_evaluate_fn(input_shape, num_classes, x_test, y_test, num_rounds):
    """Generate the function for server global model evaluation."""
    print(f">>> server.py: get_evaluate_fn: {input_shape}, {num_classes}, {num_rounds}")

    def evaluate_fn(server_round: int, parameters, config):
        # if server_round == num_rounds:
        #    return None

        # instantiate the model
        model = create_model(input_shape, num_classes)
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, to_categorical(y_test, num_classes))

        return loss, {"accuracy": accuracy}

    return evaluate_fn
