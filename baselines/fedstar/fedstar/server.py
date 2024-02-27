"""Required imports for server.py script."""

import gc
import logging
import os
import sys
import time
import uuid

import flwr
import hydra
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf

from fedstar.dataset import DataBuilder
from fedstar.models import Network

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
GRPC_MAX_MESSAGE_LENGTH: int = 536_870_912

parent_path = os.getcwd()


def get_eval_fn(ds_test, evaluation_step, num_classes, verbose):
    """Return a function to evaluate the model during federated learning.

    This function is used by the Flower framework to evaluate the model at specified
    intervals using the provided test dataset.

    Parameters
    ----------
    - ds_test (tf.data.Dataset): The dataset used for evaluation.
    - evaluation_step: After this rounds of aggregartion server evaluates the model.
    - num_classes: Number of classes/labels model has to classify.
    - verbose: How much details should be displayed.

    Returns
    -------
    - Function: An evaluation function that takes server_round, weights, and configs
    as parameters and returns the loss and accuracy.
    """

    # pylint: disable=unused-argument
    def evaluate(server_round, weights, configs):
        # Clear session
        # https://www.tensorflow.org/api_docs/python/tf/keras/backend/clear_session
        tf.keras.backend.clear_session()
        loss, acc = 0, 0
        if (server_round - 1) % evaluation_step == 0:
            model = Network(num_classes=num_classes).get_evaluation_network()
            model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True, name="loss"
                ),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            )
            model.set_weights(weights)
            loss, acc = model.evaluate(ds_test, verbose=verbose)
        gc.collect()
        return float(loss), {"accuracy": float(acc)}

    return evaluate


def get_on_fit_config_fn(total_rounds: int, epochs: int, batch_size: int):
    """Return a function to configure the federated learning fit process.

    This function is called by the Flower framework to obtain configuration
    parameters for each training round.

    Parameters
    ----------
    - total_rounds: Number of coummunication rounds between client and server.
    - epochs: Number of epochs/ iterations
    - batch_size: The number of training instances in the batch

    Returns
    -------
    - Function: A function that takes rounds, epochs, and batch_size
    as parameters and returns a configuration dictionary.
    """

    # pylint: disable=unused-argument
    def fit_config(server_round: int):
        print(
            f"\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
            f"Server started {server_round}th round of training.\n"
            f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
        )
        return {
            "rounds": str(total_rounds),
            "c_round": str(server_round),
            "epochs": str(epochs),
            "batch_size": str(batch_size),
        }

    return fit_config


class AudioServer:  # pylint: disable=too-many-instance-attributes
    """A server class for federated learning using Flower framework.

    This class sets up and runs a federated learning server using the Flower framework,
    handling the orchestration of training across multiple clients. It manages model
    training rounds, evaluation steps, and tracks overall progress and performance.

    Attributes
    ----------
    - flwr_evalution_step (int): After this many rounds the server evaluates the model.
    - flwr_min_sample_size (int): Minimum number of clients participate in a FL round.
    - flwr_num_clients (int): Total number of clients for federated learning.
    - flwr_rounds (int): Number of coummunication rounds between client and server.
    - model_num_classes (int): Number of classes/labels model has to classify.
    - model_batch_size (int): Batch size for model training.
    - model_epochs (int): Number of epochs for model training.
    - model_ds_test (tf.data.Dataset): Dataset for testing the model.
    - model_verbose (int): Verbosity level for model training.
    """

    # pylint: disable=too-many-arguments, too-many-locals
    def __init__(
        self,
        flwr_evalution_step,
        fraction_fit,
        flwr_num_clients,
        flwr_rounds,
        model_num_classes,
        model_batch_size,
        model_epochs,
        model_ds_test,
        model_verbose,
    ):
        # Flower Parameters
        self.rounds = flwr_rounds
        # Local Variables Counters and Variables
        self.current_round = 0
        self.final_accuracy = 0.0
        self.round_time = time.time()
        self.strategy = flwr.server.strategy.FedAvg(
            fraction_fit=float(fraction_fit),
            min_available_clients=flwr_num_clients,
            on_fit_config_fn=get_on_fit_config_fn(
                self.rounds, model_epochs, model_batch_size
            ),
            fraction_evaluate=0,
            min_evaluate_clients=0,
            on_evaluate_config_fn=None,
            evaluate_fn=get_eval_fn(
                model_ds_test,
                flwr_evalution_step,
                num_classes=model_num_classes,
                verbose=model_verbose,
            ),
            accept_failures=True,
        )
        self.client_manager = flwr.server.client_manager.SimpleClientManager()

    def server_start(self, server_address):
        """Start the federated learning server with the given address.

        Initializes and runs the Flower federated learning server using the specified
        server address and pre-configured strategy and client manager.

        Parameters
        ----------
        - server_address (str): The address at which the server will be accessible.
        """
        # print("|" * 50)
        # print(server_address)
        flwr.server.start_server(
            server_address=server_address,
            server=flwr.server.Server(
                client_manager=self.client_manager, strategy=self.strategy
            ),
            config=flwr.server.ServerConfig(num_rounds=self.rounds),
            strategy=self.strategy,
            grpc_max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        )


def set_logger_level():
    """Set the logging level for the 'flower' logger.

    Adjusts the logging level to INFO for the 'flower' logger if it is present in the
    logging manager's dictionary.
    """
    if "flower" in [
        repr(logging.getLogger(name))[8:].split(" ")[0]
        for name in logging.Logger.manager.loggerDict
    ]:
        logger = logging.getLogger("flower")
        logger.setLevel(logging.INFO)


def set_gpu_limits(gpu_id, gpu_memory):
    """Configure GPU settings for TensorFlow.

    Sets the CUDA_VISIBLE_DEVICES environment variable and configures TensorFlow's
    virtual device settings for GPU memory limits.

    Parameters
    ----------
    - gpu_id (str): Identifier for the GPU to be used.
    - gpu_memory (int): The maximum amount of memory (in MB) to be allocated to the GPU.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU's available. Server will run on CPU.")
    else:
        print("GPU is available trying to configure memory.")
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [
                    tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=gpu_memory
                    )
                ],
            )
        except RuntimeError as runtime_error:
            print(runtime_error)


# Change the config path and yaml file name based on expermients you want to run.
# The current configs will call the basic config file which runs the small
# experiment to check integrity.
# Eg:-
# To carry out results for experiment in row 3 row 1 col 1. The parameters will be
# config_path = conf/table_3/row_2_clients_10
# config_name = speech_commands
@hydra.main(config_path="conf", config_name="table3", version_base=None)
def main(cfg: DictConfig):
    """Run the federated learning server with Hydra configuration.

    Initializes and starts the federated learning server based on configurations
    provided by Hydra. It sets up the environment, loads the test dataset, and
    initiates the AudioServer instance to manage the federated learning process.

    Parameters
    ----------
    - cfg (DictConfig): The configuration object provided by Hydra.
    """
    # Set Experiment Parameters
    unique_id = str(uuid.uuid1())

    # print config structured as YAML
    print(OmegaConf.to_yaml(cfg))

    dataset_name = cfg.server.dataset_name
    # Notify Experiment ID to console.
    print(
        f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
        f" Experiment ID : {unique_id}\n"
        f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    )

    # GPU Setup
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        set_gpu_limits(gpu_id="0", gpu_memory=cfg.server.gpu_memory)
    # Load Test Dataset
    ds_test, num_classes = DataBuilder.get_ds_test(
        parent_path=parent_path,
        data_dir=dataset_name,
        batch_size=cfg.server.batch_size,
        buffer=1024,
        seed=cfg.server.seed,
    )
    # Create Server Object
    audio_server = AudioServer(
        flwr_evalution_step=cfg.server.eval_step,
        fraction_fit=cfg.server.fraction_fit,
        flwr_num_clients=cfg.server.num_clients,
        flwr_rounds=cfg.server.rounds,
        model_num_classes=num_classes,
        model_batch_size=cfg.server.batch_size,
        model_epochs=cfg.server.train_epochs,
        model_ds_test=ds_test,
        model_verbose=cfg.server.verbose,
    )
    # Run server
    set_logger_level()
    audio_server.server_start(cfg.server.server_address)


if __name__ == "__main__":
    main()
    sys.exit(0)
