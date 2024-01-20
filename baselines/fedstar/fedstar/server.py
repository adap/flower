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


class AudioServer:  # pylint: disable=too-many-instance-attributes
    """A server class for federated learning using Flower framework.

    This class sets up and runs a federated learning server using the Flower framework,
    handling the orchestration of training across multiple clients. It manages model
    training rounds, evaluation steps, and tracks overall progress and performance.

    Attributes
    ----------
    - flwr_evalution_step (int): Steps between evaluations in federated learning.
    - flwr_min_sample_size (int): Minimum sample size for federated learning.
    - flwr_min_num_clients (int): Minimum number of clients for federated learning.
    - flwr_rounds (int): Number of federated learning rounds.
    - model_num_classes (int): Number of classes for the model.
    - model_lr (float): Learning rate for the model.
    - model_batch_size (int): Batch size for model training.
    - model_epochs (int): Number of epochs for model training.
    - model_ds_test (tf.data.Dataset): Dataset for testing the model.
    - model_verbose (int): Verbosity level for model training.
    """

    # pylint: disable=too-many-arguments, too-many-locals
    def __init__(
        self,
        flwr_evalution_step,
        flwr_min_sample_size,
        flwr_min_num_clients,
        flwr_rounds,
        model_num_classes,
        model_lr,
        model_batch_size,
        model_epochs,
        model_ds_test,
        model_verbose,
    ):
        # Flower Parameters
        self.evalution_step = flwr_evalution_step
        self.sample_fraction = float(flwr_min_sample_size / flwr_min_num_clients)
        # print("-" * 100)
        # print(self.sample_fraction)
        self.min_sample_size = flwr_min_sample_size
        self.min_num_clients = flwr_min_num_clients
        self.rounds = flwr_rounds
        # Model Parameters
        self.num_classes = model_num_classes
        self.learning_rate = model_lr
        self.batch_size = model_batch_size
        self.epochs = model_epochs
        self.verbose = model_verbose
        self.ds_test = model_ds_test
        # Local Variables Counters and Variables
        self.current_round = 0
        self.final_accuracy = 0.0
        self.round_time = time.time()
        self.strategy = flwr.server.strategy.FedAvg(
            fraction_fit=self.sample_fraction,
            min_fit_clients=self.min_sample_size,
            min_available_clients=self.min_num_clients,
            on_fit_config_fn=self.get_on_fit_config_fn(),
            fraction_evaluate=0,
            min_evaluate_clients=0,
            on_evaluate_config_fn=None,
            evaluate_fn=self.get_eval_fn(ds_test=self.ds_test),
            accept_failures=True,
        )
        self.client_manager = flwr.server.client_manager.SimpleClientManager()
        tf.keras.backend.clear_session()

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

    def get_on_fit_config_fn(self):
        """Return a function to configure the federated learning fit process.

        This function is called by the Flower framework to obtain configuration
        parameters for each training round.

        Returns
        -------
        - Function: A function that takes rounds, epochs, batch_size, and learning_rate
        as parameters and returns a configuration dictionary.
        """

        # pylint: disable=unused-argument
        def fit_config(
            rounds=self.rounds,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
        ):
            if self.current_round != 1:
                print(
                    "\nTraining round completed in "
                    + str(round(time.time() - self.round_time, 2))
                    + " seconds."
                )
            print(
                f"\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
                f"Server started {self.current_round}th round of training.\n"
                f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
            )
            # Update round start time
            self.round_time = time.time()
            return {
                "rounds": str(self.rounds),
                "c_round": str(self.current_round),
                "epochs": str(epochs),
                "batch_size": str(batch_size),
                "learning_rate": str(learning_rate),
            }

        return fit_config

    def get_eval_fn(self, ds_test):
        """Return a function to evaluate the model during federated learning.

        This function is used by the Flower framework to evaluate the model at specified
        intervals using the provided test dataset.

        Parameters
        ----------
        - ds_test (tf.data.Dataset): The dataset used for evaluation.

        Returns
        -------
        - Function: An evaluation function that takes server_round, weights, and configs
        as parameters and returns the loss and accuracy.
        """

        # pylint: disable=unused-argument
        def evaluate(servr_round, weights, configs):
            loss, acc = 0, 0
            self.current_round += 1
            if (self.current_round - 1) % self.evalution_step == 0:
                model = Network(num_classes=self.num_classes).get_evaluation_network()
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=True, name="loss"
                    ),
                    metrics=[
                        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
                    ],
                )
                model.set_weights(weights)
                loss, acc = model.evaluate(ds_test, verbose=self.verbose)
                self.final_accuracy = acc
                # Clear Memory
                clear_memory()
            return float(loss), {"accuracy": float(acc)}

        return evaluate

    def get_accuracy(self):
        """Retrieve the final accuracy achieved by the model.

        Returns the accuracy metric recorded in the last
        evaluation step of the federated learning process.

        Returns
        -------
        - float: The final accuracy value.
        """
        return self.final_accuracy


def clear_memory():
    """Clear the TensorFlow session and collect garbage.

    This function is used to free up memory by clearing the TensorFlow backend session
    and invoking garbage collection.
    """
    gc.collect()
    tf.keras.backend.clear_session()


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
        flwr_min_sample_size=cfg.server.min_sample_size,
        flwr_min_num_clients=cfg.server.num_clients,
        flwr_rounds=cfg.server.rounds,
        model_num_classes=num_classes,
        model_lr=cfg.server.learning_rate,
        model_batch_size=cfg.server.batch_size,
        model_epochs=cfg.server.train_epochs,
        model_ds_test=ds_test,
        model_verbose=cfg.server.verbose,
    )
    # Run server
    set_logger_level()
    audio_server.server_start(cfg.server.server_address)
    return f"""\nFinal Accuracy on experiment  {unique_id}:
                {audio_server.get_accuracy():.04f}\n"""


if __name__ == "__main__":
    main()
    sys.exit(0)
