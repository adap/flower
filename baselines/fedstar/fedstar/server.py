##########################################
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
GRPC_MAX_MESSAGE_LENGTH: int = 536_870_912
##########################################
import argparse
import pathlib
import uuid
import gc
import time
import flwr
import logging
import tensorflow as tf
from fedstar.model import Network
from fedstar.data import DataBuilder
import hydra

parent_path = os.getcwd()

class AudioServer:
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
        self.min_sample_size = flwr_min_sample_size
        self.min_num_clients = flwr_min_num_clients
        self.rounds = flwr_rounds
        # Model Parameters
        self.num_classes = model_num_classes
        self.lr = model_lr
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
        print("|"*50)
        print(server_address)
        flwr.server.start_server(
            server_address="0.0.0.0:8080",
            server=flwr.server.Server(
                client_manager=self.client_manager, strategy=self.strategy
            ),
            config=flwr.server.ServerConfig(num_rounds=self.rounds),
            strategy=self.strategy,
            grpc_max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        )

    def get_on_fit_config_fn(self):
        def fit_config(
            rounds=self.rounds,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.lr,
        ):
            if self.current_round != 1:
                print(
                    "\nTraining round completed in "
                    + str(round(time.time() - self.round_time, 2))
                    + " seconds."
                )
            print(
                "\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
                + "               Server started "
                + str(self.current_round)
                + "th round of training.\n"
                + "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
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
        import tensorflow as tf

        def evaluate(servr_round, weights, configs):
            loss, acc = 0, 0
            self.current_round += 1
            if (self.current_round - 1) % self.evalution_step == 0:
                model = Network(num_classes=self.num_classes).get_evaluation_network()
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(self.lr),
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
        return self.final_accuracy


def clear_memory():
    import tensorflow as tf

    gc.collect()
    tf.keras.backend.clear_session()


def set_logger_level():
    if "flower" in [
        logging.getLogger(name).__repr__()[8:].split(" ")[0]
        for name in logging.root.manager.loggerDict
    ]:
        logger = logging.getLogger("flower")
        logger.setLevel(logging.INFO)


def set_gpu_limits(gpu_id, gpu_memory):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    import tensorflow as tf

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
        except RuntimeError as e:
            print(e)

@hydra.main(config_path="conf", config_name="base")
def main(cfg):
    # Set Experiment Parameters
    unique_id = str(uuid.uuid1())
    dataset_name = cfg["server"]["dataset_name"]
    # Notify Experiment ID to console.
    print(
        "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
        + " Experiment ID : "
        + unique_id
        + "\n"
        + "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    )
    # GPU Setup
    set_gpu_limits(gpu_id="0", gpu_memory=cfg["server"]["gpu_memory"])
    # Load Test Dataset
    ds_test, num_classes = DataBuilder.get_ds_test(
        parent_path=os.path.join(parent_path,"fedstar"), data_dir=dataset_name, batch_size=cfg["server"]['batch_size'], buffer=1024, seed=cfg["server"]['seed']
    )
    # Create Server Object
    audio_server = AudioServer(
        flwr_evalution_step=cfg["server"]["eval_step"],
        flwr_min_sample_size=cfg["server"]["min_sample_size"],
        flwr_min_num_clients=cfg["server"]["num_clients"],
        flwr_rounds=cfg["server"]["rounds"],
        model_num_classes=num_classes,
        model_lr=cfg["server"]["learning_rate"],
        model_batch_size=cfg["server"]["batch_size"],
        model_epochs=cfg["server"]["train_epochs"],
        model_ds_test=ds_test,
        model_verbose=cfg["server"]["verbose"],
    )
    # Run server
    set_logger_level()
    audio_server.server_start(cfg["server"]["server_address"])
    return f"\nFinal Accuracy on experiment  {unique_id}: {audio_server.get_accuracy():.04f}\n"


if __name__ == "__main__":
    main()
    sys.exit(0)
