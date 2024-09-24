"""Required imports for clients.py script."""

import multiprocessing
import os
import sys
from multiprocessing import Process

import flwr as fl
import hydra
import tensorflow as tf
from omegaconf import OmegaConf

from fedstar.client import AudioClient, set_logger_level

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"

multiprocessing.set_start_method("spawn", force=True)

parent_path = os.getcwd()


class Client(Process):
    """Client class acts as handler for client class.

    It manages the client creation and also manages resource allocation
    """

    def __init__(self, queue):
        Process.__init__(self)
        self.queue = queue

    def run(self):
        """Run client processes from a configuration queue.

        Continuously retrieves configurations from a queue and initiates an AudioClient
        for each. Configures GPU settings, creates and starts the client. The loop ends
        when a `None` configuration is encountered, indicating no more tasks.

        Each client is started with settings from the config, including client ID,
        server address, dataset directory, and other parameters.
        The method marks each task as done upon client completion.

        Assumes the existence of `queue`, `AudioClient`, and `Client.setup_gpu`.
        Designed for threading or multiprocessing.
        """
        while True:
            cfg = self.queue.get()
            if cfg is None:
                self.queue.task_done()
                break
            # Configure GPU
            Client.setup_gpu(
                gpu=cfg["gpu_id"],
                gpu_memory=cfg["gpu_memory"],
                client_id=cfg["client_id"],
            )
            process_path = (os.sep).join(parent_path.split(os.sep)) + os.sep
            # Create Client
            client = AudioClient(
                client_id=cfg["client_id"],
                num_clients=cfg["num_clients"],
                dataset_dir=cfg["dataset_dir"],
                parent_path=process_path,
                l_per=cfg["l_per"],
                u_per=cfg["u_per"],
                batch_size=cfg["batch_size"],
                learning_rate=cfg["learning_rate"],
                verbose=cfg["verbose"],
                seed=cfg["seed"],
                fedstar=cfg["fedstar"],
                class_distribute=cfg["class_distribute"],
                balance_dataset=cfg["balance_dataset"],
            )
            # Start Client
            try:
                if bool(cfg["verbose"]):
                    print(
                        f"""This is client {client.client_id}
                        with train dataset of {client.num_examples_train}
                        elements."""
                    )
                set_logger_level()
                fl.client.start_numpy_client(
                    server_address=cfg["server_address"], client=client
                )
                print("Client Shutdown.")
            except RuntimeError as runtime_error:
                print(runtime_error)
            # Return job done once client is terminated.
            self.queue.task_done()

    @staticmethod
    def setup_gpu(gpu, gpu_memory, client_id):
        """Configure GPU settings for TensorFlow.

        Sets the environment variable for CUDA_VISIBLE_DEVICES based on the provided
        GPU identifier. If the GPU is specified and available, it configures TensorFlow
        to limit the GPU memory usage to the specified amount. If no GPUs are available,
        or if 'gpu' is None, TensorFlow will default to CPU usage.

        Parameters
        ----------
        - gpu (str): Identifier for the GPU to be used. Set to None for CPU usage.
        - gpu_memory (int): The maximum amount of memory (in MB) to be
                            allocated to the GPU.

        Note:
        - This method is specific to TensorFlow's handling of GPU resources.
        - It should be called before initializing any TensorFlow models or operations.
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu if gpu is not None else ""

        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            print(f"No GPU's available. Client {client_id} will run on CPU.")
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


# pylint: disable=unsubscriptable-object
def distribute_gpus(num_clients, client_memory=1024):
    """Distribute GPU resources among multiple clients.

    Parameters
    ----------
    - num_clients (int): Number of clients to distribute GPUs among.
    - client_memory (int): The amount of memory (in MB) to allocate per client.

    Returns
    -------
    - List of GPU IDs or None for each client, based on memory availability.
    """
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if not gpus:
        # Scenario 1: No GPUs available, return None for all clients
        return [None] * num_clients

    # Manually specify the memory you want to expose in each GPU
    gpu_total_mem = [4000]  # Adjust based on your GPUs

    if gpus and len(gpu_total_mem) == 0:
        print("If you want to enable GPU training, please set `gpu_total_mem`")

    gpu_free_mem = gpu_total_mem.copy()
    clients_gpu = [None] * num_clients

    for client_id in range(num_clients):
        suitable_gpus = [
            i for i, mem in enumerate(gpu_free_mem) if mem >= client_memory
        ]
        if suitable_gpus:
            # Scenario 2: Allocate GPU if memory is available
            chosen_gpu = min(
                suitable_gpus, key=lambda i: gpu_total_mem[i] - gpu_free_mem[i]
            )
            gpu_free_mem[chosen_gpu] -= client_memory
            gpu_id = gpus[chosen_gpu].name.split(":")[-1]
            clients_gpu[client_id] = gpu_id
        else:
            # Scenario 3: No sufficient memory in GPUs, allocate CPU (None)
            clients_gpu[client_id] = None
    print("*" * 50)
    print(clients_gpu)
    return clients_gpu


# pylint: disable=expression-not-assigned
@hydra.main(config_path="conf", config_name="table3", version_base=None)
def main(cfg):
    """Initialize and run a multi-processing client setup using Hydra configuration.

    Loads the configuration using Hydra, distributes GPUs among clients based on
    the configuration, and sets up data for each client. The function then creates
    and starts a pool of `Client` objects, each in its own process, and feeds them
    client-specific configurations. It waits for all clients to complete their tasks.

    Parameters
    ----------
    - cfg: The configuration object provided by Hydra, containing settings for
           the number of clients, server address, dataset information, GPU memory,
           and other client-related configurations.

    The configuration (`cfg`) is expected to follow a specific structure as defined
    in the Hydra configuration files under 'conf/table3'.
    """
    print(OmegaConf.to_yaml(cfg))

    clients_gpu = distribute_gpus(
        num_clients=cfg.client.num_clients,
        client_memory=cfg.client.gpu_memory,
    )
    dataset_name = cfg.client.dataset_name

    # Load Configurations of Clients
    clients_data = [
        {
            "client_id": i,
            "num_clients": cfg.client.num_clients,
            "server_address": cfg.client.server_address,
            "dataset_dir": dataset_name,
            "batch_size": cfg.client.batch_size,
            "learning_rate": cfg.client.learning_rate,
            "gpu_id": clients_gpu[i],
            "gpu_memory": cfg.client.gpu_memory,
            "seed": cfg.client.seed,
            "l_per": cfg.client.l_per,
            "u_per": cfg.client.u_per,
            "fedstar": cfg.client.fedstar,
            "class_distribute": cfg.client.class_distribute,
            "balance_dataset": cfg.dataset_name == "speech_commands",
            "verbose": cfg.client.verbose,
        }
        for i in range(cfg.client.num_clients)
    ]

    # Start Multi-processing Clients and wait for them to finish
    clients_queue = multiprocessing.JoinableQueue()
    clients = [Client(clients_queue) for i in range(cfg.client.num_clients)]
    [client.start() for client in clients]
    [clients_queue.put(client) for client in clients_data]
    [clients_queue.put(None) for i in range(cfg.client.num_clients)]
    clients_queue.join()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
    sys.exit(0)
