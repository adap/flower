"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""

import os
import shutil
from typing import Callable, Dict, Optional, Tuple

# these are the basic packages you'll need here
# feel free to remove some if aren't needed
import flwr
import hydra
import numpy as np
import tensorflow as tf
from flwr.common import NDArrays, Scalar
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import fedmlb.dataset as fedmlb_datasets
import fedmlb.dataset_preparation as fedmlb_ds_preparation
import fedmlb.models as fedmlb_models
from fedmlb.client import TFClient
from fedmlb.fedavg_kd_model import FedAvgKDModel
from fedmlb.fedmlb_model import FedMLBModel
from fedmlb.models import create_resnet18
from fedmlb.server import MyServer
from fedmlb.utils import (
    dic_load,
    dic_save,
    get_cpu_memory,
    get_gpu_memory,
    save_results_as_pickle,
)

# Make TensorFlow logs less verbose


# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
enable_tf_gpu_growth()

TEST_BATCH_SIZE = 256


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:  # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    print("[Start Simulation]")
    # Print parsed config
    print(OmegaConf.to_yaml(cfg))

    def element_norm_fn_cifar100(image, label):
        """Normalize input images (CIFAR100)."""
        norm_layer = tf.keras.layers.Normalization(
            mean=[0.5071, 0.4865, 0.4409],
            variance=[np.square(0.2673), np.square(0.2564), np.square(0.2762)],
        )
        return norm_layer(tf.cast(image, tf.float32) / 255.0), label

    def element_fn_norm_tiny_imagenet(image, label):
        norm_layer = tf.keras.layers.Normalization(
            mean=[0.4802, 0.4481, 0.3975],
            variance=[np.square(0.2770), np.square(0.2691), np.square(0.2821)],
        )
        return norm_layer(tf.cast(image, tf.float32) / 255.0), tf.expand_dims(
            label, axis=-1
        )

    def get_evaluate_fn(
        model: tf.keras.Model, save_path: str, dataset: str, starting_round: int
    ) -> Callable[
        [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
    ]:
        """Return an evaluation function for server-side evaluation."""
        if dataset in ["cifar100"]:
            (_, _), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
            test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
            test_ds = test_ds.map(element_norm_fn_cifar100).batch(TEST_BATCH_SIZE)
        else:  # tiny-imagenet
            # center_crop = tf.keras.layers.CenterCrop(64, 64)
            center_crop = fedmlb_datasets.PaddedCenterCropCustom(64, 64)

            def center_crop_data(image, label):
                return center_crop(image), label

            test_ds = fedmlb_ds_preparation.load_test_dataset_tiny_imagenet()
            test_ds = (
                test_ds.map(element_fn_norm_tiny_imagenet)
                .map(center_crop_data)
                .batch(TEST_BATCH_SIZE)
            )

        # creating a tensorboard writer to log results
        # then results can be monitored in real-time with tensorboard
        # running the command:
        # tensorboard --logdir [HERE_THE_PATH_OF_TF_BOARD_LOGS]
        global_summary_writer = tf.summary.create_file_writer(save_path)

        # The `evaluate` function will be called after every round
        def evaluate(
            server_round: int,
            parameters: NDArrays,
            config: Dict[str, Scalar],  # pylint: disable=unused-argument
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            model.set_weights(parameters)  # Update model with the latest parameters
            loss, accuracy = model.evaluate(
                test_ds,
            )

            with global_summary_writer.as_default():
                tf.summary.scalar("loss", loss, step=server_round)
                tf.summary.scalar("accuracy", accuracy, step=server_round)

                if cfg.logging_memory_usage:
                    # logging metrics on memory usage
                    gpu_free_memory = get_gpu_memory()
                    cpu_free_memory = get_cpu_memory()
                    tf.summary.scalar(
                        "cpu_free_mem", cpu_free_memory, step=server_round
                    )
                    tf.summary.scalar(
                        "gpu_free_mem", gpu_free_memory, step=server_round
                    )

            # saving the checkpoint before the end of simulation
            if cfg.save_checkpoint and server_round == (
                cfg.num_rounds + starting_round - 1
            ):
                path = os.path.join(
                    save_path_checkpoints,
                    "checkpoints_R" + str(server_round),
                    "server_model",
                )
                server_model.save_weights(path)

                path = os.path.join(save_path_checkpoints, "dict_info")
                dic_save({"checkpoint_round": server_round}, path)

            return loss, {"accuracy": accuracy}

        return evaluate

    def fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return training configuration dict for each round."""
        config = {
            "current_round": server_round,
            "local_epochs": 5,
            "exp_decay": cfg.exp_decay,
            "lr_client_initial": cfg.lr_client,
        }
        return config

    ray_init_args = {"include_dashboard": False}
    # Parse input parameters
    algorithm = cfg.algorithm
    random_seed = cfg.random_seed
    lr_client = cfg.lr_client
    # exp_decay = cfg.exp_decay
    clipnorm = cfg.clipnorm
    l2_weight_decay = cfg.l2_weight_decay
    lambda_1 = cfg.lambda_1
    lambda_2 = cfg.lambda_2
    alpha_dirichlet = cfg.dataset_config.alpha_dirichlet
    local_updates = cfg.local_updates
    local_epochs = cfg.local_epochs
    total_clients = cfg.total_clients
    dataset = cfg.dataset_config.dataset
    restart_from_checkpoint = cfg.restart_from_checkpoint
    batch_size = cfg.batch_size

    # if cfg.batch_size is set to null,
    # local_batch_size = round(local_examples * local_epochs / local_updates)
    # if cfg.batch_size is set to a value, it will be used as local_batch_size
    if batch_size is None:
        local_batch_size_or_k_defined = "K_" + str(local_updates)
    else:
        local_batch_size_or_k_defined = "batch_size_" + str(batch_size)

    if dataset in ["cifar100"]:
        num_classes = 100
        input_shape = (None, 32, 32, 3)
    else:  # tiny-imagenet
        num_classes = 200
        input_shape = (None, 64, 64, 3)

    def client_fn(cid: str) -> TFClient:
        """Instantiate TF Client."""
        local_examples = fedmlb_datasets.load_selected_client_statistics(
            int(cid),
            total_clients=total_clients,
            alpha=alpha_dirichlet,
            dataset=dataset,
        )

        # if cfg.batch_size is set to null,
        # local_batch_size = round(local_examples * local_epochs / local_updates)
        # if cfg.batch_size is set to a value, it will be used as local_batch_size
        if batch_size is None:
            local_batch_size = round(local_examples * local_epochs / local_updates)
        else:
            local_batch_size = batch_size

        training_dataset = fedmlb_datasets.load_client_datasets_from_files(
            dataset=dataset,
            sampled_client=int(cid),
            total_clients=total_clients,
            batch_size=local_batch_size,
            alpha=alpha_dirichlet,
            seed=random_seed,
        )

        if algorithm in ["FedAvg"]:
            client_model = fedmlb_models.create_resnet18(
                num_classes=num_classes,
                input_shape=input_shape,
                norm="group",
                seed=random_seed,
            )
            client_model.compile(
                optimizer=tf.keras.optimizers.SGD(
                    learning_rate=lr_client,
                    clipnorm=clipnorm,
                    weight_decay=l2_weight_decay,
                ),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True,
                    reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                ),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            )

        elif algorithm in ["FedMLB"]:
            local_model = fedmlb_models.create_resnet18_mlb(
                num_classes=num_classes,
                input_shape=input_shape,
                norm="group",
                seed=random_seed,
            )
            global_model = fedmlb_models.create_resnet18_mlb(
                num_classes=num_classes,
                input_shape=input_shape,
                norm="group",
                seed=random_seed,
            )
            kd_loss = tf.keras.losses.KLDivergence(
                reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
            )

            client_model = FedMLBModel(
                local_model, global_model, kd_loss, lambda_1, lambda_2
            )
            client_model.compile(
                optimizer=tf.keras.optimizers.SGD(
                    learning_rate=lr_client,
                    clipnorm=clipnorm,
                    weight_decay=l2_weight_decay,
                ),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True,
                    reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                ),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            )

        else:  # algorithm in ["FedAvg+KD"]:
            local_model = fedmlb_models.create_resnet18(
                num_classes=num_classes,
                input_shape=input_shape,
                norm="group",
                seed=random_seed,
            )
            global_model = fedmlb_models.create_resnet18(
                num_classes=num_classes,
                input_shape=input_shape,
                norm="group",
                seed=random_seed,
            )
            kd_loss = tf.keras.losses.KLDivergence(
                reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
            )
            client_model = FedAvgKDModel(local_model, global_model, kd_loss, gamma=0.2)

            client_model.compile(
                optimizer=tf.keras.optimizers.SGD(
                    learning_rate=lr_client,
                    clipnorm=clipnorm,
                    weight_decay=l2_weight_decay,
                ),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True,
                    reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                ),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            )

        client = TFClient(training_dataset, client_model, local_examples, algorithm)

        # Create and return client
        return client

    server_model = create_resnet18(
        num_classes=num_classes,
        input_shape=input_shape,
        norm="group",
        seed=cfg.random_seed,
    )
    server_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    save_path_logging = os.path.join(
        "fedmlb",
        "tb_logging",
        dataset,
        "resnet18",
        algorithm,
        local_batch_size_or_k_defined,
        str(total_clients) + "_clients",
        "dir_" + str(round(alpha_dirichlet, 1)),
        "seed_" + str(random_seed),
    )

    save_path_checkpoints = os.path.join(
        "fedmlb",
        "model_checkpoints",
        dataset,
        "resnet18",
        algorithm,
        local_batch_size_or_k_defined,
        str(total_clients) + "_clients",
        "dir_" + str(round(alpha_dirichlet, 1)),
        "seed_" + str(random_seed),
    )

    starting_round = 1
    if restart_from_checkpoint:
        # if there is a checkpoint and restart_from_checkpoint is True
        # the training restart from the state saved in the most recent checkpoint
        # i.e., the one indicated in a dictionary named dict_info
        path = os.path.join(save_path_checkpoints, "dict_info.pickle")
        last_checkpoint = dic_load(path)["checkpoint_round"]
        if last_checkpoint:
            print(f"Loading saved checkpoint round {last_checkpoint}")
            path = os.path.join(
                save_path_checkpoints,
                "checkpoints_R" + str(last_checkpoint),
                "server_model",
            )
            server_model.load_weights(path)
            starting_round = last_checkpoint + 1
    else:
        # this will delete the checkpoints of previous simulations for that config
        exist = os.path.exists(save_path_checkpoints)
        if exist:
            shutil.rmtree(save_path_checkpoints, ignore_errors=True)

    tf.keras.utils.set_random_seed(cfg.random_seed * starting_round)
    params = server_model.get_weights()

    strategy = instantiate(
        cfg.strategy,
        initial_parameters=flwr.common.ndarrays_to_parameters(params),
        evaluate_fn=get_evaluate_fn(
            server_model, save_path_logging, dataset, starting_round
        ),
        on_fit_config_fn=fit_config,
    )

    # my_server = MyServer(cfg.starting_round)
    my_server = MyServer(strategy=strategy, starting_round=starting_round)
    # Start Flower simulation
    history = flwr.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.total_clients,
        client_resources={
            "num_cpus": cfg.client_resources.num_cpus,
            "num_gpus": cfg.client_resources.num_gpus,
        },
        server=my_server,
        config=flwr.server.ServerConfig(num_rounds=cfg.num_rounds),
        ray_init_args=ray_init_args,
        actor_kwargs={
            "on_actor_init_fn": enable_tf_gpu_growth
            # Enable GPU growth upon actor init
            # does nothing if `num_gpus` in client_resources is 0.0
        },
    )

    # Experiment completed. Now we save the results and
    # generate plots using the `history`
    print("................")
    print(history)

    # Hydra automatically creates an output directory
    # Let's retrieve it and save some results there
    save_path = HydraConfig.get().runtime.output_dir

    # save results as a Python pickle using a file_path
    # the directory created by Hydra for each run
    save_results_as_pickle(history, file_path=save_path)


if __name__ == "__main__":
    main()
