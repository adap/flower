"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""
# these are the basic packages you'll need here
# feel free to remove some if aren't needed
import flwr
from flwr.common import NDArrays, Scalar
from typing import Dict, Optional, Tuple
import tensorflow as tf
import os
import numpy as np
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from FedMLB.client import TFClient
import FedMLB.dataset as fedmlb_datasets
from FedMLB.FedMLBModel import FedMLBModel
from FedMLB.FedAvgKDModel import FedAvgKDModel
import FedMLB.models as fedmlb_models
from FedMLB.utils import save_results_as_pickle
from FedMLB.models import create_resnet18

TEST_BATCH_SIZE = 256

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
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
        """Utility function to normalize input images (CIFAR100)."""
        norm_layer = tf.keras.layers.Normalization(mean=[0.5071, 0.4865, 0.4409],
                                                   variance=[np.square(0.2673),
                                                             np.square(0.2564),
                                                             np.square(0.2762)])
        return norm_layer(tf.cast(image, tf.float32) / 255.0), label


    def get_evaluate_fn(model, save_path):
        """Return an evaluation function for server-side evaluation."""

        (_, _), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_ds = test_ds.map(element_norm_fn_cifar100).batch(TEST_BATCH_SIZE)

        # creating a tensorboard writer to log results
        # then results can be monitored in real-time with tensorboard
        # running the command:
        # tensorboard --logdir [HERE_THE_PATH_OF_TF_BOARD_LOGS]
        global_summary_writer = tf.summary.create_file_writer(save_path)

        # The `evaluate` function will be called after every round
        def evaluate(
                server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            model.set_weights(parameters)  # Update model with the latest parameters
            loss, accuracy = model.evaluate(test_ds)

            with global_summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=server_round)
                tf.summary.scalar('accuracy', accuracy, step=server_round)

            return loss, {"accuracy": accuracy}

        return evaluate

    def fit_config(server_round: int):
        """Return training configuration dict for each round."""
        config = {
            "current_round": server_round,
            "local_epochs": 5,
        }
        return config


    ray_init_args = {"include_dashboard": False}
    # Parse input parameters
    algorithm = cfg.algorithm
    random_seed = cfg.random_seed
    lr_client = cfg.lr_client
    clipnorm = cfg.clipnorm
    l2_weight_decay = cfg.l2_weight_decay
    lambda_1 = cfg.lambda_1
    lambda_2 = cfg.lambda_2
    alpha_dirichlet = cfg.dataset_config.alpha_dirichlet
    local_updates = cfg.local_updates
    local_epochs = cfg.local_epochs
    total_clients = cfg.total_clients
    dataset = cfg.dataset_config.dataset
    if dataset in ["cifar100"]:
        num_classes = 100
    else: # tiny-imagenet
        num_classes = 200


    def client_fn(cid) -> TFClient:
        # print(f"{cid}")

        local_examples = fedmlb_datasets.load_selected_client_statistics(int(cid), total_clients=total_clients,
                                                                         alpha=alpha_dirichlet, dataset=dataset)
        # print(local_examples)
        local_batch_size = round(local_examples * local_epochs / local_updates)

        training_dataset = fedmlb_datasets.load_client_datasets_from_files(
            dataset=dataset,
            sampled_client=int(cid),
            total_clients=total_clients,
            batch_size=local_batch_size,
            alpha=alpha_dirichlet,
            seed=random_seed)

        if algorithm in ["FedAvg"]:
            client_model = fedmlb_models.create_resnet18(num_classes=num_classes, input_shape=(None, 32, 32, 3), norm="group",
                                                         seed=random_seed)
            client_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_client,
                                                                   clipnorm=clipnorm,
                                                                   weight_decay=l2_weight_decay),
                                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                                    reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
                                 metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])

        elif algorithm in ["FedMLB"]:
            local_model = fedmlb_models.create_resnet18_mlb(num_classes=num_classes, input_shape=(None, 32, 32, 3),
                                                            norm="group",
                                                            seed=random_seed)
            global_model = fedmlb_models.create_resnet18_mlb(num_classes=num_classes, input_shape=(None, 32, 32, 3),
                                                             norm="group",
                                                             seed=random_seed)
            client_model = FedMLBModel(local_model, global_model)
            client_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_client,
                                                                   clipnorm=clipnorm,
                                                                   weight_decay=l2_weight_decay),
                                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                                    reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
                                 metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                                 kd_loss=tf.keras.losses.KLDivergence(
                                     reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
                                 lambda_1=lambda_1,
                                 lambda_2=lambda_2)

        elif algorithm in ["FedAvg+KD"]:
            local_model = fedmlb_models.create_resnet18(num_classes=num_classes, input_shape=(None, 32, 32, 3),
                                                            norm="group",
                                                            seed=random_seed)
            global_model = fedmlb_models.create_resnet18(num_classes=num_classes, input_shape=(None, 32, 32, 3),
                                                             norm="group",
                                                             seed=random_seed)
            client_model = FedAvgKDModel(local_model, global_model)
            client_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_client,
                                                                   clipnorm=clipnorm,
                                                                   weight_decay=l2_weight_decay),
                                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                                    reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
                                 metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                                 kd_loss=tf.keras.losses.KLDivergence(
                                     reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
                                 gamma=0.2)

        client = TFClient(training_dataset, client_model, local_examples, algorithm)

        # Create and return client
        return client

    tf.keras.utils.set_random_seed(cfg.random_seed)
    server_model = create_resnet18(num_classes=num_classes, input_shape=(None, 32, 32, 3), norm="group",
                                                    seed=cfg.random_seed)
    server_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
    )
    params = server_model.get_weights()

    save_path = os.path.join("FedMLB", "tb_logging", dataset, "resnet18", algorithm, str(total_clients) + "_clients",
                                       "dir_" + str(round(alpha_dirichlet, 1)), "seed_" + str(random_seed))
    strategy = instantiate(
        cfg.strategy,
        initial_parameters=flwr.common.ndarrays_to_parameters(params),
        evaluate_fn=get_evaluate_fn(server_model, save_path),
        on_fit_config_fn=fit_config,
    )

    # Start Flower simulation
    history = flwr.simulation.start_simulation(
        client_fn=client_fn,
        clients_ids=range(0, cfg.total_clients),
        num_clients=cfg.total_clients,
        client_resources={"num_cpus": 0.28},
        config=flwr.server.ServerConfig(num_rounds=cfg.num_rounds),
        ray_init_args=ray_init_args,
        strategy=strategy
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
    save_results_as_pickle(history, file_path=save_path, extra_results={})

if __name__ == "__main__":
    main()
