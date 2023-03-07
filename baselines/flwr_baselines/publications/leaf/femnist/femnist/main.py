import pathlib
from functools import partial
from logging import INFO

import flwr as fl
import hydra
import pandas as pd
from client import create_client
from constants import DEVICE, RANDOM_SEED
from dataset.dataset import (
    create_dataset,
    create_division_list,
    partition_dataset,
    partition_datasets,
    transform_datasets_into_dataloaders,
)
from dataset.nist_preprocessor import NISTPreprocessor
from dataset.nist_sampler import NistSampler
from dataset.zip_downloader import ZipDownloader
from fedavg_same_clients import FedAvgSameClients
from flwr.common.logger import log
from flwr.server.strategy import FedAvg
from omegaconf import DictConfig
from sklearn import preprocessing
from utils import setup_seed, weighted_average


@hydra.main(config_path="../conf", version_base=None)
def main(cfg: DictConfig):
    # Ensure reproducibility
    setup_seed(RANDOM_SEED)

    # Download and unzip the data
    log(INFO, "NIST data downloading started")
    nist_by_class_url = "https://s3.amazonaws.com/nist-srd/SD19/by_class.zip"
    nist_by_writer_url = "https://s3.amazonaws.com/nist-srd/SD19/by_write.zip"
    nist_by_class_downloader = ZipDownloader("data/raw", nist_by_class_url)
    nist_by_writer_downloader = ZipDownloader("data/raw", nist_by_writer_url)
    nist_by_class_downloader.download()
    nist_by_writer_downloader.download()
    log(INFO, "NIST data downloading done")

    # Preprocess the data
    log(INFO, "Preprocessing of the NIST data started")
    nist_data_path = pathlib.Path("data")
    nist_preprocessor = NISTPreprocessor(nist_data_path)
    nist_preprocessor.preprocess()
    log(INFO, "Preprocessing of the NIST data done")

    # Create information for sampling
    log(INFO, "Creation of the sampling information started")
    df_info_path = pathlib.Path("data/processed/resized_images_to_labels.csv")
    df_info = pd.read_csv(df_info_path, index_col=0)
    sampler = NistSampler(df_info)
    sampled_data_info = sampler.sample(cfg.distribution_type, cfg.dataset_fraction)
    sampled_data_info_path = pathlib.Path(
        f"data/processed/{cfg.distribution_type}_sampled_images_to_labels.csv"
    )
    sampled_data_info.to_csv(sampled_data_info_path)
    log(INFO, "Creation of the sampling information done")

    # Create a list of DataLoaders
    log(INFO, "Creation of the partitioned by writer_id PyTorch Datasets started")
    sampled_data_info = pd.read_csv(sampled_data_info_path)
    label_encoder = preprocessing.LabelEncoder()
    labels = label_encoder.fit_transform(sampled_data_info["character"])
    full_dataset = create_dataset(sampled_data_info, labels)
    division_list = create_division_list(sampled_data_info)
    partitioned_dataset = partition_dataset(full_dataset, division_list)
    partitioned_train, partitioned_validation, partitioned_test = partition_datasets(
        partitioned_dataset, random_seed=RANDOM_SEED
    )
    trainloaders = transform_datasets_into_dataloaders(
        partitioned_train, batch_size=cfg.batch_size
    )
    valloaders = transform_datasets_into_dataloaders(
        partitioned_validation, batch_size=cfg.batch_size
    )
    testloaders = transform_datasets_into_dataloaders(
        partitioned_test, batch_size=cfg.batch_size
    )
    log(INFO, "Creation of the partitioned by writer_id PyTorch Datasets done")

    # The total number of clients created produced from sampling differs (on different random seeds)
    total_n_clients = len(trainloaders)

    client_fnc = partial(
        create_client,
        trainloaders=trainloaders,
        valloaders=valloaders,
        testloaders=testloaders,
        device=DEVICE,
        num_epochs=cfg.epochs_per_round,
        learning_rate=cfg.learning_rate,
        # There exist other variants of the FEMNIST dataset with different # of classes
        num_classes=62,
        num_batches=cfg.batches_per_round,
    )

    if cfg.same_train_test_clients:
        #  Assign reference to a class
        flwr_strategy = FedAvgSameClients
    else:
        flwr_strategy = FedAvg

    strategy = flwr_strategy(
        min_available_clients=total_n_clients,
        # min number of clients to sample from for fit and evaluate
        # Keep fraction fit low (not zero for consistency reasons with fraction_evaluate)
        # and determine number of clients by the min_fit_clients
        # (it's max of 1. fraction_fit * available clients 2. min_fit_clients)
        fraction_fit=0.001,
        min_fit_clients=cfg.num_clients_per_round,
        fraction_evaluate=0.001,
        min_evaluate_clients=cfg.num_clients_per_round,
        # evaluate_fn=None, #  Leave empty since it's responsible for the centralized evaluation
        fit_metrics_aggregation_fn=weighted_average,  # todo: collect the fit metrics
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    client_resources = None
    if DEVICE.type == "cuda":
        client_resources = {"num_gpus": 1.0}

    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fnc,
        num_clients=total_n_clients,  # total number of clients in a simulation
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )

    log(INFO, history)
    pd_history_acc = pd.DataFrame(
        history.metrics_distributed["accuracy"], columns=["round", "test_accuracy"]
    )
    pd_history_loss = pd.DataFrame(
        history.losses_distributed, columns=["round", "test_loss"]
    )
    print(pd_history_acc)
    print(pd_history_loss)

    results_dir_path = pathlib.Path(cfg.results_dir_path)
    if not results_dir_path.exists():
        results_dir_path.mkdir(parents=True)

    pd_history_acc.to_csv(results_dir_path / "test_accuracy.csv")
    ax = pd_history_acc["test_accuracy"].plot()
    fig = ax.get_figure()
    fig.savefig(results_dir_path / "test_accuracy.jpg", dpi=200)

    pd_history_loss.to_csv(results_dir_path / "train_loss.csv")
    ax = pd_history_loss["test_loss"].plot()
    fig = ax.get_figure()
    fig.savefig(results_dir_path / "test_accuracy.jpg", dpi=200)


if __name__ == "__main__":
    main()
