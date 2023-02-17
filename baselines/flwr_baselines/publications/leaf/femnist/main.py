import pathlib
from functools import partial

import pandas as pd
import torch
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr_baselines.publications.leaf.femnist.client import full_client_fn
from flwr_baselines.publications.leaf.femnist.dataset import load_dataset, create_division_list, partition_dataset, \
    get_partitioned_train_test_dataset, transform_datasets_into_dataloaders
from flwr_baselines.publications.leaf.femnist.nist_downloader import NISTDownloader
from flwr_baselines.publications.leaf.femnist.nist_preprocessor import NISTPreprocessor
from flwr_baselines.publications.leaf.femnist.nist_sampler import NistSampler

if __name__ == "__main__":
    # # Download and unzip the data
    print("Data downloading started")
    nist_by_class_url = "https://s3.amazonaws.com/nist-srd/SD19/by_class.zip"
    nist_by_writer_url = "https://s3.amazonaws.com/nist-srd/SD19/by_write.zip"
    nist_by_class_downloader = NISTDownloader("data/raw", nist_by_class_url)
    nist_by_writer_downloader = NISTDownloader("data/raw", nist_by_writer_url)
    nist_by_class_downloader.download()
    nist_by_writer_downloader.download()
    print("Data downloading done")

    # Preprocess teh data
    print("Preprocessing of the data started")
    nist_data_path = pathlib.Path("data")
    nist_preprocessor = NISTPreprocessor(nist_data_path)
    nist_preprocessor.preprocess()
    print("Preprocessing of the data done")

    # Create information for sampling
    print("Creation of sampling information started")
    df_info_path = pathlib.Path("data/processed/resized_images_to_labels.csv")
    df_info = pd.read_csv(df_info_path, index_col=0)
    sampler = NistSampler(df_info)
    sampled_data_info = sampler.sample("niid", 0.05, 100)
    sampled_data_info.to_csv("data/processed/niid_sampled_images_to_labels.csv")
    print("Creation of sampling information done")

    # Create a list of DataLoaders
    print("Creation of partitioned by writer_id PyTorch Datasets started")
    sampled_data_info = pd.read_csv("data/processed/niid_sampled_images_to_labels.csv")
    full_dataset = load_dataset(sampled_data_info)
    division_list = create_division_list(sampled_data_info)
    partitioned_dataset = partition_dataset(full_dataset, division_list)
    partitioned_train, partitioned_test = get_partitioned_train_test_dataset(partitioned_dataset)
    trainloaders = transform_datasets_into_dataloaders(partitioned_train)
    testloaders = transform_datasets_into_dataloaders(partitioned_test)
    print("Creation of partitioned by writer_id PyTorch Datasets done")

    # todo: Finish the below. The below doesn't work rn
    # Prepare to launch flwr.simulation
    n_rounds = 1_000
    n_clients_per_round = 5
    local_lr = 0.001
    mini_batch_size = 10


    client_fnc = partial(full_client_fn,
                         trainloaders=trainloaders,
                         testloaders=testloaders,
                         device=torch.device("cpu"),
                         num_epochs=None,
                         learning_rate=None,
                         num_classes=62)
    strategy = fl.server.strategy.FedAvg(
        # fraction_fit=cfg.client_fraction,
        fraction_evaluate=0.0,
        # min_fit_clients=int(cfg.num_clients * cfg.client_fraction),
        min_evaluate_clients=0,
        # min_available_clients=cfg.num_clients,
        # evaluate_fn=evaluate_fn,
        # evaluate_metrics_aggregation_fn=utils.weighted_average,
    )



