import pathlib
from functools import partial

import pandas as pd
import torch
import flwr as fl
from flwr.server.strategy import FedAvg
from sklearn import preprocessing

from utils import weighted_average, steup_seed
from client import create_client
from dataset import create_dataset, create_division_list, partition_dataset, \
    partition_datasets, transform_datasets_into_dataloaders
from zip_downloader import ZipDownloader
from nist_preprocessor import NISTPreprocessor
from nist_sampler import NistSampler

DEVICE = torch.device("cpu")

if __name__ == "__main__":
    # Ensure reproducibility
    steup_seed()
    # Download and unzip the data
    print("Data downloading started")
    nist_by_class_url = "https://s3.amazonaws.com/nist-srd/SD19/by_class.zip"
    nist_by_writer_url = "https://s3.amazonaws.com/nist-srd/SD19/by_write.zip"
    nist_by_class_downloader = ZipDownloader("data/raw", nist_by_class_url)
    nist_by_writer_downloader = ZipDownloader("data/raw", nist_by_writer_url)
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
    sampled_data_info = sampler.sample("niid", 0.05, n_clients=100)
    sampled_data_info.to_csv("data/processed/niid_sampled_images_to_labels.csv")
    print("Creation of sampling information done")

    # Create a list of DataLoaders
    # Prepare the batch_size_parameter
    mini_batch_size = 10

    print("Creation of partitioned by writer_id PyTorch Datasets started")
    sampled_data_info = pd.read_csv("data/processed/niid_sampled_images_to_labels.csv")
    label_encoder = preprocessing.LabelEncoder()
    labels = label_encoder.fit_transform(sampled_data_info["character"])
    full_dataset = create_dataset(sampled_data_info, labels)
    division_list = create_division_list(sampled_data_info)
    partitioned_dataset = partition_dataset(full_dataset, division_list)
    partitioned_train, partitioned_validation, partitioned_test = partition_datasets(
        partitioned_dataset)
    trainloaders = transform_datasets_into_dataloaders(partitioned_train, batch_size=mini_batch_size)
    valloaders = transform_datasets_into_dataloaders(partitioned_validation, batch_size=mini_batch_size)
    testloaders = transform_datasets_into_dataloaders(partitioned_test, batch_size=mini_batch_size)
    print("Creation of partitioned by writer_id PyTorch Datasets done")

    # Prepare to launch flwr.simulation
    n_rounds = 1  # 00#0  # confirmed - the very last bullet point of the paper
    n_clients_per_round = 5  # confirmed - the very last bullet point of the paper
    local_lr = 0.001  # confirmed - the very last bullet point of the paper

    n_client_epochs = 5  # None
    n_client_batches = None  # 5

    client_fnc = partial(create_client,
                         trainloaders=trainloaders,
                         valloaders=valloaders,
                         testloaders=testloaders,
                         device=torch.device("cpu"),
                         num_epochs=n_client_epochs,
                         learning_rate=local_lr,
                         num_classes=62,
                         num_batches=n_client_batches)

    total_n_clients = len(trainloaders)  # the total number of clients created produced from sampling

    strategy = fl.server.strategy.FedAvg(
        min_available_clients=total_n_clients,  # min number of clients to sample from for fit and evaluate
        # Keep fraction fit low (not zero for consistency reasons with fraction_evaluate)
        # and determine number of clients by the min_fit_clients
        # (it's max of 1. fraction_fit * available clients 2. min_fit_clients)
        fraction_fit=0.001,
        min_fit_clients=n_clients_per_round,
        fraction_evaluate=0.001,
        min_evaluate_clients=n_clients_per_round,
        # evaluate_fn=evaluate_fn, #  Leave empty since it's responsible for the centralized evaluation
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    client_resources = None
    if DEVICE.type == "cuda":
        client_resources = {"num_gpus": 1}

    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fnc,
        num_clients=total_n_clients,  # total number of clients in a simulation
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )

    # keywords = ["accuracy", "test_loss", "vall_loss"]
    print(history)
    pd_history_acc = pd.DataFrame(history.metrics_distributed["accuracy"], columns=["round", "accuracy"])
    # pd_history_sth = pd.DataFrame(history.metrics_distributed["sth"], columns=["round", "sth"])
    pd_history_loss = pd.DataFrame(history.losses_distributed, columns=["round", "test_loss"])
    pd1 = pd.merge(pd_history_acc, pd_history_loss, on="round")
    print(pd1)
    pd1.to_csv("results/acc_loss.csv")
    ax = pd1[["accuracy", "test_loss"]].plot()

    # save the plot to a file
    fig = ax.get_figure()
    fig.savefig('results/acc_loss.png')
    # print(pd.merge(pd1, pd_history_sth, on="round"))
