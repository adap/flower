"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-process
your dataset (or all of the above). If the desired way of running your baseline is to
first download the dataset and partition it and then run the experiments, please
uncomment the lines below and tell us in the README.md (see the "Running the Experiment"
block) that this file should be executed first.
"""
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import shutil
import sys
import ast
np.set_printoptions(threshold=sys.maxsize)

def read_fedmlb_distribution(dataset, total_clients, alpha, data_folder="client_data"):
    file_path = os.path.join("FedMLB", data_folder, dataset, "balanced",
                             "dirichlet" + str(round(alpha, 1)) + "_clients" + str(total_clients) + ".txt")

    # reading the data from the file
    with open(file_path) as f:
        data = f.read()

    # reconstructing the data as a dictionary
    data_mlb = ast.literal_eval(data)

    return data_mlb


# def download_and_preprocess(dataset="cifar100", alpha=0.3, total_clients=100):


@hydra.main(config_path="conf", config_name="base", version_base=None)
def download_and_preprocess(cfg: DictConfig) -> None:
    """Does everything needed to get the dataset.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """

    ## print parsed config
    print(OmegaConf.to_yaml(cfg))

    dataset = cfg.dataset_config.dataset
    alpha = cfg.dataset_config.alpha_dirichlet
    total_clients = cfg.total_clients

    folder = dataset + "_mlb_dirichlet_train"
    if dataset in ["cifar100"]:
        num_classes = 100
    else:
        num_classes = 200

    # if the folder exist it is deleted and the ds partitions are re-created
    # if the folder does not exist, firstly the folder is created
    # and then the ds partitions are generated
    exist = os.path.exists(folder)
    if not exist:
        os.makedirs(folder)
    folder_path = os.path.join(folder, str(total_clients), str(round(alpha, 2)))
    exist = os.path.exists(folder_path)
    if not exist:
        os.makedirs(folder_path)
    else:
        shutil.rmtree(folder_path, ignore_errors=True)

    (x_train, y_train), (_, _) = tf.keras.datasets.cifar100.load_data()

    data_mlb = read_fedmlb_distribution(dataset, total_clients=total_clients, alpha=alpha)

    for client in data_mlb:
        list_extracted_all_labels = data_mlb[client]
        numpy_dataset_y = y_train[list_extracted_all_labels]
        numpy_dataset_x = x_train[list_extracted_all_labels]

        ds = tf.data.Dataset.from_tensor_slices((numpy_dataset_x, numpy_dataset_y))
        ds = ds.shuffle(buffer_size=4096)

        tf.data.Dataset.save(ds, path=os.path.join(folder_path, "train", str(client)))

    path = os.path.join(os.path.join(folder_path, "train"))

    list_of_narrays = []
    for sampled_client in range(0, total_clients):
        loaded_ds = tf.data.Dataset.load(
            path=os.path.join(path, str(sampled_client)), element_spec=None, compression=None, reader_func=None
        )

        print("[Client " + str(sampled_client) + "]")
        print("Cardinality: ", tf.data.experimental.cardinality(loaded_ds).numpy())

        def count_class(counts, batch, num_classes=num_classes):
            _, labels = batch
            for i in range(num_classes):
                cc = tf.cast(labels == i, tf.int32)
                counts[i] += tf.reduce_sum(cc)
            return counts

        initial_state = dict((i, 0) for i in range(num_classes))
        counts = loaded_ds.reduce(initial_state=initial_state, reduce_func=count_class)

        # print([(k, v.numpy()) for k, v in counts.items()])
        new_dict = {k: v.numpy() for k, v in counts.items()}
        # print(new_dict)
        res = np.array([item for item in new_dict.values()])
        # print(res)
        list_of_narrays.append(res)

    distribution = np.stack(list_of_narrays)
    print(distribution)
    # saving the distribution of per-label examples in a numpy file
    # this can be useful also to draw charts about the label distrib.
    path = os.path.join(folder_path, "distribution_train.npy")
    np.save(path, distribution)


if __name__ == "__main__":
    download_and_preprocess()
