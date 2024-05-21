"""Handle the dataset partitioning of the datasets being used in simulations.

Datasets: CIFAR-100 and TinyImagenet.
"""

import ast
import os
import shutil
import sys
from collections import defaultdict

import hydra
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf
from PIL import Image

np.set_printoptions(threshold=sys.maxsize)

IMAGENET_BASE_PATH = "./tiny-imagenet-200/"


def read_fedmlb_distribution(
    dataset: str, total_clients: int, alpha: float, data_folder: str = "client_data"
):
    """Read the per-client distribution of labels from file.

    Parameters
    ----------
    dataset : str
        Dataset as string.
    total_clients : int
        Total clients of the simulation.
    alpha: float
        Concentration parameter of Dirichlet distribution for label skew.
    data_folder: str
        Name of the folder that contains the files to be read.
    """
    if dataset in ["tiny-imagenet"]:
        dataset = "Tiny-ImageNet"

    file_path = os.path.join(
        "fedmlb",
        data_folder,
        dataset,
        "balanced",
        "dirichlet" + str(round(alpha, 1)) + "_clients" + str(total_clients) + ".txt",
    )

    # reading the data from the file
    with open(file_path) as distribution_file:
        data = distribution_file.read()

    # reconstructing the data as a dictionary
    data_mlb = ast.literal_eval(data)

    return data_mlb


class TinyImageNetPaths:
    """Reader for tiny-imagenet dataset on disk.

    Adapted from:
    https://github.com/jinkyu032/FedMLB/blob/main/datasets/tiny_imagenet.py
    """

    def __init__(self):
        root_dir = IMAGENET_BASE_PATH
        train_path = os.path.join(root_dir, "train")
        val_path = os.path.join(root_dir, "val")
        test_path = os.path.join(root_dir, "test")

        wnids_path = os.path.join(root_dir, "wnids.txt")
        words_path = os.path.join(root_dir, "words.txt")

        self._make_paths(train_path, val_path, test_path, wnids_path, words_path)

    def _make_paths(  # pylint: disable=too-many-arguments, disable=too-many-locals
        self, train_path, val_path, test_path, wnids_path, words_path
    ):
        self.ids = []
        with open(wnids_path, "r") as idf:
            for nid in idf:
                nid = nid.strip()
                self.ids.append(nid)
        self.nid_to_words = defaultdict(list)
        with open(words_path, "r") as words_file:
            for line in words_file:
                nid, labels = line.split("\t")
                labels = [x.strip() for x in labels.split(",")]
                self.nid_to_words[nid].extend(labels)

        self.paths = {
            "train": [],  # [img_path, id, nid, box]
            "val": [],  # [img_path, id, nid, box]
            "test": [],  # img_path
        }

        # Get the test paths
        self.paths["test"] = [os.path.join(test_path, x) for x in os.listdir(test_path)]
        # Get the validation paths and labels
        with open(os.path.join(val_path, "val_annotations.txt")) as valf:
            for line in valf:
                file_name, nid, x_0, y_0, x_1, y_1 = line.split()
                file_name = os.path.join(val_path, "images", file_name)
                bbox = int(x_0), int(y_0), int(x_1), int(y_1)
                label_id = self.ids.index(nid)
                self.paths["val"].append((file_name, label_id, nid, bbox))

        # Get the training paths
        train_nids = os.listdir(train_path)
        for nid in train_nids:
            anno_path = os.path.join(train_path, nid, nid + "_boxes.txt")
            imgs_path = os.path.join(train_path, nid, "images")
            label_id = self.ids.index(nid)
            with open(anno_path, "r") as anno_file:
                for line in anno_file:
                    fname, x_0, y_0, x_1, y_1 = line.split()
                    fname = os.path.join(imgs_path, fname)
                    bbox = int(x_0), int(y_0), int(x_1), int(y_1)
                    self.paths["train"].append((fname, label_id, nid, bbox))


def pil_loader(path: str):
    """Convert to an RGB image."""
    with open(path, "rb") as image_file:
        with Image.open(image_file) as img:
            return img.convert("RGB")


def load_test_dataset_tiny_imagenet():
    """Load test dataset for Tiny-imagenet."""
    path_obj = TinyImageNetPaths()
    samples = path_obj.paths["val"]
    data = np.array([i[0] for i in samples])

    targets = np.array([i[1] for i in samples])
    labels = []
    images = []
    for current_image in data:
        img = pil_loader(current_image)
        img_np = np.asarray(img)
        images.append(img_np)

    for target in targets:
        labels.append(target)

    x_test = np.stack(images, axis=0)
    y_test = np.stack(labels, axis=0)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    return test_ds


@hydra.main(config_path="conf", config_name="base", version_base=None)
def download_and_preprocess(cfg: DictConfig) -> None:  # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements
    """Do everything needed to get the dataset.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
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
    folder_path = os.path.join(folder, str(total_clients), str(round(alpha, 2)))
    exist = os.path.exists(folder_path)
    if not exist:
        os.makedirs(folder_path)
    else:
        shutil.rmtree(folder_path, ignore_errors=True)

    if dataset in ["tiny-imagenet"]:
        path_obj = TinyImageNetPaths()

        samples = path_obj.paths["train"]
        data = np.array([i[0] for i in samples])
        targets = np.array([i[1] for i in samples])

        labels = []
        images = []
        for current_image in data:
            img = pil_loader(current_image)
            img_np = np.asarray(img)
            images.append(img_np)

        for target in targets:
            labels.append(target)

        x_train = np.stack(images, axis=0)
        y_train = np.stack(labels, axis=0)

    elif dataset in ["cifar100"]:
        (x_train, y_train), (_, _) = tf.keras.datasets.cifar100.load_data()

    # read the distribution of per-label examples for each client
    # from txt file
    data_mlb = read_fedmlb_distribution(
        dataset, total_clients=total_clients, alpha=alpha
    )

    for client in data_mlb:
        list_extracted_all_labels = data_mlb[client]
        numpy_dataset_y = y_train[list_extracted_all_labels]
        numpy_dataset_x = x_train[list_extracted_all_labels]

        tf_dataset = tf.data.Dataset.from_tensor_slices(
            (numpy_dataset_x, numpy_dataset_y)
        )
        tf_dataset = tf_dataset.shuffle(buffer_size=4096)

        tf.data.Dataset.save(
            tf_dataset, path=os.path.join(folder_path, "train", str(client))
        )

    path = os.path.join(os.path.join(folder_path, "train"))

    list_of_narrays = []
    for sampled_client in range(0, total_clients):
        loaded_ds = tf.data.Dataset.load(
            path=os.path.join(path, str(sampled_client)),
            element_spec=None,
            compression=None,
            reader_func=None,
        )

        print("[Client " + str(sampled_client) + "]")
        print("Cardinality: ", tf.data.experimental.cardinality(loaded_ds).numpy())

        def count_class(counts, batch, num_classes=num_classes):
            _, labels = batch
            for i in range(num_classes):
                accumulator = tf.cast(labels == i, tf.int32)
                counts[i] += tf.reduce_sum(accumulator)
            return counts

        initial_state = {i: 0 for i in range(num_classes)}
        counts = loaded_ds.reduce(initial_state=initial_state, reduce_func=count_class)

        # print([(k, v.numpy()) for k, v in counts.items()])
        new_dict = {k: v.numpy() for k, v in counts.items()}
        # print(new_dict)
        res = np.array(list(new_dict.values()))
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
