"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-process
your dataset (or all of the above). If the desired way of running your baseline is to
first download the dataset and partition it and then run the experiments, please
uncomment the lines below and tell us in the README.md (see the "Running the Experiment"
block) that this file should be executed first.
"""

import bz2
import os
import shutil
import urllib.request
from typing import Optional

import numpy as np
from sklearn.datasets import load_svmlight_file


def download_data(dataset_name: Optional[str] = "cod-rna"):
    """Download (if necessary) the dataset and returns the dataset path.

    Parameters
    ----------
    dataset_name : String
        A string stating the name of the dataset that need to be dowenloaded.

    Returns
    -------
    List[Dataset Pathes]
        The pathes for the data that will be used in train and test,
        with train of full dataset in index 0
    """
    all_datasets_path = "./dataset"
    if dataset_name:
        dataset_path = os.path.join(all_datasets_path, dataset_name)
    match dataset_name:
        case "a9a":
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)
                urllib.request.urlretrieve(
                    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets"
                    "/binary/a9a",
                    f"{os.path.join(dataset_path, 'a9a')}",
                )
                urllib.request.urlretrieve(
                    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets"
                    "/binary/a9a.t",
                    f"{os.path.join(dataset_path, 'a9a.t')}",
                )
            # training then test ✅
            return_list = [
                os.path.join(dataset_path, "a9a"),
                os.path.join(dataset_path, "a9a.t"),
            ]
        case "cod-rna":
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)
                urllib.request.urlretrieve(
                    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets"
                    "/binary/cod-rna.t",
                    f"{os.path.join(dataset_path, 'cod-rna.t')}",
                )
                urllib.request.urlretrieve(
                    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets"
                    "/binary/cod-rna.r",
                    f"{os.path.join(dataset_path, 'cod-rna.r')}",
                )
            # training then test ✅
            return_list = [
                os.path.join(dataset_path, "cod-rna.t"),
                os.path.join(dataset_path, "cod-rna.r"),
            ]

        case "ijcnn1":
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)

                urllib.request.urlretrieve(
                    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets"
                    "/binary/ijcnn1.bz2",
                    f"{os.path.join(dataset_path, 'ijcnn1.tr.bz2')}",
                )
                urllib.request.urlretrieve(
                    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets"
                    "/binary/ijcnn1.t.bz2",
                    f"{os.path.join(dataset_path, 'ijcnn1.t.bz2')}",
                )
                urllib.request.urlretrieve(
                    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets"
                    "/binary/ijcnn1.tr.bz2",
                    f"{os.path.join(dataset_path, 'ijcnn1.tr.bz2')}",
                )
                urllib.request.urlretrieve(
                    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets"
                    "/binary/ijcnn1.val.bz2",
                    f"{os.path.join(dataset_path, 'ijcnn1.val.bz2')}",
                )

                for filepath in os.listdir(dataset_path):
                    abs_filepath = os.path.join(dataset_path, filepath)
                    with bz2.BZ2File(abs_filepath) as freader, open(
                        abs_filepath[:-4], "wb"
                    ) as fwriter:
                        shutil.copyfileobj(freader, fwriter)
            # training then test ✅
            return_list = [
                os.path.join(dataset_path, "ijcnn1.t"),
                os.path.join(dataset_path, "ijcnn1.tr"),
            ]

        case "space_ga":
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)
                urllib.request.urlretrieve(
                    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets"
                    "/regression/space_ga_scale",
                    f"{os.path.join(dataset_path, 'space_ga_scale')}",
                )
            return_list = [os.path.join(dataset_path, "space_ga_scale")]
        case "abalone":
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)
                urllib.request.urlretrieve(
                    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets"
                    "/regression/abalone_scale",
                    f"{os.path.join(dataset_path, 'abalone_scale')}",
                )
            return_list = [os.path.join(dataset_path, "abalone_scale")]
        case "cpusmall":
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)
                urllib.request.urlretrieve(
                    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets"
                    "/regression/cpusmall_scale",
                    f"{os.path.join(dataset_path, 'cpusmall_scale')}",
                )
            return_list = [os.path.join(dataset_path, "cpusmall_scale")]
        case "YearPredictionMSD":
            if not os.path.exists(dataset_path):
                print(
                    "long download coming -~615MB-, it'll be better if you downloaded",
                    "those 2 files manually with a faster download manager program or",
                    "something and just place them in the right folder then get",
                    "the for loop out of the if condition to alter their format",
                )
                os.makedirs(dataset_path)
                urllib.request.urlretrieve(
                    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets"
                    "/regression/YearPredictionMSD.bz2",
                    f"{os.path.join(dataset_path, 'YearPredictionMSD.bz2')}",
                )
                urllib.request.urlretrieve(
                    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets"
                    "/regression/YearPredictionMSD.t.bz2",
                    f"{os.path.join(dataset_path, 'YearPredictionMSD.t.bz2')}",
                )
                for filepath in os.listdir(dataset_path):
                    print("it will take sometime")
                    abs_filepath = os.path.join(dataset_path, filepath)
                    with bz2.BZ2File(abs_filepath) as freader, open(
                        abs_filepath[:-4], "wb"
                    ) as fwriter:
                        shutil.copyfileobj(freader, fwriter)
            return_list = [
                os.path.join(dataset_path, "YearPredictionMSD"),
                os.path.join(dataset_path, "YearPredictionMSD.t"),
            ]
        case _:
            raise Exception("write your own dataset downloader")
    return return_list


def datafiles_fusion(data_paths):
    """Merge (if necessary) the data files and returns the features and labels.

    Parmetres:
        data_paths: List[Dataset Pathes]
            - The pathes for the data that will be used in train and test,
            with train of full dataset in index 0
    Returns:
        X: Numpy array
            - The full features of the dataset.
        y: Numpy array
            - The full labels of the dataset.
    """
    data = load_svmlight_file(data_paths[0], zero_based=False)
    X = data[0].toarray()
    Y = data[1]
    for i in range(1, len(data_paths)):
        data = load_svmlight_file(
            data_paths[i], zero_based=False, n_features=X.shape[1]
        )
        X = np.concatenate((X, data[0].toarray()), axis=0)
        Y = np.concatenate((Y, data[1]), axis=0)
    return X, Y


def train_test_split(X, y, train_ratio=0.75):
    """Split the dataset into training and testing.

    Parameters
    ----------
        X: Numpy array
            The full features of the dataset.
        y: Numpy array
            The full labels of the dataset.
        train_ratio: float
            the ratio that training should take from the full dataset

    Returns
    -------
        X_train: Numpy array
            The training dataset features.
        y_train: Numpy array
            The labels of the training dataset.
        X_test: Numpy array
            The testing dataset features.
        y_test: Numpy array
            The labels of the testing dataset.
    """
    np.random.seed(2023)
    y = np.expand_dims(y, axis=1)
    full = np.concatenate((X, y), axis=1)
    np.random.shuffle(full)
    y = full[:, -1]  # for last column
    X = full[:, :-1]  # for all but last column
    num_training_samples = int(X.shape[0] * train_ratio)

    x_train = X[0:num_training_samples]
    y_train = y[0:num_training_samples]

    x_test = X[num_training_samples:]
    y_test = y[num_training_samples:]

    x_train.flags.writeable = True
    y_train.flags.writeable = True
    x_test.flags.writeable = True
    y_test.flags.writeable = True

    return x_train, y_train, x_test, y_test


def modify_labels(y_train, y_test):
    """Switch the -1 in the classification dataset with 0.

    Parameters
    ----------
        y_train: Numpy array
            The labels of the training dataset.
        y_test: Numpy array
            The labels of the testing dataset.

    Returns
    -------
        y_train: Numpy array
            The labels of the training dataset.
        y_test: Numpy array
            The labels of the testing dataset.
    """
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0
    return y_train, y_test
