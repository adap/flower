"""Download the raw datasets for MNIST, USPS, SVHN Create the MNISTM from MNIST
And download the Synth Data (Syntehtic digits Windows TM font varying the
orientation, blur and stroke colors).

This dataset is already processed.
"""
import gzip
import pickle
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import wget  # type: ignore
from torchvision import datasets  # type: ignore

from .mnistm import create_mnistm  # type: ignore

# pylint: disable=invalid-name


def decompress(infile, tofile):
    """Take data file and unzip it."""

    with open(infile, "rb") as inf, open(tofile, "w", encoding="utf8") as tof:
        decom_str = gzip.decompress(inf.read()).decode("utf-8")
        tof.write(decom_str)


def download_all(data: Dict, out_dir: Path):
    """Downloading datasets."""

    for k, v in data.items():
        print(f"Downloading: {k}\n")
        wget.download(v, out=str(out_dir / k))


def get_synthDigits(out_dir: Path):
    """get synth dataset."""

    if out_dir.exists():
        print(f"Directory ({out_dir}) exists, skipping downloading SynthDigits.")
        return

    # pylint: disable=line-too-long
    out_dir.mkdir()
    data = {}
    data[
        "synth_train_32x32.mat"
    ] = "https://github.com/domainadaptation/datasets/blob/master/synth/synth_train_32x32.mat?raw=true"
    data[
        "synth_test_32x32.mat"
    ] = "https://github.com/domainadaptation/datasets/blob/master/synth/synth_test_32x32.mat?raw=true"
    download_all(data, out_dir)
    # pylint: disable=line-too-long

    # How to proceed? It seems these `.mat` have no data. URLs found here:
    # https://domainadaptation.org/api/salad.datasets.digits.html#module-salad.datasets.digits.synth


def get_MNISTM(out_dir: Path):
    """Creates MNISTM dataset as done by https://github.com/pumpikano/tf-
    dann#build-mnist-m-dataset."""
    # steps = 'https://github.com/pumpikano/tf-dann#build-mnist-m-dataset'
    if out_dir.exists():
        print(f"> Directory ({out_dir}) exists, skipping downloading MNISTM.")
        return

    out_dir.mkdir()
    data = {}
    data[
        "BSR_bsds500.tgz"
    ] = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
    download_all(data, out_dir)

    train = torch.load("./data/MNIST/training.pt")
    test = torch.load("./data/MNIST/test.pt")
    print("Building train set...")
    train_labels = train[1]
    train = create_mnistm.create_mnistm(train[0])
    print("Building test set...")
    test_labels = test[1]
    test = create_mnistm.create_mnistm(test[0])
    val_labels = np.zeros(0)
    val = np.zeros([0, 28, 28, 3], np.uint8)

    # Save dataset as pickle
    with open(out_dir / "mnistm_data.pkl", "wb") as f:
        pickle.dump(
            {
                "train": train,
                "train_label": train_labels,
                "test": test,
                "test_label": test_labels,
                "valid": val,
                "valid_label": val_labels,
            },
            f,
            pickle.HIGHEST_PROTOCOL,
        )


def get_USPS(out_dir: Path):
    """get USPS data (handwritten digits from envelopes by the U.S.

    Postal Service)
    """

    if out_dir.exists():
        print(f"> Directory ({out_dir}) exists, skipping downloading USPS.")
        return

    out_dir.mkdir()
    data = {}
    data[
        "usps.bz2"
    ] = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2"
    data[
        "usps.t.bz2"
    ] = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2"

    download_all(data, out_dir)


def get_SVHN(out_dir: Path):
    """Get SVHN dataset (Street view house numbers)"""
    if out_dir.exists():
        print(f"> Directory ({out_dir}) exists, skipping downloading SVHN.")
        return

    out_dir.mkdir()

    data = {}
    data["train_32x32.mat"] = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
    data["test_32x32.mat"] = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"

    download_all(data, out_dir)


def get_MNIST(out_dir: Path):
    """Downloads MNIST using torchvision routines.

    Then, move processed files to directory expected by
    `utils/data_processing.py`. Delete the rest.
    """

    if (out_dir / "MNIST").exists():
        print(f"> Directory ({out_dir}) exists, skipping downloading MNIST.")
        print(type(out_dir))
        return

    datasets.MNIST(out_dir, train=True, download=True)

    datasets.MNIST(out_dir, train=False)

    train_file = "training.pt"
    test_file = "test.pt"
    shutil.move(out_dir / "MNIST" / "processed" / train_file, out_dir / "MNIST" / train_file)  # type: ignore[arg-type]
    shutil.move(out_dir / "MNIST" / "processed" / test_file, out_dir / "MNIST" / test_file)  # type: ignore[arg-type]
    shutil.rmtree(out_dir / "MNIST" / "raw")
    shutil.rmtree(out_dir / "MNIST" / "processed")


def main():
    """Get all the datasets."""

    data_dir = Path("./data")

    data_dir.mkdir(exist_ok=True)

    get_MNIST(data_dir)  # type: ignore[arg-type]

    get_SVHN(data_dir / "SVHN")

    get_USPS(data_dir / "USPS")

    get_MNISTM(data_dir / "MNIST_M")

    get_synthDigits(data_dir / "SynthDigits")


if __name__ == "__main__":
    main()
