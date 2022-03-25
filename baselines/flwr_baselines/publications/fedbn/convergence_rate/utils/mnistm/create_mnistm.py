"""! This script has been borrowed and adapted. Original script:
https://github.com/pumpikano/tf-dann/blob/master/create_mnistm.py.

It creatse the MNIST-M dataset based on MNIST
"""
import tarfile
from typing import Any

import numpy as np
import skimage  # type: ignore
import skimage.io  # type: ignore
import skimage.transform  # type: ignore


# pylint: disable=invalid-name, disable=no-member, bare-except
def compose_image(digit: Any, background: Any) -> Any:
    """Difference-blend a digit and a random patch from a background image."""
    w, h, _ = background.shape
    dw, dh, _ = digit.shape
    x = np.random.randint(0, w - dw)
    y = np.random.randint(0, h - dh)

    bg = background[x : x + dw, y : y + dh]
    return np.abs(bg - digit).astype(np.uint8)


def mnist_to_img(x: Any) -> Any:
    """Binarize MNIST digit and convert to RGB."""
    x = (x > 0).float()
    d = x.reshape([28, 28, 1]) * 255
    return np.concatenate([d, d, d], 2)


def create_mnistm(X: Any) -> Any:
    """Give an array of MNIST digits, blend random background patches to build
    the MNIST-M dataset as described in
    http://jmlr.org/papers/volume17/15-239/15-239.pdf."""

    bst_path = "./data/MNIST_M/BSR_bsds500.tgz"

    rand = np.random.RandomState(42)
    train_files = []

    with tarfile.open(bst_path, "r") as bsr_file:
        for name in bsr_file.getnames():
            if name.startswith("BSR/BSDS500/data/images/train/"):
                train_files.append(name)

        print("Loading BSR training images")
        background_data = []
        for name in train_files:
            try:
                fp = bsr_file.extractfile(name)
                bg_img = skimage.io.imread(fp)
                background_data.append(bg_img)
            except:
                continue

    X_ = np.zeros([X.shape[0], 28, 28, 3], np.uint8)
    for i in range(X.shape[0]):

        if i % 1000 == 0:
            print("Processing example", i)

        bg_img = rand.choice(background_data)
        d = mnist_to_img(X[i])
        d = compose_image(d, bg_img)
        X_[i] = d

    return X_
