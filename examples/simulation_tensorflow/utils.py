from pathlib import Path
import numpy as np
import pickle
import shutil
from typing import Callable, Optional
from tf.keras.applications import MobileNetV2
from flwr.dataset.utils.common import create_lda_partitions
from flwr.common.typing import Tuple, Weights, XY

def get_model() -> MobileNetV2:
    model = MobileNetV2((32, 32, 3), classes=10, weights=None)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def partition_dataset(dataset: XY, fed_dir: Path, num_partitions: int, alpha: float=1000.0, prefix="train") -> None:
    partitions, _ = create_lda_partitions(
        dataset, num_partitions=num_partitions, concentration=alpha, accept_imbalanced=True
    )
    images, train_labels = dataset

    splits_dir = fed_dir
    if splits_dir.exists():
        shutil.rmtree(splits_dir)
    Path.mkdir(splits_dir, parents=True)

    for p in range(num_partitions):
        labels = partitions[p][1]
        image_idx = partitions[p][0]
        imgs = images[image_idx]

        # create dir
        Path.mkdir(splits_dir / str(p))

        with open(splits_dir / str(p) / f"{prefix}.pickle", "wb") as f:
            data = (imgs, labels)
            pickle.dump(data, f)

def get_eval_fn(
    testset: XY
) -> Callable[[Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: Weights) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""

        x_test, y_test = testset 
        model = get_model()
        model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        model.set_weights(weights)
        loss, accuracy = model.evaluate(x_test, y_test)

        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate

def load_partition(partition_dir: Path, prefix: str) -> XY:
    with open(partition_dir / f'{prefix}.pickle', 'rb') as f:
        (x_train, y_train) = pickle.load(f)
    return x_train, y_train

