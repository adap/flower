import json
import logging
from datetime import datetime

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

handlers_list = [logging.StreamHandler()]

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)8.8s] %(message)s",
    handlers=handlers_list,
)

logger = logging.getLogger(__name__)


def load_partition(dataset, validation_split, batch_size):
    now = datetime.now()
    fl_task = {"dataset": dataset, "start_execution_time": now.strftime("%Y-%m-%d %H:%M:%S")}
    logging.info("FL_Task - %s", json.dumps(fl_task))

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    full_dataset = datasets.MNIST(
        root="./dataset/mnist",
        train=True,
        download=True,
        transform=transform,
    )

    test_split = 0.2
    train_size = int((1 - validation_split - test_split) * len(full_dataset))
    validation_size = int(validation_split * len(full_dataset))
    test_size = len(full_dataset) - train_size - validation_size
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, validation_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def gl_model_torch_validation(batch_size):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    val_dataset = datasets.MNIST(
        root="./dataset/mnist",
        train=False,
        download=True,
        transform=transform,
    )

    gl_val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return gl_val_loader
