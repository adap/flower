import logging
import os
import random

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fedops.client import client_utils
from fedops.client.app import FLClientTask

from fedopsmnist import data_preparation, models


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    handlers_list = [logging.StreamHandler()]
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)8.8s] %(message)s",
        handlers=handlers_list,
    )

    logger = logging.getLogger(__name__)

    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    env_task_id = os.environ.get("FEDOPS_TASK_ID")
    if env_task_id:
        cfg.task_id = env_task_id

    print(OmegaConf.to_yaml(cfg))

    train_loader, val_loader, test_loader = data_preparation.load_partition(
        dataset=cfg.dataset.name,
        validation_split=cfg.dataset.validation_split,
        batch_size=cfg.batch_size,
    )

    logger.info("data loaded")

    model = instantiate(cfg.model)
    model_type = cfg.model_type
    model_name = type(model).__name__
    train_torch = models.train_torch()
    test_torch = models.test_torch()

    task_id = cfg.task_id
    local_list = client_utils.local_model_directory(task_id)

    if local_list:
        logger.info("Latest Local Model download")
        model = client_utils.download_local_model(
            model_type=model_type,
            task_id=task_id,
            listdir=local_list,
            model=model,
        )

    registration = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "model": model,
        "model_name": model_name,
        "train_torch": train_torch,
        "test_torch": test_torch,
    }

    fl_client = FLClientTask(cfg, registration)
    fl_client.start()


if __name__ == "__main__":
    main()
