"""Runs the federated learning pipeline."""

import multiprocessing
import os
import pickle
import random
import sys
import time
import traceback

import flwr as fl
import hydra
import numpy as np
import torch
from flwr.server.history import History
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import dasha.dataset
from dasha.dataset_preparation import find_pre_downloaded_or_download_dataset

LOCAL_ADDRESS = "localhost:8080"


Exc = Exception


def _generate_seed(generator):
    return generator.integers(2**32 - 1)


def _get_dataset_input_shape(dataset):
    assert len(dataset) > 0
    sample_features, _ = dataset[0]
    return list(sample_features.shape)


def _generate_save_path(cfg: DictConfig) -> str:
    if cfg.save_path is not None:
        if HydraConfig.get().mode == hydra.types.RunMode.MULTIRUN:
            if HydraConfig.get().job.id == "0":
                assert not os.path.exists(cfg.save_path)
                os.mkdir(cfg.save_path)
            save_path = os.path.join(cfg.save_path, str(HydraConfig.get().job.id))
            os.mkdir(save_path)
        else:
            assert not os.path.exists(cfg.save_path)
            os.mkdir(cfg.save_path)
            save_path = cfg.save_path
    else:
        save_path = HydraConfig.get().runtime.output_dir
    return save_path


def _save_history(history: History, save_path: str, cfg: DictConfig) -> None:
    print(f"Saving to {save_path}")
    with open(os.path.join(save_path, "config.yaml"), "w") as file:
        OmegaConf.save(cfg, file)
    with open(os.path.join(save_path, "history"), "wb") as file:
        pickle.dump(history, file)


def _parallel_run(
    cfg: DictConfig, index_parallel: int, seed: int, queue: multiprocessing.Queue
) -> None:
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        local_address = (
            cfg.local_address if cfg.local_address is not None else LOCAL_ADDRESS
        )
        if index_parallel == 0:
            strategy_instance = instantiate(
                cfg.method.strategy, num_clients=cfg.num_clients
            )
            history = fl.server.start_server(
                server_address=local_address,
                config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
                strategy=strategy_instance,
            )
            queue.put(history)
        else:
            index_client = index_parallel - 1
            dataset = dasha.dataset.load_dataset(cfg)
            datasets = dasha.dataset.random_split(dataset, cfg.num_clients)
            local_dataset = datasets[index_client]
            function = instantiate(
                cfg.model, input_shape=_get_dataset_input_shape(dataset)
            )
            compressor = instantiate(cfg.compressor, seed=seed)
            client_instance = instantiate(
                cfg.method.client,
                function=function,
                dataset=local_dataset,
                compressor=compressor,
            )
            time.sleep(1.0)
            fl.client.start_numpy_client(
                server_address=local_address, client=client_instance
            )
    except Exc as exc:  # noqa: F841 # pylint: disable=W0703, W0612
        print(traceback.format_exc())


def run_parallel(cfg: DictConfig) -> History:
    """Run the pipeline."""
    sys.stderr = sys.stdout
    generator = np.random.default_rng(seed=42)
    processes = []
    queue: multiprocessing.Queue = multiprocessing.Queue()
    for index_parallel in range(cfg.num_clients + 1):
        seed = _generate_seed(generator)
        process = multiprocessing.Process(
            target=_parallel_run, args=(cfg, index_parallel, seed, queue)
        )
        process.start()
        processes.append(process)
    history = queue.get()
    for process in processes:
        process.join()
    return history


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Prepare a dataset, run the pipeline, and save a result."""
    save_path = _generate_save_path(cfg)
    find_pre_downloaded_or_download_dataset(cfg)
    history = run_parallel(cfg)
    _save_history(history, save_path, cfg)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
