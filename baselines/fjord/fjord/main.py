"""Main script for FjORD."""

import math
import os
import random
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Union

import flwr as fl
import hydra
import numpy as np
import torch
from flwr.client import Client, NumPyClient
from omegaconf import OmegaConf, open_dict

from .client import FJORD_CONFIG_TYPE, FjORDClient, get_agg_config
from .dataset import load_data
from .models import get_net
from .server import get_eval_fn
from .strategy import FjORDFedAVG
from .utils.logger import Logger
from .utils.utils import get_parameters


def get_fit_config_fn(
    total_rounds: int, lr: float
) -> Callable[[int], Dict[str, fl.common.Scalar]]:
    """Get fit config function.

    :param total_rounds: Total number of rounds
    :param lr: Learning rate
    :return: Fit config function
    """

    def fit_config(rnd: int) -> Dict[str, fl.common.Scalar]:
        config: Dict[str, fl.common.Scalar] = {
            "current_round": rnd,
            "total_rounds": total_rounds,
            "lr": lr,
        }
        return config

    return fit_config


def get_client_fn(  # pylint: disable=too-many-arguments
    args: Any,
    model_path: str,
    cid_to_max_p: Dict[int, float],
    config: FJORD_CONFIG_TYPE,
    train_config: SimpleNamespace,
    device: torch.device,
) -> Callable[[str], Union[Client, NumPyClient]]:
    """Get client function that creates Flower client.

    :param args: CLI/Config Arguments
    :param model_path: Path to save the model
    :param cid_to_max_p: Dictionary mapping client id to max p-value
    :param config: Aggregation config
    :param train_config: Training config
    :param device: Device to be used
    :return: Client function that returns Flower client
    """

    def client_fn(cid) -> FjORDClient:
        max_p = cid_to_max_p[int(cid)]
        log_config = {
            "loglevel": args.loglevel,
            "logfile": args.logfile,
        }
        return FjORDClient(
            cid=cid,
            model_name=args.model,
            data_path=args.data_path,
            model_path=model_path,
            know_distill=args.knowledge_distillation,
            max_p=max_p,
            p_s=args.p_s,
            fjord_config=config,
            train_config=train_config,
            log_config=log_config,
            seed=args.manual_seed,
            device=device,
        )

    return client_fn


class FjORDBalancedClientManager(fl.server.SimpleClientManager):
    """Balanced client manager for FjORD.

    This class samples equal number of clients per p-value and the rest in RR.
    """

    def __init__(self, cid_to_max_p: Dict[int, float]) -> None:
        """Ctor.

        Args:
        :param cid_to_max_p: Dictionary mapping client id to max p-value
        """
        super().__init__()
        self.cid_to_max_p = cid_to_max_p
        self.p_s = sorted(set(self.cid_to_max_p.values()))

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[fl.server.criterion.Criterion] = None,
    ) -> List[fl.server.client_proxy.ClientProxy]:
        """Sample clients in a balanced way (equal per tier, remainder in Round-Robin).

        Args:
        :param num_clients: Number of clients to sample
        :param min_num_clients: Minimum number of clients to sample
        :param criterion: Client selection criterion
        :return: List of sampled clients
        """
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]
        if num_clients > len(available_cids):
            Logger.get().info(
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        # construct p to available cids
        max_p_to_cids: Dict[float, List[int]] = {p: [] for p in self.p_s}
        random.shuffle(available_cids)
        for cid_s in available_cids:
            client_id = int(cid_s)
            client_p = self.cid_to_max_p[client_id]
            max_p_to_cids[client_p].append(client_id)

        cl_per_tier = math.floor(num_clients / len(self.p_s))
        remainder = num_clients - cl_per_tier * len(self.p_s)

        selected_cids = set()
        for p in self.p_s:
            for cid in random.sample(max_p_to_cids[p], cl_per_tier):
                selected_cids.add(cid)

        for p in self.p_s:
            if remainder == 0:
                break
            cid = random.choice(max_p_to_cids[p])
            while cid not in selected_cids:
                cid = random.choice(max_p_to_cids[p])
            selected_cids.add(cid)
            remainder -= 1

        Logger.get().debug(f"Sampled {selected_cids}")
        return [self.clients[str(cid)] for cid in selected_cids]


def main(args: Any) -> None:
    """Enter main functionality.

    Args:
    :param args: CLI/Config Arguments
    """
    torch.manual_seed(args.manual_seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    path = args.data_path
    device = torch.device("cuda") if args.cuda else torch.device("cpu")
    model_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    Logger.get().info(
        f"Training on {device} using PyTorch "
        f"{torch.__version__} and Flower {fl.__version__}"
    )

    trainloader, testloader = load_data(
        path, cid=0, seed=args.manual_seed, train_bs=args.batch_size
    )
    NUM_CLIENTS = args.num_clients
    if args.client_tier_allocation == "uniform":
        cid_to_max_p = {cid: (cid // 20) * 0.2 + 0.2 for cid in range(100)}
    else:
        raise ValueError(
            f"Client to tier allocation strategy "
            f"{args.client_tier_allocation} not currently"
            "supported"
        )

    model = get_net(args.model, args.p_s, device=device)
    config = get_agg_config(model, trainloader, args.p_s)
    train_config = SimpleNamespace(
        **{
            "batch_size": args.batch_size,
            "lr": args.lr,
            "optimiser": args.optimiser,
            "momentum": args.momentum,
            "nesterov": args.nesterov,
            "lr_scheduler": args.lr_scheduler,
            "weight_decay": args.weight_decay,
            "local_epochs": args.local_epochs,
        }
    )

    if args.strategy == "fjord_fedavg":
        strategy = FjORDFedAVG(
            fraction_fit=args.sampled_clients / args.num_clients,
            fraction_evaluate=0.0,
            min_fit_clients=args.min_fit_clients,
            min_evaluate_clients=1,
            min_available_clients=NUM_CLIENTS,
            evaluate_fn=get_eval_fn(args, model_path, testloader, device),
            on_fit_config_fn=get_fit_config_fn(args.num_rounds, args.lr),
            initial_parameters=fl.common.ndarrays_to_parameters(
                get_parameters(get_net(args.model, args.p_s, device=device))
            ),
        )
    else:
        raise ValueError(f"Strategy {args.strategy} is not currently supported")

    client_resources = args.client_resources
    if device.type != "cuda":
        client_resources = {
            "num_cpus": args.client_resources["num_cpus"],
            "num_gpus": 0,
        }

    if args.client_selection == "balanced":
        cl_manager = FjORDBalancedClientManager(cid_to_max_p)
    elif args.client_selection == "random":
        cl_manager = None
    else:
        raise ValueError(
            f"Client selection {args.client_selection} is not currently supported"
        )

    Logger.get().info("Starting simulated run.")
    # Start simulation
    fl.simulation.start_simulation(
        client_fn=get_client_fn(
            args, model_path, cid_to_max_p, config, train_config, device
        ),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_resources=client_resources,
        client_manager=cl_manager,
        ray_init_args={"include_dashboard": False},
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_app(cfg):
    """Run the application.

    Args:
    :param cfg: Hydra configuration
    """
    OmegaConf.resolve(cfg)
    logfile = os.path.join(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"], cfg.logfile
    )
    with open_dict(cfg):
        cfg.logfile = logfile
    Logger.setup_logging(loglevel=cfg.loglevel, logfile=logfile)
    Logger.get().info(f"Hydra configuration: {OmegaConf.to_yaml(cfg)}")
    main(cfg)


if __name__ == "__main__":
    run_app()
