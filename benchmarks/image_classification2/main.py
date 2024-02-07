import warnings
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import yaml
import flwr as fl

from datasets.utils.logging import disable_progress_bar
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

from baseline.utils import fit_weighted_average, eval_weighted_average, get_evaluate_fn, get_fit_config
from baseline.client import get_client_fn
from static.static_utils import resplit


warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="baseline/conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Print config structured as YAML
    print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir

    # Load static configs
    with open("static/static_config.yaml", "r") as f:
        static_config = yaml.load(f, Loader=yaml.FullLoader)

    # Download EuroSAT dataset and partition it
    partitioner = IidPartitioner(num_partitions=static_config["num_clients"])
    fds = FederatedDataset(dataset="jonathan-roberts1/EuroSAT", partitioners={"train": partitioner}, resplitter=resplit)
    centralized_testset = fds.load_full("test")

    # Configure the strategy
    strategy = instantiate(
        cfg.strategy,
        fraction_fit=static_config["fraction_fit"],  # Sample all of available clients for training
        min_available_clients=int(static_config["num_clients"]),  # Wait until all clients are available
        fraction_evaluate=1.0,  # Sample all of available clients for evaluation
        min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
        min_fit_clients=static_config["min_fit_clients"],  # Never sample less than 5 clients for training
        on_fit_config_fn=get_fit_config(static_config["local_epoch"], cfg.batch_size, cfg.lr, cfg.momentum),
        fit_metrics_aggregation_fn=fit_weighted_average,  # Weighted average fit metrics
        evaluate_metrics_aggregation_fn=eval_weighted_average,  # Weighted average eval metrics
        evaluate_fn=get_evaluate_fn(centralized_testset, cfg.save_every_round, save_path),  # Evaluation function to save global model
    )

    # Resources to be assigned to each virtual client
    client_resources = {
        "num_cpus": cfg.client_resources.num_cpus,
        "num_gpus": cfg.client_resources.num_gpus,
    }

    # Start simulation
    hist = fl.simulation.start_simulation(
        client_fn=get_client_fn(fds, static_config["test_size"], static_config["seed"]),
        num_clients=static_config["num_clients"],
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=static_config["num_rounds"]),
        strategy=strategy,
        actor_kwargs={
            "on_actor_init_fn": disable_progress_bar  # disable tqdm on each actor/process spawning virtual clients
        },
    )


if __name__ == "__main__":
    main()
