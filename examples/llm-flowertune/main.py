import warnings
import pickle

import flwr as fl
from flwr_datasets import FederatedDataset

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from dataset import get_tokenizer_and_data_collator_and_propt_formatting
from utils import get_on_fit_config, fit_weighted_average, get_evaluate_fn
from client import gen_client_fn


warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run federated LLM fine-tuning.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # Print config structured as YAML
    print(OmegaConf.to_yaml(cfg))

    # Partition dataset and get dataloaders
    fds = FederatedDataset(
        dataset=cfg.dataset.name, partitioners={"train": cfg.num_clients}
    )
    (
        tokenizer,
        data_collator,
        formatting_prompts_func,
    ) = get_tokenizer_and_data_collator_and_propt_formatting(
        cfg.model.name,
    )

    # Hydra automatically creates an output directory
    # Let's retrieve it and save some results there
    save_path = HydraConfig.get().runtime.output_dir

    # Prepare function that will be used to spawn each client
    client_fn = gen_client_fn(
        fds,
        tokenizer,
        formatting_prompts_func,
        data_collator,
        cfg.model,
        cfg.train,
        save_path,
    )

    # Instantiate strategy according to config. Here we pass other arguments
    # that are only defined at run time.
    strategy = instantiate(
        cfg.strategy,
        on_fit_config_fn=get_on_fit_config(),
        fit_metrics_aggregation_fn=fit_weighted_average,
        evaluate_fn=get_evaluate_fn(
            cfg.model, cfg.train.save_every_round, cfg.num_rounds, save_path
        ),
    )

    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        client_resources={
            "num_cpus": cfg.client_resources.num_cpus,
            "num_gpus": cfg.client_resources.num_gpus,
        },
        strategy=strategy,
    )

    # Experiment completed. Now we save the results and
    # generate plots using the `history`
    print("................")
    print(history)

    # Save results as a Python pickle using a file_path
    # the directory created by Hydra for each run
    with open(f"{save_path}/results.pkl", "wb") as f:
        pickle.dump(history, f)


if __name__ == "__main__":
    main()
