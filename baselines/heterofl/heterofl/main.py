"""Runs federated learning for given configuration in base.yaml."""

import pickle
from pathlib import Path

import flwr as fl
import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from heterofl import client, models, server
from heterofl.client_manager_heterofl import ClientManagerHeteroFL
from heterofl.dataset import load_datasets
from heterofl.model_properties import get_model_properties
from heterofl.utils import ModelRateManager, get_global_model_rate, preprocess_input


# pylint: disable=too-many-locals,protected-access
@hydra.main(config_path="conf", config_name="base.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # print config structured as YAML
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)

    data_loaders = {}

    (
        data_loaders["entire_trainloader"],
        data_loaders["trainloaders"],
        data_loaders["label_split"],
        data_loaders["valloaders"],
        data_loaders["testloader"],
    ) = load_datasets(
        "heterofl" if "heterofl" in cfg.strategy._target_ else "fedavg",
        config=cfg.dataset,
        num_clients=cfg.num_clients,
        seed=cfg.seed,
    )

    model_config = preprocess_input(cfg.model, cfg.dataset)

    model_split_rate = None
    model_mode = None
    client_to_model_rate_mapping = None
    model_rate_manager = None
    history = None

    if "HeteroFL" in cfg.strategy._target_:
        # send this array(client_model_rate_mapping) as
        # an argument to client_manager and client
        model_split_rate = {"a": 1, "b": 0.5, "c": 0.25, "d": 0.125, "e": 0.0625}
        # model_split_mode = cfg.control.model_split_mode
        model_mode = cfg.control.model_mode

        client_to_model_rate_mapping = [float(0) for _ in range(cfg.num_clients)]
        model_rate_manager = ModelRateManager(
            cfg.control.model_split_mode, model_split_rate, model_mode
        )

        model_config["global_model_rate"] = model_split_rate[
            get_global_model_rate(model_mode)
        ]

    test_model = models.create_model(
        model_config,
        model_rate=(
            model_split_rate[get_global_model_rate(model_mode)]
            if model_split_rate is not None
            else None
        ),
        track=True,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )

    get_model_properties(
        model_config,
        model_split_rate,
        model_mode + "" if model_mode is not None else None,
        data_loaders["entire_trainloader"],
        cfg.dataset.batch_size.train,
    )

    # prepare function that will be used to spawn each client
    client_train_settings = {
        "epochs": cfg.num_epochs,
        "optimizer": cfg.optim_scheduler.optimizer,
        "lr": cfg.optim_scheduler.lr,
        "momentum": cfg.optim_scheduler.momentum,
        "weight_decay": cfg.optim_scheduler.weight_decay,
        "scheduler": cfg.optim_scheduler.scheduler,
        "milestones": cfg.optim_scheduler.milestones,
    }

    if "clip" in cfg:
        client_train_settings["clip"] = cfg.clip

    optim_scheduler_settings = {
        "optimizer": cfg.optim_scheduler.optimizer,
        "lr": cfg.optim_scheduler.lr,
        "momentum": cfg.optim_scheduler.momentum,
        "weight_decay": cfg.optim_scheduler.weight_decay,
        "scheduler": cfg.optim_scheduler.scheduler,
        "milestones": cfg.optim_scheduler.milestones,
    }

    client_fn = client.gen_client_fn(
        model_config=model_config,
        client_to_model_rate_mapping=client_to_model_rate_mapping,
        client_train_settings=client_train_settings,
        data_loaders=data_loaders,
    )

    evaluate_fn = server.gen_evaluate_fn(
        data_loaders,
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        test_model,
        models.create_model(
            model_config,
            model_rate=(
                model_split_rate[get_global_model_rate(model_mode)]
                if model_split_rate is not None
                else None
            ),
            track=False,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        )
        .state_dict()
        .keys(),
        enable_train_on_train_data=(
            cfg.enable_train_on_train_data_while_testing
            if "enable_train_on_train_data_while_testing" in cfg
            else True
        ),
    )
    client_resources = {
        "num_cpus": cfg.client_resources.num_cpus,
        "num_gpus": cfg.client_resources.num_gpus if torch.cuda.is_available() else 0,
    }

    if "HeteroFL" in cfg.strategy._target_:
        strategy_heterofl = instantiate(
            cfg.strategy,
            model_name=cfg.model.model_name,
            net=models.create_model(
                model_config,
                model_rate=(
                    model_split_rate[get_global_model_rate(model_mode)]
                    if model_split_rate is not None
                    else None
                ),
                device="cpu",
            ),
            optim_scheduler_settings=optim_scheduler_settings,
            global_model_rate=(
                model_split_rate[get_global_model_rate(model_mode)]
                if model_split_rate is not None
                else 1.0
            ),
            evaluate_fn=evaluate_fn,
            min_available_clients=cfg.num_clients,
        )

        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=cfg.num_clients,
            config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
            client_resources=client_resources,
            client_manager=ClientManagerHeteroFL(
                model_rate_manager,
                client_to_model_rate_mapping,
                client_label_split=data_loaders["label_split"],
            ),
            strategy=strategy_heterofl,
        )
    else:
        strategy_fedavg = instantiate(
            cfg.strategy,
            # on_fit_config_fn=lambda server_round: {
            #     "lr": cfg.optim_scheduler.lr
            #     * pow(cfg.optim_scheduler.lr_decay_rate, server_round)
            # },
            evaluate_fn=evaluate_fn,
            min_available_clients=cfg.num_clients,
        )

        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=cfg.num_clients,
            config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
            client_resources=client_resources,
            strategy=strategy_fedavg,
        )

    # save the results
    save_path = HydraConfig.get().runtime.output_dir

    # save the results as a python pickle
    with open(str(Path(save_path) / "results.pkl"), "wb") as file_handle:
        pickle.dump({"history": history}, file_handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save the model
    torch.save(test_model.state_dict(), str(Path(save_path) / "model.pth"))


if __name__ == "__main__":
    main()
