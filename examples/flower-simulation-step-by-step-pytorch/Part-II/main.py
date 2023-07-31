import pickle
from pathlib import Path

import hydra
from hydra.utils import instantiate, call
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import flwr as fl

from dataset import prepare_dataset
from client import generate_client_fn
from server import get_on_fit_config, get_evalulate_fn


# !!!! The code in this directory is the result of adpating the project first shown
# in <LINK> to make better use of Hydra's config system. It is recommended to first
# check the original code. There you'll find also additional comments walking you
# through in detail what each part of the code does.


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    ## 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir

    ## 2. Prepare your dataset
    trainloaders, validationloaders, testloader = prepare_dataset(
        cfg.num_clients, cfg.batch_size
    )

    ## 3. Define your clients
    # client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)

    # Let's pass the config node that defines the model. in this way changing models doesn't
    # require any changes to the code (just run the code with a different config)
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.model)

    ## 4. Define your strategy
    # strategy = fl.server.strategy.FedAvg(fraction_fit=0.00001,
    #                                      min_fit_clients=cfg.num_clients_per_round_fit,
    #                                      fraction_evaluate=0.00001,
    #                                      min_evaluate_clients=cfg.num_clients_per_round_eval,
    #                                      min_available_clients=cfg.num_clients,
    #                                      on_fit_config_fn=get_on_fit_config(cfg.config_fit),
    #                                      evaluate_fn=get_evalulate_fn(cfg.num_classes,
    #                                                                   testloader),
    #                                                                   )

    # So we have replaced the above with just a single line. Now if we want to use a different strategy,
    # even if it uses new arguments, you can leave the code below as is and pick a different config
    # The line below is instantiating the `strategy` node in the config. The result is an object of
    # the type specified in the _target_ field of it's config structure. For example, if you are using
    # the default (`conf/strategy/fedavg.yaml`), then _target_ is `flwr.server.strategy.FedAvg`.
    # The moment you run the experiment (i.e. when the config is parsed) not all field would be defined.
    # for instance, the testloader is not ready so `evaluate_fn` argument cannot be set. You can pass them
    # manually the moment you call `instantiate`. (if you are familiar with Python partials, this is similar)
    strategy = instantiate(
        cfg.strategy, evaluate_fn=get_evalulate_fn(cfg.model, testloader)
    )

    ## 5. Start Simulation
    # As you'll notice, we can start the simulation in exactly the same way as we did in the previous project.
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 0.0},
    )

    ## 6. Save your results
    # now we save the results of the simulation.
    results_path = Path(save_path) / "results.pkl"

    results = {"history": history, "anythingelse": "here"}

    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
