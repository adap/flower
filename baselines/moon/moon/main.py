"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""
# these are the basic packages you'll need here
# feel free to remove some if aren't needed
import hydra
from omegaconf import DictConfig, OmegaConf
from moon import client, server
from moon.dataset_preparation import partition_data, get_dataloader


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 1. Print parsed config
    print(OmegaConf.to_yaml(cfg))

    # 2. Prepare your dataset
    # here you should call a function in datasets.py that returns whatever is needed to:
    # (1) ensure the server can access the dataset used to evaluate your model after
    # aggregation
    # (2) tell each client what dataset partitions they should use (e.g. a this could
    # be a location in the file system, a list of dataloader, a list of ids to extract
    # from a dataset, it's up to you)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        dataset=cfg.dataset.name, 
        datadir=cfg.dataset.dir, 
        parittion=cfg.dataset.partition, 
        num_clients=cfg.num_clients, 
        beta=cfg.dataset.beta,
    )
    
    train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(dataset=cfg.dataset.name,
                                                                            datadir=cfg.dataset.dir,
                                                                            train_bs=cfg.batch_size,
                                                                            test_bs=32)
    

    # 3. Define your clients
    # Define a function that returns another function that will be used during
    # simulation to instantiate each individual client
    # client_fn = client.<my_function_that_returns_a_function>()
    client_fn = client.gen_client_fn(
        num_clients=cfg.num_clients,
        num_epochs=cfg.num_epochs,
        trainloaders=trainloaders,
        valloaders=valloaders,
        num_rounds=cfg.num_rounds,
        learning_rate=cfg.learning_rate,
        model=cfg.model,
    )

    # 4. Define your strategy
    # pass all relevant argument (including the global dataset used after aggregation,
    # if needed by your method.)
    # strategy = instantiate(cfg.strategy, <additional arguments if desired>)

    # 5. Start Simulation
    # history = fl.simulation.start_simulation(<arguments for simulation>)

    # 6. Save your results
    # Here you can save the `history` returned by the simulation and include
    # also other buffers, statistics, info needed to be saved in order to later
    # on generate the plots you provide in the README.md. You can for instance
    # access elements that belong to the strategy for example:
    # data = strategy.get_my_custom_data() -- assuming you have such method defined.
    # Hydra will generate for you a directory each time you run the code. You
    # can retrieve the path to that directory with this:
    # save_path = HydraConfig.get().runtime.output_dir
