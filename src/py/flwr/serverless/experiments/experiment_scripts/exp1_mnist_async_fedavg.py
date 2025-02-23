import wandb

# import set_random_seed
from tensorflow.keras.utils import set_random_seed
from experiments.utils.federated_learning_runner import FederatedLearningRunner

# main function
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()
    # starts a new run
    set_random_seed(117)

    # shared config parameters
    num_nodes = 2
    federated_type = "concurrent"
    dataset = "mnist"
    strategy = "fedavg"
    epochs = 1000
    batch_size = 32
    steps_per_epoch = 64
    lr = 0.001

    base_config = {
        "net": "simple",
        "test_steps": None,
    }

    # async partitioned
    config1 = {
        "use_async": True,
        "data_split": "partitioned",
        "epochs": epochs,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "lr": lr,
        "num_nodes": num_nodes,
        "federated_type": federated_type,
        "dataset": dataset,
        "strategy": strategy,
    }

    # async skewed
    config2 = {
        "use_async": True,
        "data_split": "skewed",
        "epochs": epochs,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "lr": lr,
        "num_nodes": num_nodes,
        "federated_type": federated_type,
        "dataset": dataset,
        "strategy": strategy,
    }

    # async random
    config3 = {
        "use_async": True,
        "data_split": "random",
        "epochs": epochs,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "lr": lr,
        "num_nodes": num_nodes,
        "federated_type": federated_type,
        "dataset": dataset,
        "strategy": strategy,
    }
    # sync partitioned
    config4 = {
        "use_async": False,
        "data_split": "partitioned",
        "epochs": epochs,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "lr": lr,
        "num_nodes": num_nodes,
        "federated_type": federated_type,
        "dataset": dataset,
        "strategy": strategy,
    }

    # sync skewed
    config5 = {
        "use_async": False,
        "data_split": "skewed",
        "epochs": epochs,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "lr": lr,
        "num_nodes": num_nodes,
        "federated_type": federated_type,
        "dataset": dataset,
        "strategy": strategy,
    }

    # sync random
    config6 = {
        "use_async": False,
        "data_split": "random",
        "epochs": epochs,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "lr": lr,
        "num_nodes": num_nodes,
        "federated_type": federated_type,
        "dataset": dataset,
        "strategy": strategy,
    }

    configs = [config1, config2, config3, config4, config5, config6]

    for _config in configs:
        config = {**base_config, **_config}
        if config["use_async"]:
            use_async = "async"
        else:
            use_async = "sync"
        data_split = config["data_split"]
        # print(os.getenv("WANDB_PROJECT"))
        # wandb.init(
        #     project=os.getenv("WANDB_PROJECT"),
        #     entity="flwr_serverless",
        #     name=f"mnist_{use_async}_{data_split}_split",
        #     config=config,
        # )
        federated_learning_runner = FederatedLearningRunner(
            config=config,
            tracking=False,
        )
        federated_learning_runner.run()
        # wandb.finish()
