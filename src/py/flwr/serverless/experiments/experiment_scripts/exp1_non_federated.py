import wandb
from tensorflow.keras.utils import set_random_seed

from experiments.utils.non_federated_runner import NonFederatedRunner

# main function
if __name__ == "__main__":
    # starts a new run
    set_random_seed(117)

    num_nodes = 2
    use_async = True
    shuffled = True  # if true, the order of the data is shuffled before partitioning
    federated_type = "concurrent"  # options: concurrent, sequential, pseudo-concurrent
    dataset = "mnist"
    strategy = "fedavg"

    # TODO: grab configs (overrides) from wandb and put them in
    #    a list called configs.
    #   Then, iterate over configs and run the experiment
    config = {
        "epochs": 128,
        "batch_size": 32,
        "steps_per_epoch": 16,
        "lr": 0.0004,
        "num_nodes": num_nodes,
        "use_async": use_async,
        "federated_type": federated_type,
        "dataset": dataset,
        "strategy": strategy,
        "shuffled": shuffled,
    }

    num_nodes = 2
    dataset = "mnist"

    wandb.init(
        project="test-project", entity="flwr_serverless", name="non_federated", config=config
    )
    nonfederated_runner = NonFederatedRunner(config, num_nodes, dataset)
    nonfederated_runner.run()
    wandb.finish()
