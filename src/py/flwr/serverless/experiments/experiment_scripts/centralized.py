import wandb
from tensorflow.keras.utils import set_random_seed

from experiments.utils.centralized_runner import CentralizedRunner

# main function
if __name__ == "__main__":
    # starts a new run
    set_random_seed(117)

    num_nodes = 1
    dataset = "mnist"

    config = {
        "epochs": 128,
        "batch_size": 32,
        "steps_per_epoch": 8,
        "lr": 0.001,
        "shuffled:": False,
        "num_nodes": num_nodes,
        "dataset": dataset,
    }

    # federeated run w/ FedAvg
    wandb.init(
        project="experiments", entity="flwr_serverless", name="centralized", config=config
    )
    centralized_runner = CentralizedRunner(config, num_nodes, dataset)
    centralized_runner.run()
    wandb.finish()
