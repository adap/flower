import os
import warnings
from hydra import compose, initialize

import flwr as fl
from flwr_datasets import FederatedDataset

from dataset import get_tokenizer_and_data_collator_and_propt_formatting
from client import gen_client_fn
from utils import get_on_fit_config, fit_weighted_average


warnings.filterwarnings("ignore", category=UserWarning)

NUM_ROUNDS = 100
save_path = "./results/"

with initialize(config_path="conf"):
    cfg = compose(config_name="config")

# Reset the number of number
cfg.num_rounds = NUM_ROUNDS
cfg.train.num_rounds = NUM_ROUNDS

# Create output directory
if not os.path.exists(save_path):
    os.mkdir(save_path)

# Partition dataset and get dataloaders
# We set the number of partitions to 20 for fast processing.
fds = FederatedDataset(
    dataset=cfg.dataset.name, partitioners={"train": cfg.num_clients}
)
(
    tokenizer,
    data_collator,
    formatting_prompts_func,
) = get_tokenizer_and_data_collator_and_propt_formatting(cfg.model.name)


# ClientApp for client #1 (Flower Next)
client1 = fl.client.ClientApp(
    client_fn=gen_client_fn(
        fds,
        tokenizer,
        formatting_prompts_func,
        data_collator,
        cfg.model,
        cfg.train,
        save_path,
        partition_id=0,
        api=True,
    ),
)


# ClientApp for client #2 (Flower Next)
client2 = fl.client.ClientApp(
    client_fn=gen_client_fn(
        fds,
        tokenizer,
        formatting_prompts_func,
        data_collator,
        cfg.model,
        cfg.train,
        save_path,
        partition_id=1,
        api=True,
    ),
)


# Instantiate strategy.
strategy = fl.server.strategy.FedAvg(
    min_available_clients=2,  # Simulate a 2-client setting
    fraction_fit=1.0,
    fraction_evaluate=0.0,  # no client evaluation
    on_fit_config_fn=get_on_fit_config(),
    fit_metrics_aggregation_fn=fit_weighted_average,
)

# ServerApp for Flower-Next
server = fl.server.ServerApp(
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
)
