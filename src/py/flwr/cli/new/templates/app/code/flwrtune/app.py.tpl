"""$project_name: A Flower / flwrtune app."""

import os
from datetime import datetime
from hydra import compose, initialize
from hydra.utils import instantiate
from dataset import get_tokenizer_and_data_collator_and_propt_formatting

import flwr as fl
from flwr_datasets import FederatedDataset

from $import_name.client import gen_client_fn
from $import_name.server import get_on_fit_config, fit_weighted_average, get_evaluate_fn


# Initialise regular config
with initialize(config_path="conf"):
    cfg = compose(config_name="config")

# Initialise static config
with initialize(config_path="conf"):
    cfg_static = compose(config_name="static_config")

cfg.train.num_rounds = cfg_static.num_rounds

# Create output directory given current timestamp
current_time = datetime.now()
folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
save_path = os.path.join(os.getcwd(), f"results/{folder_name}")
os.makedirs(save_path, exist_ok=True)

# Partition dataset and get dataloaders
partitioner = instantiate(cfg_static.partitioner)
fds = FederatedDataset(
    dataset=cfg_static.dataset.name, partitioners={"train": partitioner}
)
(
    tokenizer,
    data_collator,
    formatting_prompts_func,
) = get_tokenizer_and_data_collator_and_propt_formatting(cfg.model.name)

# ClientApp for Flower-Next
client = fl.client.ClientApp(
    client_fn=gen_client_fn(
        fds,
        tokenizer,
        formatting_prompts_func,
        data_collator,
        cfg.model,
        cfg.train,
        save_path,
    ),
)

# Instantiate strategy according to config. Here we pass other arguments
# that are only defined at run time.
strategy = instantiate(
    cfg.strategy,
    on_fit_config_fn=get_on_fit_config(),
    fit_metrics_aggregation_fn=fit_weighted_average,
    evaluate_fn=get_evaluate_fn(
        cfg.model, cfg.train.save_every_round, cfg_static.num_rounds, save_path
    ),
)

# ServerApp for Flower-Next
server = fl.server.ServerApp(
    config=fl.server.ServerConfig(num_rounds=cfg_static.num_rounds),
    strategy=strategy,
)
