"""$project_name: A Flower / FlowerTune app."""

import os
import warnings
from datetime import datetime

from flwr_datasets import FederatedDataset
from hydra import compose, initialize
from hydra.utils import instantiate

from flwr.client import ClientApp
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from $import_name.client_app import gen_client_fn, get_parameters
from $import_name.dataset import get_tokenizer_and_data_collator_and_propt_formatting
from $import_name.models import get_model
from $import_name.server_app import fit_weighted_average, get_evaluate_fn, get_on_fit_config

# Avoid warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"

# Initialise regular config
with initialize(config_path="conf", version_base="1.1"):
    cfg = compose(config_name="config")

# Initialise static config
with initialize(config_path="conf", version_base="1.1"):
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

# ClientApp for Flower Next
client = ClientApp(
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

# Get initial model weights
init_model = get_model(cfg.model)
init_model_parameters = get_parameters(init_model)
init_model_parameters = ndarrays_to_parameters(init_model_parameters)

def server_fn(context: Context):
    # Instantiate strategy according to config. Here we pass other arguments
    # that are only defined at runtime.
    strategy = instantiate(
        cfg.strategy,
        on_fit_config_fn=get_on_fit_config(),
        fit_metrics_aggregation_fn=fit_weighted_average,
        initial_parameters=init_model_parameters,
        evaluate_fn=get_evaluate_fn(
            cfg.model, cfg.train.save_every_round, cfg_static.num_rounds, save_path
        ),
    )

    config = ServerConfig(num_rounds=cfg_static.num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# ServerApp for Flower Next
server = ServerApp(server_fn=server_fn)
