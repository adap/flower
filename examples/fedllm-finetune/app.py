import os
import warnings
from omegaconf import DictConfig, OmegaConf

import flwr as fl
from flwr_datasets import FederatedDataset

from dataset import get_tokenizer_and_data_collator_and_propt_formatting
from client import gen_client_fn_api
from utils import get_on_fit_config, fit_weighted_average


warnings.filterwarnings("ignore", category=UserWarning)

NUM_ROUNDS = 2
save_path = "./results/"

# Define model config
model_cfg = OmegaConf.create(
    {
        "name": "openlm-research/open_llama_3b_v2",
        "quantization": 4,
        "gradient_checkpointing": True,
        "lora": {"peft_lora_r": 32, "peft_lora_alpha": 64},
    }
)
# Define training config
train_cfg = OmegaConf.create(
    {
        "num_rounds": NUM_ROUNDS,
        "save_every_round": 5,
        "learning_rate_max": 5e-5,
        "learning_rate_min": 1e-6,
        "seq_length": 512,
        "training_arguments": {
            "output_dir": None,
            "learning_rate": None,
            "per_device_train_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "logging_steps": 10,
            "num_train_epochs": 3,
            "max_steps": 10,
            "report_to": None,
            "save_steps": 1000,
            "save_total_limit": 10,
            "gradient_checkpointing": model_cfg.gradient_checkpointing,
            "lr_scheduler_type": "constant",
        },
    }
)

# Create output directory
if not os.path.exists(save_path):
    os.mkdir(save_path)

# Partition dataset and get dataloaders
# We set the number of partitions to 20 for fast processing.
fds = FederatedDataset(dataset="vicgalle/alpaca-gpt4", partitioners={"train": 20})
(
    tokenizer,
    data_collator,
    formatting_prompts_func,
) = get_tokenizer_and_data_collator_and_propt_formatting(model_cfg.name)


# ClientApp for Flower-Next
client = fl.client.ClientApp(
    client_fn=gen_client_fn_api(
        fds,
        tokenizer,
        formatting_prompts_func,
        data_collator,
        model_cfg,
        train_cfg,
        save_path,
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
