[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "$package_name"
version = "1.0.0"
description = ""
license = "Apache-2.0"
dependencies = [
    "flwr[simulation]>=1.10.0",
    "flwr-datasets>=0.3.0",
    "trl==0.8.1",
    "bitsandbytes==0.43.0",
    "scipy==1.13.0",
    "peft==0.6.2",
    "transformers==4.39.3",
    "sentencepiece==0.2.0",
    "omegaconf==2.3.0",
]

[tool.hatch.build.targets.wheel]
packages = ["."]

[tool.flwr.app]
publisher = "$username"

[tool.flwr.app.components]
serverapp = "$import_name.server_app:app"
clientapp = "$import_name.client_app:app"

[tool.flwr.app.config]
model.name = "mistralai/Mistral-7B-v0.3"
model.quantization = 4
model.gradient_checkpointing = true
model.lora.peft_lora_r = 32
model.lora.peft_lora_alpha = 64
train.save_every_round = 5
train.learning_rate_max = 5e-5
train.learning_rate_min = 1e-6
train.seq_length = 512
train.training_arguments.output_dir = ""
train.training_arguments.learning_rate = ""
train.training_arguments.per_device_train_batch_size = 16
train.training_arguments.gradient_accumulation_steps = 1
train.training_arguments.logging_steps = 10
train.training_arguments.num_train_epochs = 3
train.training_arguments.max_steps = 10
train.training_arguments.save_steps = 1000
train.training_arguments.save_total_limit = 10
train.training_arguments.gradient_checkpointing = true
train.training_arguments.lr_scheduler_type = "constant"
strategy.fraction_fit = $fraction_fit
strategy.fraction_evaluate = 0.0
num-server-rounds = 200

[tool.flwr.app.config.static]
dataset.name = "$dataset_name"

[tool.flwr.federations]
default = "local-simulation"

[tool.flwr.federations.local-simulation]
options.num-supernodes = $num_clients
options.backend.client-resources.num-cpus = 6
options.backend.client-resources.num-gpus = 1.0
