# =====================================================================
# For a full TOML configuration guide, check the Flower docs:
# https://flower.ai/docs/framework/how-to-configure-pyproject-toml.html
# =====================================================================

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "$package_name"
version = "1.0.0"
description = ""
license = "Apache-2.0"
# Dependencies for your Flower App
dependencies = [
    "flwr[simulation]>=1.24.0",
    "flwr-datasets>=0.5.0",
    "torch==2.4.0",
    "trl==0.8.1",
    "bitsandbytes==0.45.4",
    "scipy==1.13.0",
    "peft==0.6.2",
    "transformers==4.50.3",
    "sentencepiece==0.2.0",
    "omegaconf==2.3.0",
    "hf_transfer==0.1.8",
]

[tool.hatch.build.targets.wheel]
packages = ["."]

[tool.flwr.app]
publisher = "$username"

# Point to your ServerApp and ClientApp objects
# Format: "<module>:<object>"
[tool.flwr.app.components]
serverapp = "$import_name.server_app:app"
clientapp = "$import_name.client_app:app"

# Custom config values accessible via `context.run_config`
[tool.flwr.app.config]
model.name = "mistralai/Mistral-7B-v0.3"
model.quantization = 4
model.gradient-checkpointing = true
model.lora.peft-lora-r = 32
model.lora.peft-lora-alpha = 64
train.save-every-round = 5
train.learning-rate-max = 5e-5
train.learning-rate-min = 1e-6
train.seq-length = 512
train.training-arguments.output-dir = ""
train.training-arguments.learning-rate = ""
train.training-arguments.per-device-train-batch-size = 16
train.training-arguments.gradient-accumulation-steps = 1
train.training-arguments.logging-steps = 10
train.training-arguments.num-train-epochs = 3
train.training-arguments.max-steps = 10
train.training-arguments.save-steps = 1000
train.training-arguments.save-total-limit = 10
train.training-arguments.gradient-checkpointing = true
train.training-arguments.lr-scheduler-type = "constant"
strategy.fraction-train = $fraction_train
strategy.fraction-evaluate = 0.0
num-server-rounds = 200

# Dataset config (static for FlowerTune LLM Leaderboard)
[tool.flwr.app.config.static]
dataset.name = "$dataset_name"

# Default federation to use when running the app
[tool.flwr.federations]
default = "local-simulation"

# Local simulation federation with $num_clients virtual SuperNodes
[tool.flwr.federations.local-simulation]
options.num-supernodes = $num_clients
options.backend.client-resources.num-cpus = 6
options.backend.client-resources.num-gpus = 1.0
