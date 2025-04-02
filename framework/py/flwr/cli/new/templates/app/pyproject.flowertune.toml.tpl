[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "$package_name"
version = "1.0.0"
description = ""
license = "Apache-2.0"
dependencies = [
    "flwr[simulation]>=1.18.0",
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

[tool.flwr.app.components]
serverapp = "$import_name.server_app:app"
clientapp = "$import_name.client_app:app"

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
strategy.fraction-fit = $fraction_fit
strategy.fraction-evaluate = 0.0
num-server-rounds = 200

[tool.flwr.app.config.static]
dataset.name = "$dataset_name"

[tool.flwr.federations]
default = "local-simulation"

[tool.flwr.federations.local-simulation]
options.num-supernodes = $num_clients
options.backend.client-resources.num-cpus = 6
options.backend.client-resources.num-gpus = 1.0
