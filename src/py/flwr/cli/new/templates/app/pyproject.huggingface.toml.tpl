[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "$package_name"
version = "1.0.0"
description = ""
license = "Apache-2.0"
dependencies = [
    "flwr[simulation]>=1.17.0",
    "flwr-datasets>=0.5.0",
    "torch==2.5.1",
    "transformers>=4.30.0,<5.0",
    "evaluate>=0.4.0,<1.0",
    "datasets>=2.0.0, <3.0",
    "scikit-learn>=1.6.1, <2.0",
]

[tool.hatch.build.targets.wheel]
packages = ["."]

[tool.flwr.app]
publisher = "$username"

[tool.flwr.app.components]
serverapp = "$import_name.server_app:app"
clientapp = "$import_name.client_app:app"

[tool.flwr.app.config]
num-server-rounds = 3
fraction-fit = 0.5
local-epochs = 1
model-name = "prajjwal1/bert-tiny" # Set a larger model if you have access to more GPU resources
num-labels = 2

[tool.flwr.federations]
default = "localhost"

[tool.flwr.federations.localhost]
options.num-supernodes = 10

[tool.flwr.federations.localhost-gpu]
options.num-supernodes = 10
options.backend.client-resources.num-cpus = 4 # each ClientApp assumes to use 4CPUs
options.backend.client-resources.num-gpus = 0.25 # at most 4 ClientApps will run in a given GPU
