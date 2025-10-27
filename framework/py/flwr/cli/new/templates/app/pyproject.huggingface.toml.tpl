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
    "torch>=2.7.1",
    "transformers>=4.30.0,<5.0",
    "evaluate>=0.4.0,<1.0",
    "datasets>=2.0.0, <3.0",
    "scikit-learn>=1.6.1, <2.0",
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
num-server-rounds = 3
fraction-train = 0.5
local-steps = 5
model-name = "prajjwal1/bert-tiny" # Set a larger model if you have access to more GPU resources
num-labels = 2

# Default federation to use when running the app
[tool.flwr.federations]
default = "localhost"

# Local simulation federation with 10 virtual SuperNodes
[tool.flwr.federations.localhost]
options.num-supernodes = 10

# Local simulation federation with 10 virtual SuperNodes
# making use of GPUs
[tool.flwr.federations.localhost-gpu]
options.num-supernodes = 10
options.backend.client-resources.num-cpus = 4 # each ClientApp assumes to use 4CPUs
options.backend.client-resources.num-gpus = 0.25 # at most 4 ClientApps will run in a given GPU

# Remote federation example for use with SuperLink
[tool.flwr.federations.remote-federation]
address = "<SUPERLINK-ADDRESS>:<PORT>"
insecure = true  # Remove this line to enable TLS
# root-certificates = "<PATH/TO/ca.crt>"  # For TLS setup
